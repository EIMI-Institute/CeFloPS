# a more performant simulation implementation based on arrays more than objects
import matplotlib.pyplot as plt
import numpy as np
import CeFloPS.simulation.settings as settings
import CeFloPS.simulation.common.functions as f
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
import time as t
from matplotlib.pyplot import cm
import random
from CeFloPS.simulation.common.vol_sampler_triple import (
    volume_sampler_triple as volume_sampler,
)
from CeFloPS.simulation.common.sampler import SamplerCollection
from CeFloPS.logger_config import setup_logger
import logging
logger = setup_logger(__name__, level=logging.WARNING)
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import os,uuid
import pandas as pd




import pickle
import lz4.frame
from typing import Any

class CompressedBatchLog:
    def __init__(self, buffer_size=10000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.compressed_chunks = []

    def log(self, entry: Any):
        self.buffer.append(entry)
        if len(self.buffer) >= self.buffer_size:
            self._flush()

    def _flush(self):
        if self.buffer:
            packed = pickle.dumps(self.buffer, protocol=pickle.HIGHEST_PROTOCOL)
            compressed = lz4.frame.compress(packed)
            self.compressed_chunks.append(compressed)
            self.buffer.clear()

    def finalize(self):
        """Flush remaining uncompressed buffer into compressed list."""
        self._flush()

    def get_all(self):
        """Returns the full, decompressed log as a list of entries."""
        self.finalize()
        all_entries = []
        for chunk in self.compressed_chunks:
            unpacked = pickle.loads(lz4.frame.decompress(chunk))
            all_entries.extend(unpacked)
        return all_entries

    def get_chunks(self):
        """Returns list of compressed byte chunks (send for IPC, storage, ...)."""
        self.finalize()
        return self.compressed_chunks.copy()

    @staticmethod
    def decompress_chunks(chunk_list):
        """Static helper: decompress+unpickle list of chunks into one event list."""
        out = []
        for chunk in chunk_list:
            out.extend(pickle.loads(lz4.frame.decompress(chunk)))
        return out



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log_none(frame, elem, prev_info):
    if elem.new is None:  # Use `elem.new` for the new value
        name = elem.alias if elem.alias else "variable"
        print(f"Variable '{name}' set to None at:")
        import traceback
        traceback.print_stack()
        assert False  # Halt execution


def update_ready_mask(ready_mask, indices_to_update, times,cell_states, max_time):
    # Set cells to alive only if their time < max_time and >= 0
    ready_mask[indices_to_update] = (cell_states[indices_to_update] != 2) & (times[indices_to_update] < max_time)
    return ready_mask
def compute_path_tangents(path):
    """
    path: ndarray of shape (n, 3)
    Returns:
        tangents: ndarray of shape (n, 3), normalized
    """
    if len(path)==1:
        return None
    path = np.asarray(path)
    n = path.shape[0]
    tangents = np.zeros_like(path)
    # Central differences for interior
    tangents[1:-1] = path[2:] - path[:-2]
    # Forward/backward for endpoints
    tangents[0] = path[1] - path[0]
    tangents[-1] = path[-1] - path[-2]
    # Normalize (avoid division by zero)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    tangents = tangents / norms
    return tangents
 
class Simulation:
    def __init__(
        self,
        cell_count,
        initial_distribution,
        rois,
        roi_regionsmapping,
        interval,
        concentrations,
        state_machine,
        stop_condition=settings.stop_condition,
        roi_mapping_str=None,
        loggingcount=settings.CELLS_WITH_PATH_LOGGING,
        seed=0,
        mode="parallel",
        store_dir=None
    ):
        #initialise seed for randomness of events
        np.random.seed(seed)
        random.seed(seed)

        self.directory=store_dir
        class KNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Linear(1, 48), nn.Tanh(),
                    nn.Linear(48, 48), nn.Tanh(),
                    nn.Linear(48, 2),
                    nn.Softplus()
                )
            def forward(self, t):
                t = t.view(-1,1)
                return self.model(t)

        k_net = KNet()
        class PKODE(nn.Module):
            def __init__(self, k_net, K1, K2, K3, n_organs):
                super().__init__()
                self.k_net = k_net
                self.register_buffer("K1", torch.tensor(K1, dtype=torch.float32, device=device))
                self.register_buffer("K2", torch.tensor(K2, dtype=torch.float32, device=device))
                self.register_buffer("K3", torch.tensor(K3, dtype=torch.float32, device=device))
                self.n_organs = n_organs
            def forward(self, t, y):
                rates = self.k_net(t.view(1))  # shape (1,2)
                kinj = rates[0,0]
                kelim = rates[0,1]
                dy = torch.zeros_like(y)
                dy[0] = -kinj * y[0]
                dy[1] = kinj * y[0] - kelim * y[1]
                dy[2] = kelim * y[1]
                for i in range(self.n_organs):
                    region_idx = 3+2*i
                    trap_idx = 3+2*i+1
                    dy[1]    -= self.K1[i] * y[1]
                    dy[region_idx] += self.K1[i] * y[1]
                    dy[region_idx] -= (self.K2[i]+self.K3[i])*y[region_idx]
                    dy[1]    += self.K2[i]*y[region_idx]
                    dy[trap_idx] += self.K3[i]*y[region_idx]
                return dy



        TOTAL_NUMBER_OF_CELLS=cell_count
        self.cellcount=cell_count
        self.thetas = np.random.uniform(0, 2*np.pi, size=TOTAL_NUMBER_OF_CELLS)
        self.roi_mapping_str = roi_mapping_str
        distributor = initial_distribution  # rois, array len rois OR "random"
        self.artery_nearest_object_cache = dict()  # cache for veinend calculation
        self.interval = interval
        self.state_machine = state_machine
        self.position_log = CompressedBatchLog(buffer_size=10000)
        self.tag_log = CompressedBatchLog(buffer_size=10000)
        self.region_log = CompressedBatchLog(buffer_size=10000)
        self.state_log = CompressedBatchLog(buffer_size=10000) 

        self.rois = rois
        self.blood_roi = [roi for roi in rois if roi.name == "blood"][0]
        self.vessel_list = self.blood_roi.geometry
        for v in self.vessel_list:
            v.register_volumes()

        #create vectors for vessels
        self.vessel_profiles_dict = {}
        self.vessel_rs_dict = {}
        for vessel in self.blood_roi.geometry:
            for volume in vessel.volumes:
                rs = vessel.profiles[(volume.id, "rs")]
                # check if rs prob cdf exists, else create it
                if (volume.id, "rs_prob") in vessel.profiles:
                    rs_prob_cdf = np.array(vessel.profiles[(volume.id, "rs_prob")])
                    rs_prob_pdf = np.diff(rs_prob_cdf, prepend=0)
                    rs_prob_pdf = rs_prob_pdf / np.sum(rs_prob_pdf)
                else: 
                    rs_prob_pdf = np.array(rs) / np.sum(rs)
                self.vessel_profiles_dict[volume.id] = {
                    'v_r': vessel.profiles[(volume.id, "v_r")],
                    'rs': rs,
                    'rs_prob_pdf': rs_prob_pdf,
                    "normals":  compute_path_tangents(vessel.path)
                }
                self.vessel_rs_dict[volume.id] = vessel.profiles[(volume.id, "rs")]

        #print(self.vessel_profiles_dict)
        # -------------- create Cells ---------------------

        N = TOTAL_NUMBER_OF_CELLS
        TICK_TIME=100
        self.TICK_TIME=1000
        # the comments reference the dtype choice!
        self.current_positions = np.zeros((N, 3), dtype=np.float32)# most often between 0 and 300 per value
        self.current_times     = np.zeros(N, dtype=np.float32)#up to a few thousand floating point values, also set to -1 for signaling
        self.reference_indices = np.zeros((N,), dtype=np.int32)#between 0 and high thousands, most is probably 8 million
        self.last_tags= np.full(N, -1, dtype=np.int16)#probably -1 to a few thousands, <10k
        self.tag_str_to_int = {}
        self.tag_int_to_str = []
        self.rad_indices=np.zeros((N,), dtype=np.uint16)#0 to 1000
        self.last_update_times = np.zeros(N, dtype=np.float32)#same as the times above but not negative
        if settings.rate_time_start is not None:
            self.last_update_times[:] = settings.rate_time_start
        else:
            self.last_update_times[:] = self.current_times[:]
        self.tracked_mask = np.ones(N, dtype=bool)#boolean mask
        self.time_until_tick = np.full(N, TICK_TIME, dtype=np.float32)#same as other times 




        self.vessel_obj_to_id = {v: i for i, v in enumerate(self.vessel_list)}
        self.vessel_id_to_obj = {i: v for i, v in enumerate(self.vessel_list)}

        #store vector of vessel and voi ids per cell
        self.current_vessel_ids = np.full(N, -1, dtype=np.int16)#-1 to a few thousand
        #for regions/vois
        # lookup dicts
        self.region_name_to_id = {roi.name: i for i, roi in enumerate(rois)}
        self.region_id_to_roi = {i: roi for i, roi in enumerate(rois)}
        self.blood_region_id = self.region_name_to_id["blood"]
        #assert False, len(list(self.region_name_to_id.values()))
        # -1 to a few hundred (regions/vois)
        self.current_regions = np.full(N, -1, dtype=np.int16)
        self.vessel_id_to_volume_id_map={i: v.volumes[0].id for i,v in zip(range(len(self.vessel_list)),self.vessel_list)}



        for i, (position, roi) in enumerate(distributor.get_distribution(
                    larm=settings.ARMINJECTION, rleg=settings.LEGINJECTION
                )):
            structure, idx = position

            # Map structure/roi to region integer ID
            region_id = self.region_name_to_id[roi.name]
            self.current_regions[i] = region_id

            self.reference_indices[i] = idx




            # For position: get the actual xyz coordinate according to what structure is
            if hasattr(structure, 'path'):  # it's a vessel
                vessel_id = self.vessel_obj_to_id[structure]
                self.current_vessel_ids[i] = vessel_id
                p = structure.path[idx]


                volume_id = self.vessel_id_to_volume_id_map[self.vessel_obj_to_id[structure]]
                vessel_profile = self.vessel_profiles_dict[volume_id]
                rs_len = len(vessel_profile['rs'])
                rs_prob_pdf = vessel_profile['rs_prob_pdf']
                self.rad_indices[i] = random.choices(
                    range(rs_len),
                    weights=rs_prob_pdf,
                    k=1
                )[0]



                p = structure.path[idx]
            else: # it's a VOI/geometry
                p = structure.get_points()[idx]
                self.current_vessel_ids[i] = -1

            self.current_positions[i] = p
            if i >= N-1:
                break


        self.last_tags = np.full(self.cellcount, -1, dtype=np.int16)       
        self.last_regions = np.full(self.cellcount, -1, dtype=np.int16)


        # -------------- load compartment model rates ---------------------

        self.organs = ['muscle', 'lung', 'liver', 'grey_matter', 'all', 'myocardium', 'spleen', 'guc_lesions',
          'cortex', 'whitematter', 'cerebellum', 'thyroid', 'pancreas', 'kidney'  ]
        self.K1 = np.array([0.026, 0.023, 0.660, 0.107, 0.553, 0.832, 1.593, 0.022,
            0.0896, 0.0337, 0.1318, 0.9663, 0.3561, 0.7023])
        self.K2 = np.array([0.249, 0.205, 0.765, 0.165, 1.213, 2.651, 2.867, 0.296,
            0.2532, 0.1347, 0.6280, 4.6042, 1.7077, 1.3542])
        self.K3 = np.array([0.016, 0.001, 0.002, 0.067, 0.039, 0.099, 0.006, 0.771,
            0.2213, 0.0482, 0.1870, 0.0748, 0.0787, 0.1778])
        #TODO dynamique!
        self.n_organs = len(self.organs)
        self.pkode = PKODE(k_net, self.K1, self.K2, self.K3, self.n_organs).to(device)
        model_path = settings.model_path

        
        self.region_Q_idx_mapping = {}
        self.region_id_to_organ_idx = {}

        blood_region_id = self.region_name_to_id["blood"]
        self.region_Q_idx_mapping[blood_region_id] = 1

        upper_organs=[s.upper() for s in self.organs]
        for roi in self.rois[:-1]:
            found=False
            for k,v in self.roi_mapping_str.items():
                if roi.name in v:
                    for i,oname in enumerate(upper_organs):
                        for i, oname in enumerate(upper_organs):
                            if k.upper() in oname:
                                region_id = self.region_name_to_id[roi.name]
                                q_idx = 3 + 2 * i    
                                self.region_Q_idx_mapping[region_id] = q_idx
                                self.region_id_to_organ_idx[region_id] = i
                                print(oname)

                    print("Found",roi.name,k)
                    if "Heart" in k:
                        region_id = self.region_name_to_id[roi.name]
                        self.region_Q_idx_mapping[region_id]=1

                        print(region_id)
                        #assert any([k.upper() in s for s in upper_organs])
                    found=True
            if not found:
                assert False, roi.name
        self.fraction_times =  [2.20981910e-06, 5.70305698e-04, 1.23035137e-04, 0.00000000e+00,
                                3.54260135e-05, 2.18537197e-05, 5.70073850e-05, 1.90444012e-05,
                                0.00000000e+00, 6.78705182e-05, 1.79546299e-04, 2.06231573e-05,
                                6.75859269e-06, 0.00000000e+00, 3.72396618e-06, 2.11478113e-05,
                                8.18161607e-05, 7.42289661e-06, 3.61815014e-07, 5.94930282e-07,
                                1.18427505e-05, 3.42299288e-06, 9.45291128e-06, 4.06863644e-05,
                                3.03127623e-02, 0.00000000e+00, 1.75437931e-02, 4.96218017e-05,
                                1.02337250e-04, 1.46044924e-03, 5.79019118e-04, 3.17650322e-03,
                                1.29189696e-02, 7.41644139e-04, 4.83794030e-05, 3.84663329e-04,
                                2.01025008e-05, 1.75215887e-04, 2.61788951e-04, 1.87792354e-04,
                                2.02476039e-02, 1.84284175e-02, 1.34948081e-02, 8.78581975e-01]

 
        self.organ_fractions = np.zeros(self.n_organs)
        for reg_id, frac in enumerate(self.fraction_times):
            organ_idx = self.region_id_to_organ_idx.get(reg_id, None)
            if organ_idx is not None:
                self.organ_fractions[organ_idx] += frac
        #assert False
        if torch.cuda.is_available():
            map_location = torch.device('cuda')
        else:
            map_location = torch.device('cpu')
        print("Mapping model to ", map_location)
        self.pkode.load_state_dict(torch.load(model_path, map_location=map_location))
        self.pkode.load_state_dict(torch.load(model_path, map_location=map_location))

        dt_min = 0.1
        TIME_LIMIT_MIN = settings.TIME_LIMIT_S / 60.0
        t_pre = np.arange(0, TIME_LIMIT_MIN + dt_min, dt_min)
        with torch.no_grad():
            k_pred = self.pkode.k_net(torch.from_numpy(t_pre).float().unsqueeze(1).to(device)).cpu().numpy()  # [N, 2]
        self.k_inj_time = k_pred[:,0]     # [N,]
        self.k_elim_time = k_pred[:,1]    # [N,]
        self.t_pre = t_pre
        if settings.SET_ELIM_ZERO:
            self.k_elim_time[:] = 0

        #set injection times according to the model, treating k_inj as a cumulative function and using inverse transform sampling
        cdf = np.cumsum(self.k_inj_time)
        cdf = cdf / cdf[-1]
        us = np.random.rand(self.cellcount)
        indices = np.searchsorted(cdf, us)
        self.current_times = self.t_pre[indices]
 
        # self.k_inj_time: predicted injection rates for points in self.t_pre
        # self.current_times: shape (cellcount,) - die per Inverse-Sampling gezogenen Injektionszeiten

        """ plt.figure(figsize=(10,5))

        # Plot predicted injection rate (k_inj_time)
        plt.subplot(2,1,1)
        plt.plot(self.t_pre, self.k_inj_time, label="Predicted k_inj(t)")
        plt.xlabel("Zeit t (min)")
        plt.ylabel("k_inj (Rate)")
        plt.title("Vorhergesagte Injektionsrate über der Zeit")
        plt.legend()

        # Plot histogram of actual sampled times
        plt.subplot(2,1,2)
        plt.hist(self.current_times, bins=50, density=True, alpha=0.7, color='tab:blue')
        plt.xlabel("Tatsächliche Injektionzeit von Zellen")
        plt.ylabel("Dichte / Normiertes Histogramm")
        plt.title("Histogramm der gesampelten Injektionszeiten")

        plt.tight_layout()
        plt.show()

        assert False """
        # cellstate as in the compartmentconcentrationmodel for fast transition matrix
        # Build mapping
        region_names = [roi.name for roi in rois]
        region_name_to_id = {roi.name: i for i, roi in enumerate(rois)}
        blood_region_id = region_name_to_id['blood']
        n_organs = len(region_names) - 1  # not counting blood

        region_id_to_stateidx = {}
        for i, name in enumerate(region_names):
            region_id = region_name_to_id[name]
            if name == 'blood':
                region_id_to_stateidx[region_id] = 1
            else:
                # i-1 because blood is at i=0
                region_id_to_stateidx[region_id] = 3 + 2 * (i-1)

        # vectorized mapping
        max_region_id = max(region_id_to_stateidx)
        self.regionid2cellstate = np.full(max_region_id+1, -1, dtype=np.int32)
        for regid, stateidx in region_id_to_stateidx.items():
            self.regionid2cellstate[regid] = stateidx

        # assign initial states
        self.cell_states = self.regionid2cellstate[self.current_regions]
        self.last_states = self.regionid2cellstate[self.current_regions]

        #check if there is already a model trained on that and load it or train one if not
        #load_or_compute_rate_functions(voi_combination)



        #calculate expected fraction of eliminated cells at the end of simulation
        #TODO based on occuring delays in simulation, set the elimination fraction at end
        eliminated_fraction_at_end=0.2
        buffer_against_randomness=0.1   #log 10 percent more
        updated_loggingcount=loggingcount+(eliminated_fraction_at_end+buffer_against_randomness)*loggingcount

        #set tracking array compensating for expected simulation ending time loss
        for i in range(len(self.current_times)):
            if i >= updated_loggingcount:
                 self.tracked_mask[i] = False  # set to not track those cells that are not needed to get loggincount many logged not eliminated cells



        #check validity for given parameters
        assert "blood" in [roi.name for roi in rois], "We need vessels for the simulation"
        for roi in rois:
                if "blood" not in roi.name:
                    assert roi.geometry!=None
        for roi in rois[:-1]:assert not any([type(x)==list for x in roi.compartment_model.C_outs])

        #check vessel connection validity
        for vessel in self.blood_roi.geometry:
            vessel.path = list(vessel.path)
            vessel.times = list(
                vessel.times
            )  # convert to list for faster append operations
            for l in vessel.links_to_vois:
                assert l.target_tissue in rois, vessel.associated_vesselname
                assert l.target_tissue.geometry!=None

        self.simulation_location_sampler = None
        self.simulation_location_sampler = SamplerCollection(
            ["vein", "artery", "vois"], stepsize=0.2
        )
        self.volume_samplers = []  # TODO create them again with their correct ID
        logging_vessels = []
        entry_indices = []
        exit_indices = []
        for sampler in self.volume_samplers:
            logging_vessels.append(sampler.entry_object)
            entry_indices.append(sampler.entry_index)
            exit_indices.append(sampler.exit_index)
        self.logging_vessels = set(logging_vessels)
        self.entry_indices = set(entry_indices)
        self.exit_indices = set(exit_indices)
        count = 0
        #tagcodes for efficient storage
        for tag in ["VB", "B",'beating_heart.systole.pericardium'] + [r.name for r in self.rois]:
            if tag not in self.tag_str_to_int:
                code = len(self.tag_str_to_int)
                self.tag_str_to_int[tag] = code
                self.tag_int_to_str.append(tag)

        self.tag_codes = np.array(list(self.tag_str_to_int.values()), dtype=np.uint16)


        self.time = 0.0
        self.sim_concentration = dict()  # roi concenctrations in cells per ml tissue
        self.untracked_cells = []
        self.stop_condition = stop_condition
        self.mapping = roi_regionsmapping
        self.iterations = 0
        for roi in self.rois:
            if roi.name != "blood":
                volumes = 0
                for string, values in roi_mapping_str.items():
                    if roi.name in values:
                        volumes = sum([r.volume for r in self.rois if r.name in values])
                        # logger.print("roi name in mapping")
                logger.print(roi.name, roi.volume, volumes)
                if volumes == 0:
                    volumes = (
                        roi.volume
                    )  # TODO check if better option exists, tracking integrated vois
                roi.set_target_fraction(roi.volume / volumes)
            else:
                if "blood" in roi.name:
                    for vessel in roi.geometry:
                        vessel.register_volumes()  # TODO force?
                    for vessel in roi.geometry:
                        for v in vessel.profiles.values():
                            assert v!=None
        self.track_compartment_plotter = []
        for roi in self.rois:
            if roi.name != "blood":
                self.track_compartment_plotter += roi.compartment_plotter
            if mode == "sequential":
                self.sim_concentration[roi.name] = roi.get_concentration()
            # roi.update_concentration_share(self)  # init concentration share
        # init time
        self.roi_speeds = np.zeros(len(self.rois))
        for region_id, roi in self.region_id_to_roi.items():
            if roi != self.blood_roi:
                self.roi_speeds[region_id] = roi.speed
        self.prev_time = 0

        if mode == "sequential":
            self.log = dict()
            self.log["time"] = [self.time]
            for roi in self.rois:
                self.log[roi] = [self.sim_concentration[roi.name]]

        self.end_veins = [
            vessel
            for vessel in self.blood_roi.geometry
            if vessel.type == "vein"
            and len([link for link in vessel.links_to_path if link.source_index == 0])
            == 0
        ]
        logger.print("Number of veinentries: ", len(self.end_veins))
        logger.print("Precaching:")

        # self.pre_cache()
        logger.print("Done")
        # logger.print(self.artery_nearest_object_cache)

    def run_parallel(self, signal_int, queue, event, barrier):
        """
        Run simulation in array-centric, vectorized manner
        """


        #Initial log of current state
        for cell_idx in range(self.cellcount):
            if self.current_vessel_ids[cell_idx]!=-1:
                tag=self.vessel_id_to_obj[self.current_vessel_ids[cell_idx]].node_tags[self.reference_indices[cell_idx]]
            else:
                tag = simulation.region_id_to_roi[simulation.current_regions[cell_idx]].name
            tag_code = self.tag_to_int(tag)
            self.tag_log.log(
                (cell_idx, self.current_times[cell_idx], tag_code)
            )
            self.last_tags[cell_idx] = tag_code
            if self.tracked_mask[cell_idx]:
                self.position_log.log(
                    (cell_idx, self.current_times[cell_idx], *self.current_positions[cell_idx])
                )
            region_id = self.current_regions[cell_idx]
            self.region_log.log(
                (cell_idx, self.current_times[cell_idx], region_id)
            )
            self.last_regions[cell_idx] = region_id 

            self.last_states[cell_idx]=self.cell_states[cell_idx]
            self.state_log.log(
                (cell_idx, self.current_times[cell_idx],self.cell_states[cell_idx] )
            )

        self.alive_mask = self.current_times >= 0  # or another status indicator if one wants to track an amount that entered certain states etc

        while np.any(self.alive_mask):
            indices = np.where(self.alive_mask)[0] 
            self.state_machine.t_cellstate_vectorized(indices, self) 

            # Update the alive_mask according to new times/state
            update_ready_mask(self.alive_mask, indices, self.current_times,self.cell_states, settings.TIME_LIMIT_S)
            #print("Upd:",len(indices),max(self.current_times),min(self.current_times))
            #if len(indices)<=2: print(min(self.current_times),self.current_times[indices],self.current_regions[indices],self.cell_states[indices],self.current_vessel_ids[indices])
            if settings.SYNCHRONIZE:
                assert False, "No more synchronization support"
            #   queue.put(np.mean(self.current_times) / settings.SIMULATION_TIME)

            self.iterations += 1  # or time-based
            times_alive = self.current_times[self.current_times >= 0]
            if len(times_alive) == 0 or np.all(np.isnan(times_alive)):
                mean_alive = settings.TIME_LIMIT_S / 100   # fallback value
            else:
                mean_alive = np.nanmean(times_alive)

            percent_done = int(100 * mean_alive / settings.TIME_LIMIT_S)
            percent_done = max(0, min(100, percent_done))
            signal_int.value = percent_done
            #signal_int.value=int((settings.TIME_LIMIT_S/np.mean(self.current_times))*100)
        print("reached end")
        # Finish: prepare results for outside process
        # TODO Gather only tracked cells for logging if needed

        c1_indices = set([3 + 2*i for i in range(self.n_organs)])
        c2_indices = set([4 + 2*i for i in range(self.n_organs)])
        # save_per_cell_logs(self, self.directory, c1_indices, c2_indices)

        # print_log_as_dataframe(self)
        #df = build_log_dataframe(self)
        #plot_bound_fractions(df, c1_indices, c2_indices)
        #save_tracks_not_eliminated(df, self, self.directory, c1_indices, c2_indices)
        #plot_occupancy_stack(df)
        #plot_occupancy_stack_alive_only(df)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.end_of_worker(queue,self.directory)
        res = {
            "tag_log": self.tag_log.get_chunks(),           # List of LZ4-compressed chunks
            "position_log": self.position_log.get_chunks(),
            "region_log": self.region_log.get_chunks(),
            "state_log": self.state_log.get_chunks()
        }


        signal_int.value = -5

        logger.info("Simulation ended")





    def end_of_worker(self, queue, outdir):
        # examples for decompressing logs
        tag_entries = CompressedBatchLog.decompress_chunks(self.tag_log.get_chunks())
        pos_entries = CompressedBatchLog.decompress_chunks(self.position_log.get_chunks())
        region_entries = CompressedBatchLog.decompress_chunks(self.region_log.get_chunks())
        state_entries = CompressedBatchLog.decompress_chunks(self.state_log.get_chunks())

        res = {
            "tag_log": write_log_to_parquet(tag_entries, ["cell_id", "time", "tag"], outdir, "tag"),
            "position_log": write_log_to_parquet(pos_entries, ["cell_id", "time", "x", "y", "z"], outdir, "position"),
            "region_log": write_log_to_parquet(region_entries, ["cell_id", "time", "region"], outdir, "region"),
            "state_log": write_log_to_parquet(state_entries, ["cell_id", "time", "state"], outdir, "state"),
        }
        # Free memory immediately
        del tag_entries, pos_entries, region_entries, state_entries

        print("About to put result with parquet filepaths on queue")
        queue.put((res, 1))
        print("Put result on queue; worker done")

    def vectorized_compartment_sampling(self,sell_idx):
        #check which regions the cell is connected to and check against that as well as the elimination compartment using the full transition matrix of the model with the trained input and extract.
        pass
     
    def tag_to_int(self, tag):
        """Get the code for a tag string only if it is in tag_codes."""
        code = self.tag_str_to_int[tag]
        if code in self.tag_codes:
            return code
        raise ValueError(f"Tag '{tag}' not in tag_codes.")

    def int_to_tag(self, code):
        """Get the tag string only if code is in tag_codes."""
        if code in self.tag_codes:
            return self.tag_int_to_str[code]
        raise ValueError(f"Tag code '{code}' not in tag_codes.")
def write_log_to_parquet(log_entries, columns, outdir, logname):
    if log_entries:
        df = pd.DataFrame(log_entries, columns=columns)
        outpath = os.path.join(outdir, f"log_{logname}_{uuid.uuid4().hex}.parquet")
        df.to_parquet(outpath, compression="brotli")
        return outpath
    else:
        return None
def build_log_dataframe(simulation):
    arr = np.array(list(simulation.global_cell_log), dtype=object)
    if arr.shape[0] == 0:
        print("No logs to show!")
        return pd.DataFrame()
    pos = np.stack(arr[:,2])
    df = pd.DataFrame({
        "cell_id": arr[:,0].astype(np.uint32),
        "time": arr[:,1].astype(np.float32),
        "x": pos[:,0],
        "y": pos[:,1],
        "z": pos[:,2],
        "region_id": arr[:,3].astype(np.uint8),
        "tag": arr[:,4].astype(np.uint8),
        "state": arr[:,5].astype(np.uint16),
        "changed": arr[:,5].astype(np.bool_)
    })
    # Map region ID to name:
    region_id_to_name = {v:k for k,v in simulation.region_name_to_id.items()}
    #df["region_name"] = df["region_id"].map(region_id_to_name)
    return df

    # Or, export for further processing in Excel, CSV, etc.:
    # df.to_csv("full_cell_log.csv", index=False)
def plot_bound_fractions(df, c1_indices, c2_indices,output_dir=None):
    times = np.sort(df.loc[df.time >= 0, "time"].unique())  # times only up to elimination
    region_names = sorted(df.region_name.unique())
    # We will plot: for each region, fraction of cells that are in C1 or C2
    fractions_c1 = []
    fractions_c2 = []
    for t in times:
        at_time = df[(df.time == t) & (df.time >=0)]
        total = at_time.shape[0]
        # C1:
        c1 = at_time[at_time.state.isin(c1_indices)]
        c2 = at_time[at_time.state.isin(c2_indices)]
        fractions_c1.append(len(c1)/total if total else 0)
        fractions_c2.append(len(c2)/total if total else 0)
    import matplotlib.pyplot as plt
    plt.plot(times, fractions_c1, label="Fraction C1 (bound)")
    plt.plot(times, fractions_c2, label="Fraction C2 (trapped)")
    plt.xlabel("Time")
    plt.ylabel("Fraction")
    plt.title("Fraction of Cells per State over Time")
    plt.legend()
    plt.tight_layout()
    if output_dir is not None:
        path = os.path.join(output_dir, "fraction_bound_over_time.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print("Plot saved to", path)
    #plt.show()
def save_tracks_not_eliminated(df, simulation, output_dir, c1_indices, c2_indices):
    # time == -1 signals elimination
    alive_ids = df.groupby('cell_id').filter(lambda x: (x['time'] >= 0).all())['cell_id'].unique()
    for cell in alive_ids:
        track = df[(df.cell_id == cell) & (df.time >= 0)].sort_values("time")
        # create "s" series (compartment-logic, as in your code):
        ss = np.zeros(len(track), dtype=int)
        for i, region in enumerate(track.region_id):
            if region == simulation.blood_region_id:
                ss[i] = 0
            elif region in c1_indices:
                ss[i] = 1
            elif region in c2_indices:
                ss[i] = 2
            else:
                ss[i] = -1
        out = np.column_stack([track[["x","y","z"]].values, track["time"].values, ss])
        out_path = os.path.join(output_dir, f"cell_{int(cell)}.txt")
        np.savetxt(out_path, out, fmt="%.6f %.6f %.6f %.6f %d")



def plot_occupancy_stack(df, tag_int_to_str, output_dir=None):
    """
    Stackplot: count for each tag at each time interval how many cells are there, including 'eliminated' (tag -1, black).
    Each cell is *exclusively* at each time in exactly one region: injection, a tag or eliminated.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    df = df.sort_values(['cell_id', 'time'])
    df = df[df.time >= 0]
    valid_times = np.sort(df["time"].unique())
    if len(valid_times) == 0:
        print("No times for occupancy plot.")
        return
    tag_codes = sorted([c for c in df["tag"].unique() if c != -1])
    eliminated_present = (df["tag"] == -1).any()

    def code_to_tag(code):
        code_int = int(code)
        if 0 <= code_int < len(tag_int_to_str):
            return tag_int_to_str[code_int]
        elif code_int == -1:
            return "eliminated"
        else:
            print(f"WARNING: tag code {code_int} out of bounds for tag_int_to_str of length {len(tag_int_to_str)}; setting as UNK_{code_int}")
            return f"UNK_{code_int}"

    tags_sorted = [code_to_tag(code) for code in tag_codes]
    stack_keys = ['injection'] + tags_sorted + ['eliminated']
    tag2idx = {k: i for i, k in enumerate(stack_keys)}
    occupancy = np.zeros((len(stack_keys), len(valid_times)), dtype=int)

    grouped = df.groupby("cell_id")
    for cid, group in grouped:
        group = group.sort_values('time')
        times = group["time"].values
        tags = [code_to_tag(tag) if isinstance(tag, (int, np.integer)) else tag for tag in group["tag"].values]
        # Build filtered_events [(time, tag)] only when tag changes
        change_indices = [0] + [i+1 for i in range(len(tags)-1) if tags[i] != tags[i+1]]
        filtered_events = [(times[i], tags[i]) for i in change_indices]
        min_time = valid_times[0]
        max_time = valid_times[-1]
        was_eliminated = (filtered_events and filtered_events[-1][1] == "eliminated")
        elim_time = filtered_events[-1][0] if was_eliminated else None

        if filtered_events:
            first_ev_time = filtered_events[0][0]
            if first_ev_time > min_time:
                start_idx = 0
                end_idx = np.searchsorted(valid_times, first_ev_time, side="left")
                occupancy[tag2idx['injection'], start_idx:end_idx] += 1
        else:
            occupancy[tag2idx['injection'], :] += 1

        for i, (start, tag) in enumerate(filtered_events):
            if tag == "eliminated":
                continue  # handled after
            if i+1 < len(filtered_events):
                end = filtered_events[i+1][0]
            elif was_eliminated:
                end = elim_time
            else:
                end = max_time + 1e-6
            start_idx = np.searchsorted(valid_times, start, side='left')
            end_idx = np.searchsorted(valid_times, end, side='left')
            occupancy[tag2idx[tag], start_idx:end_idx] += 1

        if was_eliminated:
            elim_idx = np.searchsorted(valid_times, elim_time, side='left')
            occupancy[tag2idx["eliminated"], elim_idx:] += 1

    neongreen = [0.0, 1.0, 0.196, 1.0]
    base_colors = list(plt.cm.tab20(np.linspace(0, 1, len(tags_sorted))))
    colors = [neongreen] + base_colors + [(0,0,0,1)]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(valid_times, occupancy, labels=stack_keys, step='post', alpha=0.85, colors=colors)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cells in tag")
    ax.set_title("Occupancy (all cells) Over Time (injection=neongreen, eliminated=black)")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.tight_layout()
    if output_dir is not None:
        plot_path = os.path.join(output_dir, "occupancy_stackplot_filled.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print("Plot saved to ", plot_path)
    else:
        plt.show()
def plot_occupancy_stack_alive_only(df, tag_int_to_str, output_dir=None):
    """
    Stackplot: For each time interval, count how many cells are in each tag state.
    Plots only cells that NEVER had tag == -1; does NOT plot an 'eliminated' compartment.
    Each cell is exclusively in one region (or 'injection') at each time.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # Find all cell_ids that ever had tag == -1, and EXCLUDE them
    eliminated_ids = set(df.loc[df["tag"] == -1, "cell_id"].unique())
    live_df = df[(df.time >= 0) & (~df["cell_id"].isin(eliminated_ids))]
    if live_df.empty:
        print("No alive cells to plot. Skipping stackplot.")
        return

    valid_times = np.sort(live_df["time"].unique())
    if len(valid_times) == 0:
        print("No time points to plot. Skipping stackplot.")
        return

    tag_codes = sorted(live_df["tag"].unique())

    def code_to_tag(code):
        code_int = int(code)
        if 0 <= code_int < len(tag_int_to_str):
            return tag_int_to_str[code_int]
        else:
            print(f"WARNING: tag code {code_int} out of bounds for tag_int_to_str; setting as UNK_{code_int}")
            return f"UNK_{code_int}"

    tags_sorted = [code_to_tag(code) for code in tag_codes]
    if len(tags_sorted) == 0:
        print("No tags found for occupancy curve. Skipping stackplot.")
        return

    stack_keys = ["injection"] + tags_sorted
    tag2idx = {k: i for i, k in enumerate(stack_keys)}
    occupancy = {key: np.zeros(len(valid_times), dtype=int) for key in stack_keys}

    grouped = live_df.groupby("cell_id")
    for cid, group in grouped:
        group_sorted = group.sort_values("time")
        times = group_sorted["time"].values
        tags = [
            code_to_tag(tag)
            if isinstance(tag, (int, np.integer)) else tag
            for tag in group_sorted["tag"].values
        ]
        change_indices = [0] + [i+1 for i in range(len(tags)-1) if tags[i] != tags[i+1]]
        filtered_events = [(times[i], tags[i]) for i in change_indices]
        min_time = valid_times[0]
        max_time = valid_times[-1]
        if filtered_events:
            first_ev_time = filtered_events[0][0]
            if first_ev_time > min_time:
                start_idx = 0
                end_idx = np.searchsorted(valid_times, first_ev_time, side="left")
                occupancy['injection'][start_idx:end_idx] += 1
        else:
            occupancy['injection'][:] += 1
        for i, (start, tag) in enumerate(filtered_events):
            if i+1 < len(filtered_events):
                end = filtered_events[i+1][0]
            else:
                end = max_time + 1e-6
            start_idx = np.searchsorted(valid_times, start, side='left')
            end_idx = np.searchsorted(valid_times, end, side='left')
            occupancy[tag][start_idx:end_idx] += 1

    neongreen = [0.0, 1.0, 0.196, 1.0]
    base_colors = list(plt.cm.tab20(np.linspace(0, 1, len(tags_sorted))))
    colors = [neongreen] + base_colors

    stack = np.vstack([occupancy[k] for k in stack_keys])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(valid_times, stack, labels=stack_keys, step='post', alpha=0.85, colors=colors)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cells in tag")
    ax.set_title("Occupancy (tag) Over Time (only non-eliminated cells, injection=neongreen)")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.tight_layout()
    if output_dir is not None:
        plot_path = os.path.join(output_dir, "occupancy_stackplot_alive.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print("Plot saved to ", plot_path)
    else:
        plt.show()
def vectorized_resample_radii(self, cell_indices):
    """
    Resample (draw new) radial indices for a set of cells from their vessel volume radius CDFs.
    """
    for i in cell_indices:
        vessel = self.current_vessels[i]
        idx = self.reference_indices[i]
        if len(vessel.profiles) == 0:
            self.rad_indices[i] = 0
            continue
        vol = vessel.get_volume_by_index(idx)
        rs_probs = vessel.profiles[(vol.id, "rs_prob")]
        num_r = len(rs_probs)
        new_r = random.choices(
            range(num_r),
            cum_weights=rs_probs,
            k=1
        )[0]
        self.rad_indices[i] = new_r
