import CeFloPS.simulation.common.functions as f
from .shared_geometry import SharedGeometry, SharedGeometryConfig
import CeFloPS.simulation.settings as settings
from .compartment_model import CompartmentModel
from .unit_conv import mm3_ml
#from .flowwalk import generate_arrays, generate_gradient_array TODO fix no cuda no error!
import CeFloPS.simulation.common.vessel_functions as vf
import pickle
import random
from .compartment_plotter import CompartmentPlotter
from CeFloPS.simulation.common.vessel2 import TissueLink
from CeFloPS.simulation.common.functions import calculate_distance_euclidian, normalize
import matplotlib.pyplot as plt
import numpy as np
import os
import CeFloPS.visualization as vis


# tissue_roi.py
from multiprocessing import get_context


class Tissue_roi:
    @classmethod
    def set_disk_load_mode(cls, mode):
        SharedGeometryConfig.set_mode(mode)
        print(f"Set disk load mode to: {mode}")





    def __getstate__(self):
        state = self.__dict__.copy()
        if SharedGeometryConfig.disk_load_mode:
            # Disk save: preserve raw data
            state["_geometry"] = self._geometry
            state["_vectorfield"] = self._vectorfield if hasattr(self,"_vectorfield") else None
            state["geometry"] = None
            state["vectorfield"] = None
        else:
            # Process mode: keep SharedGeometry, remove raw data
            state["_geometry"] = None
            state["_vectorfield"] = None
            if self.geometry is None:
                raise ValueError(f"[PROCESS] {self.name}: Geometry is None during pickling!")
        return state

    def __setstate__(self, state):

        self.__dict__.update(state)
        # Only recreate geometry if in disk mode
        if SharedGeometryConfig.disk_load_mode:
            print(f"[DISK] Recreating geometry for {self.name}")
            #assert state["_geometry"] !=None, "Diskmode should contain geometries pre computed"

            if state.get("_geometry") is not None:
                self.geometry = SharedGeometry(state["_geometry"])
                print("Points",len(self.geometry.get_points()))
                self._geometry= None # DO not store a second time (only one sharedmemory),state["_geometry"]
            else:
                self.geometry=""
                print("NO GEOMETRY LOADED AS NO _geometry available!")
            if state.get("_vectorfield") is not None:
                self.vectorfield = SharedGeometry(state["_vectorfield"])
                self._vectorfield = None # state["_vectorfield"]
            if False and self.geometry.get_points() is None:#TODO
                raise ValueError(f"[DISK] {self.name}: Failed to load geometry!")
        else:
            if "_shared_geometry" not in state:
                self._shared_geometry = state["geometry"]
                #assert state["geometry"]!=None


    def __init__(
        self,
        name,
        path,
        roi_type,
        compartment_model_ks,
        blood_roi,
        mesharray,
        meshvolume,
        meshcentroid,
        k_name="NAN",
        roi_name="NAN",
        store_loc=False,
    ):
        self._shared_geometry = None
        self.k_name = k_name
        self.roi_name = roi_name
        # self.concentration_share_current = self.get_concentration_share()
        self.cells_in_blood = 0
        self.veins = None  # set before sim
        self.name = name.split(".stl")[0]
        self.concentration_current = 0
        self.share_current = 0
        self.share_target = 0
        self.target_fraction = 1
        self.type = roi_type
        self.volume = 0
        self.volume_ml = 0
        self.blood_roi = blood_roi
        self.veins = []  # (entrypoint vein, vein, closest meshindex, meshpoint)
        self.cell_status = None
        self.compartment_model_ks = compartment_model_ks
        self.commpartment_model = None
        self.compartment_plotter = []
        self.outlets = []  # (vein_vessel,vein_index, roiarrayindex)
        self.inlets = []  # (artery_vessel,artery_index, roiarrayindex)

        # load geometry or create teststump
        if path != "test":
            self.center = meshcentroid
            self.volume = meshvolume
            # sort mesharray entries
            mesharray = sorted(mesharray, key=lambda x: tuple(x))
            """ vf.set_vessels_for_roi_single(self, self.blood_roi)#set vessels that lead to roi
            found_veinindices=
            found_art_indices
            mesharray, roi_array = flowwalk.generate_arrays(roi_points_sorted,settings.ROI_VOXEL_PITCH, found_veinindices, found_art_indices) """
            self.geometry = SharedGeometry(mesharray)
            if store_loc:
                self._geometry = mesharray
            # vf.set_vessels_for_roi_single(self, self.blood_roi)#set vessels that lead to roi
            if roi_type == 2:
                # self.vectorfield = SharedGeometry(roi_array)
                modelcreator = CompartmentModel()
                self.compartment_model = modelcreator.create(
                    2,
                    compartment_model_ks,
                    self,
                    self.blood_roi,
                )
                self.k1 = compartment_model_ks[0]

                # self.blood_roi.register_connected_roi_comp(self)
            elif roi_type == 3:
                # self.vectorfield = SharedGeometry(roi_array)

                modelcreator = CompartmentModel()
                self.compartment_model = modelcreator.create(
                    2,
                    compartment_model_ks,
                    self,
                    self.blood_roi,
                )
                self.k1 = compartment_model_ks[0]

            else:
                assert False
                #self.geometry = None
                self.volume = 800 * 1000  # 800 ml
                self.k1 = random.random()

        self.concentration_share_current = 0
        self.concentration_share_target = 0
        self.volume_ml = mm3_ml(self.volume)

    def add_compartment_plotter(self):
        self.compartment_plotter.append(CompartmentPlotter(self))
    @property
    def geometry(self):
        return self._shared_geometry
    @geometry.setter
    def geometry(self, value):
            if value is None:
                print(f"geometry set to None in process {os.getpid()}:")
                import traceback
                traceback.print_stack()
                assert False  # Halt
            self._shared_geometry = value
    def reload(self, blood_roi):
        # after loading with trimesh set all vessels to their correct pointers
        self.blood_roi = blood_roi
        vessels=blood_roi.geometry
        assert type(blood_roi) is not list
        for i in range(len(self.outlets)):
            vess, x, y = self.outlets[i]
            matching_vess = [v for v in vessels if v.id == vess.id][0]
            self.outlets[i] = matching_vess, x, y
        for i in range(len(self.inlets)):
            vess, x, y = self.inlets[i]
            matching_vess = [v for v in vessels if v.id == vess.id][0]
            self.inlets[i] = matching_vess, x, y
        self.recreate_compartments()
        # update links to self
        for vessel in vessels:
            for i in range(len(vessel.links_to_vois)):
                l = vessel.links_to_vois[i]
                av, ai, vv, vi = (
                    l.target_tissue,
                    l.target_index,
                    l.source_vessel,
                    l.source_index,
                )
                print(av, ai, vv, vi)
                if av.name == self.name:
                    vessel.links_to_vois[i] = TissueLink(vv, vi, self, ai)
        #self.geometry = SharedGeometry(self._geometry)
        #self.vectorfield = SharedGeometry(self._vectorfield)

    def create_flow_vectorfield(self, save=False, store_loc=False, safe_variant=False):
        print("VARIANT SAFE", safe_variant)
        if hasattr(self, "vectorfield"):
            if self.vectorfield != None and safe_variant == False:
                return
        self.vein_indices = [
            voi_index for vein_vessel, vein_index, voi_index in self.outlets
        ]
        self.art_indices = [
            voi_index for artery_vessel, artery_index, voi_index in self.inlets
        ]
        # min 1 of each:
        assert len(self.art_indices) > 0
        assert len(self.vein_indices) > 0
        # all in array:
        assert all(
            [i < len(self.geometry.get_points()) and i >= 0 for i in self.vein_indices]
        )
        assert all(
            [i < len(self.geometry.get_points()) and i >= 0 for i in self.art_indices]
        )
        if not safe_variant:
            vectorfield = generate_arrays(
                self.geometry.get_points(),
                settings.ROI_VOXEL_PITCH,
                vein_indices=self.vein_indices,
                art_indices=self.art_indices,
            )[1]
            # get fitting value and test if pathes are correctly contructed

        else:  # take safe way
            vectorfield = generate_gradient_array(
                self.geometry.get_points(),
                settings.ROI_VOXEL_PITCH,
                vein_indices=self.vein_indices,
                art_indices=self.art_indices,
            )[1]
        # print(vectorfield)
        self.vectorfield = SharedGeometry(vectorfield)
        if store_loc:
            self._vectorfield = vectorfield
        # print(self.vectorfield.get_points()[0])
        # assert False
        if save:
            try:
                with open(
                    settings.cached_flowfield_dir + "/" + self.cache_name, "wb"
                ) as handle:
                    pickle.dump(vectorfield, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(e)

    def recreate_compartments(self):
        modelcreator = CompartmentModel()
        self.compartment_model = modelcreator.create(
            2,
            self.compartment_model_ks,
            self,
            self.blood_roi,
        )
        self.k1 = self.compartment_model_ks[0]

    def set_target_fraction(self, fraction):
        self.target_fraction = fraction

    def get_volume(self):
        return self.volume_ml
 

    def update_concentration(self):
        # print(self.name, "concentration and cells", self.compartment_model.get_full_cell_count() / self.volume, self.compartment_model.get_full_cell_count())
        self.concentration_current = (
            self.compartment_model.get_full_cell_count()
        ) / self.volume_ml
        # self.cells_in_blood
        # TODO change to incluse cells in cappilars?
        # self.concentration_target = self.simulation.share[self.name] * CeFloPS.simulation.cell_count / self.volume

    def update_cell_share(self, simulation):
        assert False
        self.share_current = (
            self.compartment_model.get_full_cell_count()
            + self.cells_in_blood / CeFloPS.simulation.cell_count
        )
        # self.share_target = self.simulation.share[self.name]

    def get_cells(self):
        return self.compartment_model.get_full_cell_count()  # + self.cells_in_blood

    def update_concentration_share(self, simulation):
        # all cells for mapping / mapping volume
 

        index = [
            index
            for index, name in enumerate(simulation.sim_concentration.keys())
            if name == self.name
        ][0]
        self.concentration_share_current = f.normalize(
            [value for value in CeFloPS.simulation.sim_concentration.values()]
        )[index]
        # print(
        #    "cocnentration share cuzrrent", self.name, self.concentration_share_current
        # )

        self.concentration_share_target = (
            CeFloPS.simulation.concentration[
                [key for key, v in CeFloPS.simulation.mapping.items() if self in v][0]
            ]
            * self.target_fraction  # only needed for total cells not for concentration
        ) 

    def get_concentration_share(self, simulation):
        self.update_concentration_share(simulation)
        return self.concentration_share_current
 

    def get_concentration(self):
        self.update_concentration()
        return self.concentration_current

    def get_attraction(self, simulation):
        self.update_concentration_share(simulation)
        return self.concentration_share_target - self.get_concentration_share(
            simulation
        )
 

    def get_cell_compartment_entry(self, C_a):
        res = self.k1 * self.volume_ml  # ml per min
        res = res / 60  # ml per s
        res = (
            C_a * res  # C a is already the cell! No need to multiply
        )  # Beq per ml * ml per s = bq*ml/s*ml = 1/s // s = beq per s => cells per s
        return res  # cells per s or beq per second depending on C_a input unit

    def get_cell_compartment_entry_chance(self):
        return self.k1


# fit the capillary speedup!
import yaml


def flow_walk_step(roi, pos_index, speed=1, already_visited=set(), iteration=0):
    roi_array = roi.vectorfield.get_points()
    roi_points_sorted = roi.geometry.get_points()
    selection = [
        int(i)
        for k, i in enumerate(roi_array[pos_index][0])  # nbs
        if roi_array[pos_index][1][k] > 0
        and i != -1
        and roi_array[pos_index][1][k] not in already_visited
    ]  # possible indexes from current index
    # print("AROUND",roi_array[pos_index][0])
    chances = [
        roi_array[pos_index][1][k]
        for k, i in enumerate(roi_array[pos_index][0])
        if roi_array[pos_index][1][k] > 0 and i != -1
    ]
    if len(selection) == 0 or iteration == -1:

        return None

    chances = normalize(chances)
    new_index = random.choices(selection, weights=chances, k=1)[0]
    time_taken = (
        calculate_distance_euclidian(
            roi_points_sorted[pos_index], roi_points_sorted[new_index]
        )
        / speed
    )
    new_point = roi_points_sorted[new_index]
    return new_index, new_point, time_taken


def generate_times_per_inlet(amount_of_paths, roi, scale_value=100000000, speed=1):
    starts = [v[2] for v in roi.inlets]
    qs = [
        int(v[0].volumes[-1].get_symval(v[0].volumes[-1].Q_1)) * scale_value
        for v in roi.inlets
    ]

    # Ensure each q is at least 1
    qs = [max(1, q) for q in qs]

    if amount_of_paths == "q":
        total = sum(qs)
        if total > 10000:
            reduce_by = total - 10000
            # Sort indices by descending q to prioritize reducing larger quantities first
            sorted_indices = sorted(range(len(qs)), key=lambda i: -qs[i])
            for i in sorted_indices:
                if reduce_by <= 0:
                    break
                current_q = qs[i]
                max_reduction = current_q - 1  # Can't reduce below 1
                if max_reduction >= reduce_by:
                    qs[i] -= reduce_by
                    reduce_by = 0
                else:
                    qs[i] -= max_reduction
                    reduce_by -= max_reduction
        pathamounts = qs
    else:
        num_inlets = len(starts)
        total_requested = amount_of_paths * num_inlets
        if total_requested > 10000:
            base = 10000 // num_inlets
            remainder = 10000 % num_inlets
            pathamounts = [base + 1] * remainder + [base] * (num_inlets - remainder)
        else:
            pathamounts = [amount_of_paths] * num_inlets

    paths = [[], []]
    print("Total paths:", sum(pathamounts))

    for i, cindex in enumerate(starts):
        pathamount = pathamounts[i]
        for number in range(pathamount):
            print(number, "/", pathamount)
            current_index = cindex
            path = [roi.geometry.get_points()[current_index]]
            times = [0]
            iteration = 0
            visited_set = set()
            while True:
                x = flow_walk_step(
                    roi,
                    current_index,
                    speed,
                    iteration=iteration,
                    already_visited=visited_set,
                )
                iteration += 1
                if x is None or iteration == 50000:
                    break
                current_index, next_point, time = x
                times.append(time)
                path.append(next_point)
            paths[0].append(path)
            paths[1].append(times)
    return paths


def fit_capillary_speed(
    connected_vois, plot=False, visual=False, save=True, store_loc=False
):
    unsafe = False
    # put name entry in yaml if not already there
    yaml_values_path = settings.parent_dir + "/voi_speed_fitting_values.yaml"
    yaml_coeff_path = settings.parent_dir + "/coefficients_vois.yaml"
    yaml_keys_path = settings.parent_dir + "/voi_speed_fitting_keys.yaml"

    def get_matching_entry(valfilekeydict, valfilevaluedict, voiname, plot=False):
        selkey = None
        for key, stringkey in valfilekeydict.items():
            if stringkey in voiname:
                selkey = key
        if selkey == None:
            selkey = "all other tissues"
            print("no key for speed found for ", voiname)
            # return 4#use 4 by default
        return valfilevaluedict[selkey], selkey

    for voi in connected_vois:
        scale_value = 1000
        while any(
            [
                v[0].volumes[-1].get_symval(v[0].volumes[-1].Q_1) * scale_value < 1
                for v in voi.inlets
            ]
        ):
            scale_value *= 2
            print(scale_value)
        paths, times = generate_times_per_inlet("q", voi, scale_value, speed=1)
        unsafes = 0
        for p in paths:
            # print(p[-1])
            if not tuple(p[-1]) in [
                tuple(voi.geometry.get_points()[m]) for m in voi.vein_indices
            ]:
                print("Path didnt end at veinpoint")
                unsafes += 1
        if unsafes > len(paths) / 2:  # more thatn 50% are very long, probably stuck
            unsafe = True
        if unsafe:
            # regenerate with alternative method
            voi.create_flow_vectorfield(
                save=save, store_loc=store_loc, safe_variant=True
            )
            paths, times = generate_times_per_inlet("q", voi, scale_value, speed=1)
            for p in paths:
                if not tuple(p[-1]) in [
                    tuple(voi.geometry.get_points()[m]) for m in voi.vein_indices
                ]:
                    assert False, "Safe method was unsafe?"
        # print("PATHS",paths)
        print([a[2] for a in voi.outlets])
        print(
            [([voi._geometry[inlet[2]]], [200, 0, 0, 200]) for inlet in voi.inlets],
            "------------",
            [([voi._geometry[outlet[2]]], [0, 0, 200, 200]) for outlet in voi.outlets],
        )
        # print([([voi._geometry[inlet[3]]],[200,0,0,200]) for inlet in voi.inlets]+[([voi._geometry[outlet[3]]],[0,0,200,200]) for outlet in voi.outlets])
        if visual:
            vis.show(
                [([voi._geometry[inlet[2]]], [200, 0, 0, 200]) for inlet in voi.inlets]
                + [
                    ([voi._geometry[outlet[2]]], [0, 20, 200, 250])
                    for outlet in voi.outlets
                ]
                + [(path, [200, 100, 100, 200]) for path in paths]
                + [(list(voi._geometry), [10, 10, 10, 1])]
            )
        # now take those paths and calculate mean, shift until desired and save the corresponding speedup
        pathlengths = sorted([int(sum(path)) for path in times])

        # Count the occurrences of each x value
        with open(yaml_values_path, "r") as f:
            want_values = yaml.safe_load(f) or {}
        with open(yaml_keys_path, "r") as f:
            want_keys = yaml.safe_load(f) or {}

        wanted_avg, save_key = get_matching_entry(want_keys, want_values, voi.name)
        print(voi.name, wanted_avg, np.mean(pathlengths))
        fit_coef = np.mean(pathlengths) / wanted_avg  # antiproportional
        if os.path.exists(yaml_coeff_path):
            with open(yaml_coeff_path, "r") as f:
                coeff_data = yaml.safe_load(f) or {}
        else:
            coeff_data = {}
        # add the value to the yaml yaml_coeff_path file if it wasnt in before
        current_value = coeff_data.get(save_key)
        voi.speed = current_value
        if current_value is None:  # or not np.isclose(current_value, fit_coef):
            coeff_data[save_key] = float(fit_coef)
            voi.speed = float(fit_coef)
            with open(yaml_coeff_path, "w") as f:
                yaml.dump(coeff_data, f, default_flow_style=False)

        print(voi.name, voi.speed)
        if plot:
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.hist(
                pathlengths,
                bins=range(min(pathlengths), max(pathlengths) + 2),
                edgecolor="black",
                align="left",
            )
            plt.xlabel("Pathtime (s)")
            plt.ylabel("Frequency")
            plt.title("Histogram of occuring pathtimes")

            plt.tight_layout()
            plt.show()
