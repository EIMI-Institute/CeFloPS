import os
import pickle
import time
import sys
print(sys.path)
import numpy as np
#sys.modules['numpy._core'] = np.core
from multiprocessing import (
    SimpleQueue,
    Value,
    Process,
    cpu_count,
    current_process,
    Barrier,
    Event,
)
import random
import argparse
from numba import jit, config
from scipy.interpolate import interp1d
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
np.random.seed(0)
random.seed(0)
 
#load simulation from vectorized sim instead
from CeFloPS.simulation.vectorized_cells_simulation import Simulation
from CeFloPS.simulation.common.distributor import Distributor
import CeFloPS.simulation.settings as settings
from CeFloPS.simulation.common.vol_sampler_triple import volume_sampler_triple
from CeFloPS.simulation.common.sampler import Sampler
from CeFloPS.simulation.simsetup import *
import CeFloPS.simulation.sim_configs.transition_config_tlinks as transition_config_tlinks
import CeFloPS.simulation.sim_configs.transition_config_tlinksFW_STEADY_STATE as transition_config_tlinksFW_STEADY_STATE
from CeFloPS.simulation.common.compartment_plotter import CompartmentPlotter
from CeFloPS.simulation.common.functions import *
import sys
import CeFloPS.simulation.sim_configs.transition_config_vectorized_steady as steady_vectorized

from multiprocessing import Process, SimpleQueue
from functools import wraps
from multiprocessing import get_context, Event, Value, SimpleQueue
from CeFloPS.simulation.common.shared_geometry import SharedGeometryConfig

from CeFloPS.logger_config import setup_logger
import logging

from CeFloPS.simulation.sim_output_generation import create_GATE_output, update_countlog_using_loclist, create_simlog_from_loclist,save_roi_data,save_roi_data,save_data,settings_store_sim_info,process_simulation_results

logger = setup_logger(__name__, level=logging.WARNING)

sys.setrecursionlimit(100000)
state_machine = transition_config_tlinksFW_STEADY_STATE.DefStateMachine()
if False:# and not settings.USE_FLOW_WALK:
    state_machine = transition_config_tlinks.DefStateMachine()
def speed_dummy(x,y):
    return x

state_machine =steady_vectorized.DefStateMachineVec()
path = r"./resources/concentrations/"
parser = argparse.ArgumentParser(description="Script to run a simulation")
parser.add_argument(
    "sim_name",
    type=str,
    help="string to append to the simulation folders name",
)

parser.add_argument(
    "--test", type=str, help="Override settings with testsimulation data."
)
args = parser.parse_args()
if args.sim_name != None:
    settings.tags = args.sim_name
is_test_run = False
if args.test == "True" or False:  # changed to be always true for the demo
    print("Simulation in testmode!")
    is_test_run = True
    settings.PATH_TO_VESSELS = r"./testvessels_directions_volumes.pickle"
    settings.NEGATIVE_SPACES = []
    settings.PATH_TO_STLS = r"./resources"
    settings.SUBTRACT_VESSELMESHES = False
    settings.ARMINJECTION = False
np.seterr("raise")


class simple_TRoi:
    def __init__(self,name,trimesh_repr,voxel_size=1.5):
        self.name=name
        self.trepr=trimesh_repr
        vox = trimesh_repr.voxelized(pitch=voxel_size)
        # Points for all occupied voxels
        self.geometry = vox.points
        # Volume of the mesh
        self.volume = trimesh_repr.volume
        self.roi_name=name

        self.inlets=[]
        self.outlets=[]
    def inflow(self):
        q=0
        for inlet in self.inlets:
            print("in:",inlet[0].associated_vesselname)
            q+=inlet[0].q
        return q
    def outflow(self):
        q=0
        for inlet in self.outlets:
            print("out:",inlet[0].associated_vesselname)
            q+=inlet[0].q
        return q

def log(iterated, interval, concentration, counts=None):
    if "time" in simlog:
        simlog["time"].append(iterated * interval)
    else:
        simlog["time"] = [iterated * interval]
    for key, value in concentration.items():
        if key in simlog:
            simlog[key].append(value)
        else:
            simlog[key] = [value]
    if counts is not None:
        if "time" in countlog:
            countlog["time"].append(iterated * interval)
        else:
            countlog["time"] = [iterated * interval]
        for key, value in counts.items():
            if key in countlog:
                countlog[key].append(value)
            else:
                countlog[key] = [value]
    # print("concentration:",concentration)
def worker_initializer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Set shared memory mode first
        SharedGeometryConfig.set_mode(False)

        # Extract communication queue from args (adjust index based on your actual argument order)
        comm_queue = args[7]

        # Validate ROIs before proceeding
        final_rois = args[0]
        for roi in final_rois[:-1]:
            logger.print("Check")
            try:
                points = roi.geometry.get_points()
                if len(points) == 0:
                    comm_queue.put(('ERROR', f"Empty geometry in ROI {roi.name}"))
                    return  # Exit early on failure
            except Exception as e:
                comm_queue.put(('ERROR', f"ROI validation failed for {roi.name}: {str(e)}"))
                return


        # Only proceed to actual simulation if validation passed
        return func(*args, **kwargs)
    return wrapper
directory = (
        settings.results_dir
        + "/simulation_result_"
        + settings.tags
        + "_"
        + str(int(time.time()))
    )

@worker_initializer
def simulate(
    final_rois,
    cell_count,
    roi_mapping,
    interval,
    concentrations,
    stop_cond,
    signal_int,
    queue,
    event,
    barrier,
    roi_mapping_str,
    logcount,
    seed_num
):
    print("started process")
    # TODO give other startconcentrations for shared sim
    # set vessel roi relations

    print("Starting sim", os.getpid())
    #assert False,roi_mapping_str

    simulation = Simulation(
        cell_count,
        Distributor(
            final_rois,
            [0 if "blood" not in roi.name else cell_count for roi in final_rois],
            cell_count,
        ),
        rois=final_rois,
        roi_regionsmapping=roi_mapping,
        interval=interval,
        concentrations=concentrations,
        state_machine=state_machine,
        stop_condition=stop_cond,
        roi_mapping_str=roi_mapping_str,
        loggingcount=logcount,
        store_dir=directory,
        seed=seed_num
    )

    print(
        f"Setup simulation for {cell_count} cells and {settings.TIME_LIMIT_S} seconds, stop condition is {settings.stop_condition}"
    )
    print("ROIs:")
    for roi in final_rois:
        print(roi.name)
    # print(f"Initial distribution is {initial_distribution.type}")
    return simulation.run_parallel(signal_int, queue, event, barrier)


def divide_amount_evenly(amount, size):
    base_value = amount // size
    remainder = amount % size
    result = [base_value] * size
    for i in range(remainder):
        result[i] += 1
    return result


def terminate_all(processes):
    for _process in processes:
        if _process.is_alive():
            _process.terminate()
            _process.join()


def create_roi_mappings(final_rois, region_patterns, blood_roi):
    """Create ROI mappings based on preconfigured region patterns and naming conventions."""
    roi_mapping = {}
    roi_mapping_str = {}

    # Process all defined regions from the pattern configuration
    for region_data in region_patterns.values():
        original_name = region_data["original"]
        patterns = (
            [
                p.lower()
                for p in region_data["patterns"]
                if p not in ["----", "---bone.---"]
            ]
            if region_data["patterns"] != ["----"]
            else []
        )

        # Handle special cases with multiple patterns
        if original_name.lower() == "muscle":
            patterns.extend(["musc", "muscle"])

        matched_rois = []
        for roi in final_rois:
            roi_lower = roi.name.lower()
            # Match against all valid patterns for this region
            if any(p in roi_lower for p in patterns):
                matched_rois.append(roi)

        roi_mapping[original_name] = matched_rois
        roi_mapping_str[original_name] = [roi.name for roi in matched_rois]

    # Add blood entry and handle composite regions
    roi_mapping["blood"] = [blood_roi]
    roi_mapping_str["blood"] = [blood_roi.name]

    # Handle composite groupings
    composite_definitions = {
        "adrenal gland": ["adrenal"],
        "gastrointestinal tract": ["intest", "throat", "stomach"],
        "kidneys": ["kidney"],
        "pancreas": ["pancreas"],
    }

    for comp_name, keywords in composite_definitions.items():
        comp_rois = [
            roi
            for roi in final_rois
            if any(kw in roi.name.lower() for kw in keywords)
            and roi not in {r for rs in roi_mapping.values() for r in rs}
        ]
        roi_mapping[comp_name] = comp_rois
        roi_mapping_str[comp_name] = [roi.name for roi in comp_rois]

    return roi_mapping, roi_mapping_str



# Top-level wrapper for worker initialization

def create_process(*args, **kwargs):
    """Process factory using properly decorated functions"""
    ctx = get_context("spawn")
    return ctx.Process(*args, **kwargs)
# Decorate simulate at module level


from sympy.core.numbers import Float
from copyreg import pickle


# Add before process creation
def reduce_float(obj):
    return (float, (float(obj),))

import re

import re

def normalize_name(name):
    import re
    return re.sub(r'[\W_]+', '', str(name).lower())


def roi_delivers_map(final_rois, region_remmapping):
    """
    Maps regions (keys) to ROIs whose .name/.k_name/.roi_name contain
    any value (from the dict's values) as a substring.
    Throws ValueError if ANY ROI is not mapped to at least one region.
    Returns: (roi_mapping_str, roi_mapping)
    """
    roi_mapping_str = {region: [] for region in region_remmapping}
    roi_mapping = {region: [] for region in region_remmapping}
    roi_to_regions = {id(r): set() for r in final_rois}
    already_mapped = set()
    # For each region, look for ROI where any value is a substring
    for region, patterns in region_remmapping.items():
        if not isinstance(patterns, (list, tuple)):
            patterns = [patterns]
        patterns = [normalize_name(p) for p in patterns if isinstance(p, str) and p != "----" and p.strip() != ""]
        for r in final_rois:
            if id(r) in already_mapped:
                continue
            sfields = [
                normalize_name(getattr(r, 'name', "")),
                normalize_name(getattr(r, 'roi_name', "")),
                normalize_name(getattr(r, 'k_name', ""))
            ]
            for p in patterns:
                if p and any(p in field for field in sfields):
                    roi_mapping[region].append(r)
                    roi_mapping_str[region].append(r.name)
                    roi_to_regions[id(r)].add(region)
                    already_mapped.add(id(r))
                    break

    # Check for unmapped ROIs
    unmapped = [r for r in final_rois if not roi_to_regions[id(r)]]
    if unmapped:
        print(region_remmapping,sfields)
        raise ValueError(
            f"[roi_delivers_map] The following ROI(s) could not be mapped to any region: {[r.name for r in unmapped]}\n"
            f"Example: {vars(unmapped[0])}"
        )

    return roi_mapping_str, roi_mapping
pickle(Float, reduce_float)

import multiprocessing as mp
import yaml
import os

def set_capillary_speed(connected_vois):
    """
    For each VOI in connected_vois, sets .speed according to precomputed coefficients in YAML.

    Assumes:
      - settings.parent_dir is set
      - 'settings.parent_dir/coefficients_vois.yaml' contains {region_name: speed}
      - 'settings.parent_dir/voi_speed_fitting_keys.yaml' contains {region_name: substring_to_match}
      - If no match is found, skips or sets None for speed

    Returns:
      A dict mapping voi.name -> assigned speed, for debug/testing.
    """
    yaml_coeff_path = settings.parent_dir + "/coefficients_vois.yaml"
    yaml_keys_path = settings.parent_dir + "/voi_speed_fitting_keys.yaml"

    # Load the keys/strings YAML for matching
    if os.path.exists(yaml_keys_path):
        with open(yaml_keys_path, "r") as f:
            want_keys = yaml.safe_load(f) or {}
    else:
        want_keys = {}

    # Load the coefficients
    if os.path.exists(yaml_coeff_path):
        with open(yaml_coeff_path, "r") as f:
            coeff_data = yaml.safe_load(f) or {}
    else:
        coeff_data = {}

    def get_matching_key(valfilekeydict, voiname):
        """Finds the save_key for a VOI name based on substring matching from key dict."""
        for key, stringkey in valfilekeydict.items():
            if stringkey in voiname:
                return key
        # Default/fallback key
        return "all other tissues"

    voi_speed_assignment = {}
    for voi in connected_vois:
        save_key = get_matching_key(want_keys, voi.name)
        speed_val = coeff_data.get(save_key)
        voi.speed = speed_val
        voi_speed_assignment[voi.name] = speed_val

    return voi_speed_assignment
if current_process().name == "MainProcess":
    from CeFloPS.simulation.common.shared_geometry import SharedGeometryConfig

    # Enable disk mode for main process loading
    cell_count = settings.CELL_COUNT

    mp.freeze_support()
    mp.set_start_method("spawn", force=True)

    concentrations = dict()
    print("Testrun: ", is_test_run)

    import pickle
    Tissue_roi.set_disk_load_mode(True)
    SharedGeometryConfig.set_mode(True)
    with open(
        settings.cache_dir +"/final_rois_lungbalance_2.pickle",
        "rb",
    ) as input_file:
        final_rois, roi_mapping = pickle.load(input_file)
    #TODO region_remapping = settings.region_remapping
    region_remapping=roi_mapping
    set_capillary_speed(final_rois[:-1])
    #assert False, final_rois[0].__dict__
    from CeFloPS.simulation.common.vessel2 import TissueLink

    blood_roi=final_rois[-1]
    assert not isinstance(blood_roi,Tissue_roi)
    vessels=blood_roi.geometry
    def load_vesseltags(vessels,final_rois):
        try:
            with open(settings.vessel_dir+'/vessel_nodetag_data_no_musc.pkl', 'rb') as file:
                loaded_data = pickle.load(file)
            #apply to vessels
            loaded_roinames=[roi.name for roi in final_rois]
            for vessel in vessels:
                # Ensure vessel has 'node_tags' initialized
                if not hasattr(vessel, "node_tags")or True:
                    vessel.node_tags = ["B" for _ in range(len(vessel.path))]

                # Check if the vessel ID is in `loaded_data`
                if vessel.id in loaded_data:
                    # Access the indices associated with each mesh for this vessel
                    for mesh_name, indices in loaded_data[vessel.id].items():  
                        if mesh_name in loaded_roinames or True:
                            for index in indices: 
                                print(f"Processing Vessel ID: {vessel.id}, Mesh: {mesh_name}, Index: {index}")

                                # Only try to replace if 'index' is within bounds
                                if index < len(vessel.node_tags) and vessel.node_tags[index] == "B":
                                    vessel.node_tags[index] = mesh_name
        except:
            #Fallback to mark everything as blood
            for vessel in vessels:
                for i in range(len(vessel.path)):
                    vessel.node_tags[i]="B"
        for vessel in vessels:
                for i in range(len(vessel.path)):
                    if vessel.node_tags[i]=="B" and vessel.type=="vein":
                        vessel.node_tags[i]="VB"

    load_vesseltags(vessels,[])

    def update_objects_with_max_iteration(objects):
        """
        Updates objects with max_iteration_number based on orig.

        Args:
            objects: A list of objects, each with 'orig' and 'iteration_number' attributes.
        """
        max_iterations = {}  # Dictionary to store max iteration per orig

        # First pass: Find the maximum iteration number for each orig
        for obj in objects:
            orig = obj.orig
            iteration = obj.iteration_number
            if orig not in max_iterations or iteration > max_iterations[orig]:
                max_iterations[orig] = iteration

        # Second pass: Update the objects with the max iteration number
        for obj in objects:
            obj.max_iteration_number = max_iterations[obj.orig]


    if not os.path.exists(directory):
        os.makedirs(directory)
    update_objects_with_max_iteration(blood_roi.geometry)
    for roi in final_rois[:-1]:assert not any([type(x)==list for x in roi.compartment_model.C_outs])
    print(final_rois)
    for roi in final_rois[:-1]:
        blood_roi.register_connected_roi_comp(roi)
        roi.reload(blood_roi)

    #Create a new roi mapping and roi mapping string from final_rois:

    """ roi_mapping, roi_mapping_str = create_roi_mappings(
        final_rois, roi_mapping, blood_roi
    ) """
    roi_mapping_str, roi_mapping= roi_delivers_map(final_rois[:-1], region_remapping)
    for roi in final_rois[:-1]:assert not any([type(x)==list for x in roi.compartment_model.C_outs])

    for roi in final_rois[:-1]:
        assert roi._vectorfield==None
        assert roi._geometry==None
        assert roi.geometry!=None




    shared_geometries = []
    for roi in final_rois[:-1]:
        print(len(roi.geometry.get_points()))
        if len(roi.geometry.get_points())==0:assert False
    """ final_rois, blood_roi, roi_mapping, roi_mapping_str = load_vois(
        vessel_not_allowed_in_path=is_test_run
    )# connects vessels to vois. registers connections, register volumes, tries traversion """
    print("precall", len(blood_roi.geometry))

    for vessel in blood_roi.geometry:
        vessel.unregister_functions()
        for profilekey, value in vessel.profiles.items():
            if profilekey[1] == "v_r":
                assert value == None
    # vf.set_rois_for_vessel (final_rois) done in single ones
    for vessel in blood_roi.geometry:
        for profilekey, value in vessel.profiles.items():
            if profilekey[1] == "v_r":
                assert value == None
    #assert blood_roi in final_rois
    blood_roi.additional_vois = None  # reset additional vois to not copy them
    #blood_roi2.additional_vois = None
    # Disable disk mode for multiprocessing
    Tissue_roi.set_disk_load_mode(False) 

    simlog = dict()
    countlog = dict()
    merged_count = dict()
    interval = settings.INTERVAL
    iterated = 0
    if settings.PROCESS_COUNT > 0:
        process_count = min(cpu_count(), settings.PROCESS_COUNT)
    else:
        if settings.PROCESS_COUNT < cpu_count():
            process_count = cpu_count() - settings.PROCESS_COUNT
        else:
            assert False, "Process count should be in [PCOUNTMAX-1,...,-1,1,...N]"

    start_time = time.time()
    processes = []
    process_comms = dict()
    ready_event = Event()
    barrier = Barrier(process_count + 1)
    cell_counts = divide_amount_evenly(amount=settings.CELL_COUNT, size=process_count)
    tracked_cell_counts = divide_amount_evenly(
        amount=settings.CELLS_WITH_PATH_LOGGING, size=process_count
    )
    for i in range(process_count):
        process_comms[i] = [SimpleQueue(), Value("i", -3), ready_event, barrier]
        _process = create_process(  # Process(
            target=simulate,
            args=(
                final_rois,
                cell_counts[i],
                roi_mapping,
                interval,
                concentrations,
                settings.stop_condition,
                process_comms[i][1],
                process_comms[i][0],
                process_comms[i][2],
                process_comms[i][3],
                roi_mapping_str,
                tracked_cell_counts[i],
                i
            ),
        )
        processes.append(_process)
    for _process in processes:
        _process.start()
        logger.print("Started Process:",_process)
    time.sleep(1)
    result = []
    printcount = 0
    time_spent = []
    try:
        median_iterated = 0
        process_error = False
        while not all([process_comms[i][1].value == -5 for i in range(process_count)]):
            # check if ALL signalints are 2:

            if process_comms[0][1].value >= 0 and settings.SYNCHRONIZE:
                lead_time = process_comms[0][1].value
                if all(
                    [
                        process_comms[i][1].value == lead_time
                        for i in range(process_count)
                    ]
                ):
                    # if all are 3 calculate current sim and queue to every process
                    cellcounts = [
                        process_comms[i][0].get() for i in range(process_count)
                    ]

                    new_concentrations, global_count = calculate_sim_concentrations(
                        cellcounts
                    )
                    for i in range(process_count):
                        process_comms[i][0].put(new_concentrations)
                    for i in range(process_count):
                        process_comms[i][1].value = -4
                        # they may set themselves to 0

                    log(iterated, interval, new_concentrations, global_count)
                    time_spent.append(time.time())
                    iterated += 1
            if printcount > 66:
                # check for crashed processes

                for _process in processes:
                    if _process.exitcode is not None:  # Process has finished
                        if _process.exitcode != 0:
                            print(
                                f"Process {_process.pid} failed with exit code {_process.exitcode}. Terminating all."
                            )
                            terminate_all(processes)
                            process_error = True

                if process_error:
                    break
                # addition of avg_iterated to avoid a mean on an empty list
                prev_med_percent = median_percent if 'median_percent' in locals() else 0
                percent_list = [
                    process_comms[i][1].value
                    for i in range(process_count)
                    if process_comms[i][1].value > 0
                ]
                median_percent = np.median(percent_list + [prev_med_percent])

                if (median_percent - prev_med_percent > 0) and not settings.SYNCHRONIZE:
                    time_spent.append(time.time())

                # Estimate time remaining based on percent done:
                fraction_done = median_percent / 100.0
                if fraction_done > 0:
                    total_estimated_time = (time.time() - time_spent[0]) / fraction_done
                    time_left_sec = total_estimated_time * (1 - fraction_done)
                else:
                    time_left_sec = float('nan')

                if (
                    settings.parallel_sim_feedback == "PROC"
                    or settings.parallel_sim_feedback == "FULL"
                ):
                    print(
                        "PStat:",
                        [process_comms[i][1].value / 100 if process_comms[i][1].value != -5 else 100 for i in range(process_count)],
                        f"Median progress: {median_percent:.1f}%",
                        "Target 100%",
                    )

                # ETR print:
                if len(time_spent) > 1:
                    print("ETR:", round(time_left_sec / 60, 2), "minutes")
                else:
                    print("ETR: NAN (not enough progress yet)")

                if (
                    settings.parallel_sim_feedback == "PROC"
                    or settings.parallel_sim_feedback == "FULL"
                ):
                    print(
                        "PStat:",
                        [process_comms[i][1].value/100 for i in range(process_count)],
                        iterated * interval,
                        "Target 100",
                        #settings.TIME_LIMIT_S / settings.INTERVAL,
                    )
                printcount = 0
            printcount += 1
            time.sleep(0.015)
        # print(process_comms[0][0].empty())
        end_time = time.time() - start_time
        # cleanup
        result = []
        for i, _process in enumerate(processes):
            print(f"Waiting for result from process {i}...")
            res = process_comms[i][0].get()
            print(f"Received result from process {i}")
            result.append(res)
            _process.join()
            print("Process EXITCODE", _process.exitcode)
    except KeyboardInterrupt:
        print("Terminating all processes due to keyboard interrupt.")
        process_error = True
        terminate_all(processes)
    process_simulation_results(result, directory, final_rois, roi_mapping_str, interval)
    settings.store_sim_info(directory,end_time)


def close_rois(final_rois):
    """Closes ROIs."""
    for roi in final_rois[:-1]:
        if hasattr(roi, "geometry") and not roi.geometry==None:
            roi.geometry.close()
