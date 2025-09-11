# This file holds all setting constants for the program, please set the appropiate values for your system.

# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------REALTIVE PATHS ------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------


import os
import pickle
import glob
import yaml

import logging
from CeFloPS.logger_config import setup_logger

logger = setup_logger(__name__, level=logging.CRITICAL)

# Determine simulation path variables
settings_path = os.path.abspath(__file__)

# path to parent directory
parent_dir = os.path.dirname(settings_path)
sim_folderparts = parent_dir.replace("\\", "/").split("/")[:-1]
sim_folder = ""
for part in sim_folderparts:
    sim_folder += part + "/"
sim_folder = sim_folder[:-1]
logger.print("sim folderr", sim_folder, sim_folderparts, parent_dir)
# paths to result folders
results_dir = os.path.join(sim_folder, "sim_results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
vessel_dir = os.path.join(sim_folder, "vessel_output")
if not os.path.exists(vessel_dir):
    os.makedirs(vessel_dir)
cache_dir = os.path.join(sim_folder, "cache")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
cached_flowfield_dir = os.path.join(sim_folder, "cached_flowfields")
if not os.path.exists(cached_flowfield_dir):
    os.makedirs(cached_flowfield_dir)


# filename of the fitted ODE model    
filename = "pkode_fits_grid/pkode_injend0.16666666666666666_injw2.00_tailw10.00_tailfrom40.0.pt"
model_path = os.path.join(parent_dir, filename)
 


# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------SIMULATION ------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

PROCESS_COUNT = 6 #number of processes to spawn
CELLS_WITH_PATH_LOGGING = 1000 # number of cells to log their positional data
CELL_COUNT = 1000 # number of cells to simulatoi in total
USE_FLOW_WALK = True # load flowwalk information, this is overriden to true internally atm
SET_ELIM_ZERO=False # ignore the elimination compartment
rate_time_start=None  #can be set to a time value in seconds from which the model should start its ODE behaviour
TIME_LIMIT_S = 100# simulaiton time limit
CHANGE_LIMIT =0# can be set to ignore the ODE behaviour until time s. This does not initialize the probability vector at s, so at the point of allowance a lot of transitions will happen
RETURN_AFTER_C1_RELEASE = True # behaviour of a cell: should it return to bloodstream after exiting C1 or be able to reenter in the same "flush"
CAPILLARY_SPEED = 4  # mm/s default capillary speed, fitted values are used if available 
IGNORE_TLINKS = False #shpuld Tlinks be ignored (those that are not definitely connected but modeled by hand/heuristically, e.g. muscles)
CELL_INITAL_DISTRIBUTION = "random"  # overriden by arm or leg injection
SUBTRACT_VESSELMESHES = True #should vesselmeshes be subtracted from the VOIs available vector positions?
# constants for simulation
LEGINJECTION = True#False
ARMINJECTION = False#True  # OVERRIDES RANDOM, or leg!
ROI_VOXEL_PITCH = 1 # voxel pitch
TCHANCE = 0.1 # change for tlinks, as they can not be determined by blood flow
# should r have a positional influence on the cells in vessels:
SHOW_R_CHANGES = False # deprecated option to not reflect displacements inside of vessels
STEPSIZE = 0.2 # used in vesselcreation
MAXDELAY = 5 # deprecated option for injection timeframe
USE_VOL_SPEED = True #usage of interpolated speeds from literature values vs calculated ones assuming laminar flow



SYNCHRONIZE = False #synchronization between the processes, deprecated as they are no longer dependent
R_BIAS = 0.7 # deprecated method that used old displacement for new displacement, we keep the same radial displacement as flow is laminar on average
INTERVAL = 0.6 # simulation update interval (at least progress each cell INTERVAL time per iteration of the simulation loop)


# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------IDENTIFIER ------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------
VESSEL_VEINLIKE = "vl."  # vessel stl that has bloodflow towards the heart
VESSEL_ARTERYLIKE = "al."  # vessel stl that has bloodflow away from the heart
ROI_3D = "organs"  # Identifier with which 3d random walk should be used
ROI_2_5D = [
    "stomach",
    "intest",
    "throat",
    "trachea",
]  # rois in which only the outer layer of the mesh shall be traversed; usually tubes / not filled with tissue
NEGATIVE_SPACES = [
    # "heart.systole.lv4.stl",
    # "heart.systole.lam.stl",
    # "beating_heart.systole.sys_rv_0.stl",
    # "beating_heart.systole.sys_lv_4_0.stl",
    "bronchial_tree.trachea.stl",
    "bronchial_tree.bronchi.stl",
]  # heartchambers, bronchial tree. Vessels automatically get subtracted!


# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------VESSELCREATION ------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

# constants for mesh vessel conversion
GUIDE_POINT_DISTANCE = 4.5
DISTANCE_BETWEEN_PATHPOINTS = 1
MAX_PATHINDEX_DISTANCE_VOLUMES = 2

# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------PATHS ------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

# input paths
PATH_TO_STLS = "pathto/STL"  # path to stl files for vessels
PATH_TO_ROIS = "pathto/STL"  # path to stl files for rois: muscles, organs, etc.
CALIBRATION_INPUT_CONCENTRATIONS = "concentrations/" #deprecated
PATH_TO_VESSELS = vessel_dir + "/" + "1_1_synthsim.pickle"#path to vesselfiles to use in scripts
# output paths:
PATH_TO_VESSELOUTPUT = vessel_dir
tags = ""
descriptionpath = None

# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------REGIONMAPPING ------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------RATE CONSTANTS ------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------OUTPUT ------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------
COLLECT_KEYFRAMES = [6, 18, 30, 60, 300, 600]  # [i for i in range(45)]
MINIMAL_TIME_DIFFERENCE_OUTPUT = 1  # PPS (PointsPerSecond) 10 per second times 45 seconds times 1000000 = 450 000 000 Zeilen; 450*10⁶ / 1400 = 321428
PLOT_COMPARTMENTS_WITH_NAMEPART = ["lung"]
parallel_sim_feedback = (
    "FULL"  # PROC for processes , CELL for cellcount per region, FULL for both
)


# MISC
# define remappings to load under different rate constants
remappings = [("pancreas", "GUC_lesions"), ("Pancreas", "GUC_lesions")]
TRACER_HALFLIFE = (
    6600  # half-life of tracer in seconds, FDG is default with 110 minutes
)

# identifier to load stl files accordingly
MUSCLE_IDENTIFIER = "---"
LUNG_IDENTIFIER = "lung"
LIVER_IDENTIFIER = "liver"
GREY_MATTER_IDENTIFIER = "Grey_matter"
MYOCARDIUM_IDENTIFIER = "---" 
SPLEEN_IDENTIFIER = "spleen"
GUC_IDENTIFIER = "---"


# -----------------------------------------------------------------------
# -------------The rest is methods to load the yaml files----------------
# -----------------------------------------------------------------------
all_files = [x for x in glob.glob(PATH_TO_STLS + "/*.stl")]

VESSELPATHS = [
    item for item in all_files if VESSEL_ARTERYLIKE in item or VESSEL_VEINLIKE in item
]
identifiers = {
    MUSCLE_IDENTIFIER,
    LUNG_IDENTIFIER,
    LIVER_IDENTIFIER,
    GREY_MATTER_IDENTIFIER,
    MYOCARDIUM_IDENTIFIER,
    SPLEEN_IDENTIFIER,
    GUC_IDENTIFIER,
    "adrenal",
    "intest",
    "throat",
    "stomach",
    "kidney",
    "Kidney",
    "pancreas",
    "Pancreas",
}


def stop_condition(simulation):
    return (
        simulation.time >= TIME_LIMIT_S
    )  # stop condition gets checked for simulations, other may be specified


def get_k_name(name):
    if MUSCLE_IDENTIFIER in name:
        return "Muscle"
    elif LUNG_IDENTIFIER in name:
        return "Lung"
    elif LIVER_IDENTIFIER in name:
        return "Liver"
    elif GREY_MATTER_IDENTIFIER in name:
        return "Grey_matter"
    elif MYOCARDIUM_IDENTIFIER in name:
        return "Myocardium"
    elif SPLEEN_IDENTIFIER in name:
        return "Spleen"
    elif GUC_IDENTIFIER in name:
        return "GUC_lesions"
    else:
        return "All"


def get_roi_name(name):
    for ident in identifiers + [
        "intest",
        "throat",
        "stomach",
        "kidney",
        "Kidney",
        "pancreas",
        "Pancreas",
        "Grey_matter",
        "Myocardium",
        "GUC_lesions",
        "adrenal",
    ]:
        if ident in name:
            return ident


def reset_identifiers(new_idents):
    identifiers = new_idents


# lookup rate constants
def get_k_values(name):
    k_values = dict()
    k_values["Muscle"] = [0.026, 0.249, 0.016]  # K1, k2, k3, k4 as 0
    k_values["Lung"] = [0.023, 0.205, 0.001]
    k_values["Liver"] = [0.660, 0.765, 0.002]
    k_values["Grey_matter"] = [0.107, 0.165, 0.067]
    k_values["All"] = [0.553, 1.213, 0.039]
    k_values["Myocardium"] = [0.832, 2.651, 0.099]
    k_values["Spleen"] = [1.593, 2.867, 0.006]
    k_values["GUC_lesions"] = [
        0.022,
        0.296,
        0.771,
    ]  # TODO change back later, these values were changed

    if MUSCLE_IDENTIFIER in name:
        return k_values["Muscle"]
    elif LUNG_IDENTIFIER in name:
        return k_values["Lung"]
    elif LIVER_IDENTIFIER in name:
        return k_values["Liver"]
    elif GREY_MATTER_IDENTIFIER in name:
        return k_values["Grey_matter"]
    elif MYOCARDIUM_IDENTIFIER in name:
        return k_values["Myocardium"]
    elif SPLEEN_IDENTIFIER in name:
        return k_values["Spleen"]
    elif GUC_IDENTIFIER in name:
        return k_values["GUC_lesions"]
    else:
        return k_values["All"]


# store metadata about run simulation in results folder
def store_sim_info(folder, time_taken):
    simulation_information = dict()
    speed_fun = "interpolate"
    if USE_VOL_SPEED:
        speed_fun = "vol if available"
    simulation_information["speed_function"] = speed_fun
    simulation_information["roi_connections"] = (
        "inner first, all to closest, vein at inner vessel"
    )
    simulation_information["artery_decision"] = "Flow"
    simulation_information["roi_decision"] = "Concentration"
    simulation_information["vein_decision"] = "Flow"
    simulation_information["vessels"] = PATH_TO_VESSELS
    simulation_information["roi_map"] = (
        MUSCLE_IDENTIFIER
        + ", "
        + LUNG_IDENTIFIER
        + ", "
        + LIVER_IDENTIFIER
        + ", "
        + GREY_MATTER_IDENTIFIER
        + ", "
        + MYOCARDIUM_IDENTIFIER
        + ", "
        + SPLEEN_IDENTIFIER
        + ", "
        + GUC_IDENTIFIER
    )
    simulation_information["cell_count"] = CELL_COUNT
    simulation_information["time"] = TIME_LIMIT_S
    simulation_information["injection"] = "ARMINJECTION: " + str(ARMINJECTION)
    simulation_information["injection_timespan"] = MAXDELAY
    simulation_information["SIMDURATION"] = time_taken
    descriptionpath = folder
    with open(folder + "/sim_settings.txt", "w") as file:
        # Iterate over dictionary items
        for key, value in simulation_information.items():
            # Write key-value pair to file
            file.write(f"{key}: {value}\n")


def store_vessels(
    vessels, name="volumed_vessels_coronary_artery_connected_100_armed_r"
):
    for vessel in vessels:
        vessel.unregister_functions()
    with open(vessel_dir + f"./{name}.pickle", "wb") as handle:
        pickle.dump(vessels, handle, protocol=pickle.HIGHEST_PROTOCOL)


def add_to_simdescription(string):
    assert descriptionpath != None
    with open(descriptionpath + "/sim_settings.txt", "a") as file:
        # Iterate over dictionary items
        file.write("\n" + string)


OVERRIDE_SETTINGS_WITH_FILESETTINGS = True
rate_constants = dict()
region_remapping = dict()


class Settings_from_files:
    """Defines object to read settings from yaml files and update this modules variables accordingly"""

    class Read_setting:
        """Read settings from yaml file, stores filename and content"""

        def __init__(self, path):
            self.name = path.replace("\\", "/").split("/")[-1]
            self.content = self.read_yaml_file(path)

        def read_yaml_file(self, path):
            # load the specified file, lowercase all keys and put all values in a set if region is in the name (for faster checking)
            with open(path, "r") as file:
                settings_read = yaml.safe_load(file)
            self.check_settings_valid(settings_read)
            return settings_read

        def check_settings_valid(self, read_settings):
            assert all(
                [
                    read_settings[setting] != None
                    for k, setting in enumerate(read_settings)
                ]
            ), (
                "Value missing for settings:"
                + self.name
                + str(
                    [
                        (k, setting)
                        for k, setting in enumerate(read_settings)
                        if read_settings[setting] == None
                    ]
                )
            )

    def __init__(self, path):
        self.settings_files = glob.glob(path + "/*.yaml")
        self.read_settings = [
            Settings_from_files.Read_setting(p)
            for p in self.settings_files
            if "calibration" not in p and "coefficient" not in p
        ]
        self.rate_regions = None
        self.id_regions = None
        self.concentration_regions = None
        for d in self.read_settings:
            if d.name == "region_id_remappings.yaml":
                self.id_regions = d
            elif d.name == "regions_rate_constants.yaml":
                self.rate_regions = d
            elif d.name == "region_concentration_mapping.yaml":
                self.concentration_regions = d
        self.valid_regions()

    # methods to load informations
    def valid_regions(self):
        """
        Checks whether every region with specified identifiers also has specified rate constants
        """
        # logger.print(self.settings_files   )
        # logger.print(self.rate_regions)
        logger.print(self.id_regions)
        # logger.print(self.concentration_regions)
        assert (
            self.rate_regions is not None
            and self.id_regions is not None
            and self.concentration_regions is not None
        ), "Setting files missing"

        for id_region in self.id_regions.content:
            assert (
                id_region in self.rate_regions.content
            ), "Cannot load ROI without specified rate-constants! Specify them in regions_rate_constants.yaml using the same key (key: value)"
        for new_mapping in self.concentration_regions.content:
            for id_region in self.concentration_regions.content[new_mapping]:
                assert (
                    id_region in self.rate_regions.content
                ), "Cannot find ROI ID without corresponding specified rate-constants! Specify them in regions_rate_constants.yaml using the same key (key: value)"

    def get_rate_constant(self, named_file):
        """Return the rate constants after matching with identifiers and rate constants

        Args:
            named_file (_type_): _description_
        """
        identifier = self.get_identifier(named_file)
        return self.get_rate_constant_identifier(identifier)

    def get_identifier(self, substring):
        as_list = lambda obj: [obj] if type(obj) is not list else obj
        for region in self.id_regions.content:
            for elem_strin in as_list(self.id_regions.content[region]):
                if elem_strin in substring:
                    return region
        return None

    def get_rate_constant_identifier(self, identifier):
        """Return rate constants given an identifier string and load settings

        Args:
            identifier (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.rate_regions.content[identifier] if identifier is not None else None

    def get_concentration_identifier(self, rate_identifier):
        """get_concentration_identifier Return concentration key for given rate identifier

        Args:
            rate_identifier (str): identifying string
        """
        return (
            self.concentration_regions.content[rate_identifier]
            if rate_identifier in self.concentration_regions.content
            else rate_identifier
        )

    def overwrite_settings_coded(self, class_object_holding_settings):
        """Overrides the attributes of a given class containing attributes with the same identifying string

        Args:
            class_object_holding_settings (object): class object with associated "settings" attributes to be overwritten
        """
        for read_setting in self.read_settings:
            if read_setting == self.id_regions:
                setattr(
                    class_object_holding_settings, str("region_remapping"), read_setting
                )
            if read_setting == self.rate_regions:
                setattr(
                    class_object_holding_settings, str("rate_constants"), read_setting
                )
                # assert False,self.id_regions.content#if read_setting not in {self.concentration_regions}:#self.id_regions,self.rate_regions
            for entry in read_setting.content:
                if (
                    hasattr(class_object_holding_settings, str(entry))
                    and read_setting.content[entry] != "DEFAULT"
                ):
                    setattr(
                        class_object_holding_settings,
                        str(entry),
                        read_setting.content[entry],
                    )
                else:
                    logger.print(
                        "Class does not have an attribute '" + str(entry) + "'"
                    )
        class_object_holding_settings.reset_identifiers(self.id_regions)


settings_from_files = Settings_from_files(parent_dir)
