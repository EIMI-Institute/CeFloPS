#combine vessels into one pickle object

import json 
import glob
import pickle
from tqdm import tqdm 
import os
import sys 

module_path = os.path.abspath(os.path.join("./../.."))
if module_path not in sys.path:
    sys.path.append(os.path.join("./../.."))  
    sys.path.append(os.path.join("./.."))   
import CeFloPS.simulation.settings as settings 
from CeFloPS.simulation.common.functions import *
from CeFloPS.simulation.common.vessel_functions import *
import CeFloPS.simulation.settings as settings
import CeFloPS.simulation.simsetup as simsetup
import logging
import traceback
from CeFloPS.data_processing.vessel_processing_funs import *
from logger_config import setup_logger, set_global_log_level

path_to_vesselsplit=str(settings.PATH_TO_STLS) + "/vessels_split"
 
vessels=[]
for filepath in glob.glob(path_to_vesselsplit+"/*.pickle"):
    with open(
                filepath,
                "rb",
            ) as input_file:
                vessels.append(pickle.load(input_file))

with open("./vessels_combined.pickle", "wb") as handle:
        pickle.dump(vessels, handle, protocol=pickle.HIGHEST_PROTOCOL)