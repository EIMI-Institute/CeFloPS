import concurrent.futures
import glob
import pickle
from tqdm import tqdm
from tqdm.auto import tqdm
import os
import sys
import re
import math
import numpy as np
import trimesh

module_path = os.path.abspath(os.path.join("./../.."))
if module_path not in sys.path:
    sys.path.append(os.path.join("./../.."))
    sys.path.append(os.path.join("./.."))
import sys, os, argparse
import CeFloPS.simulation.settings as settings
from CeFloPS.simulation.common.functions import *
from CeFloPS.simulation.common.vessel_functions import *
import CeFloPS.simulation.settings as settings
import CeFloPS.simulation.simsetup as simsetup
import logging
import traceback
from CeFloPS.data_processing.submesh_processing import process_single_submesh
from CeFloPS.data_processing.vessel_processing_funs import *
from logger_config import setup_logger, set_global_log_level

# Configure logger for this module with a specific level

logger = setup_logger(__name__, level=logging.WARNING)
set_global_log_level(logging.WARNING)


def process_submesh(
    submesh, name, integer, min_distance_search, distance_between_pathpoints
):
    try:
        vessel = process_single_submesh(submesh, integer)
        return vessel
    except Exception as e:
        error_message = f"Error processing submesh {name}: {str(e)}"
        error_message += "\n" + traceback.format_exc()
        log_error(error_message)
        return None


def log_error(message):
    with open(
        str(settings.PATH_TO_STLS) + "/vessels_split/" + "error_log.txt", "a"
    ) as log_file:
        log_file.write(message + "\n")


def load_and_submit_submeshes(
    file_name, save_folder, min_distance_search, distance_between_pathpoints
):
    try:
        with open(file_name, "rb") as input_file:
            # Increase buffering parameter
            submeshes, names = pickle.load(
                input_file, fix_imports=True, encoding="ASCII", errors="strict"
            )
    except (pickle.UnpicklingError, EOFError) as e:
        error_message = f"Error loading pickle file {file_name}: {str(e)}"
        error_message += "\n" + traceback.format_exc()
        log_error(error_message)
        print(error_message)
        return []
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []

        for i, submesh_and_name in enumerate(zip(submeshes, names)):
            submesh, name = submesh_and_name
            print(
                f"Processing submesh: {name}, total submeshes: {len(submeshes)}, total names: {len(names)}"
            )
            submesh_name = name[:-2]
            save_path = file_name.replace("\\", "/").split("/")[-1]
            target_save_path = f"{save_folder}/{save_path}_{i}.pickle"

            # Check if the file already exists
            if not os.path.exists(target_save_path) or True:
                futures.append(
                    executor.submit(
                        process_submesh,
                        submesh,
                        name,
                        i,
                        min_distance_search,
                        distance_between_pathpoints,
                    )
                )

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    results.append((result, submesh_name))
            except Exception as e:
                error_message = f"Error processing future {future}: {str(e)}"
                error_message += "\n" + traceback.format_exc()
                log_error(error_message)

    return results


def main():
    path_to_pickles = str(settings.PATH_TO_STLS) + "/submeshes_split/*.pickle"

    save_folder = str(settings.PATH_TO_STLS) + "/vessels_split"

    try:
        os.mkdir(save_folder)
    except OSError as error:
        print(error)

    min_distance_search = settings.GUIDE_POINT_DISTANCE
    distance_between_pathpoints = settings.DISTANCE_BETWEEN_PATHPOINTS

    pickle_files = glob.glob(path_to_pickles)
    print(path_to_pickles, len(pickle_files))
    if len(pickle_files) == 0:
        load_submeshes_from_dir()  # load and spllit meshes first
        pickle_files = glob.glob(path_to_pickles)
    with tqdm(total=len(pickle_files), desc="Processing Files") as pbar:
        for file_name in pickle_files:
            try:
                results = load_and_submit_submeshes(
                    file_name,
                    save_folder,
                    min_distance_search,
                    distance_between_pathpoints,
                )
                print(f"Number of files to save: {len(results)}")
                for i, vessel_and_name in enumerate(results):
                    vessels, name = vessel_and_name
                    save_path = file_name.replace("\\", "/").split("/")[-1]
                    target_save_path = f"{save_folder}/{save_path}_{i}.pickle"
                    logger.warning("---" + str(vessels) + "  " + target_save_path)
                    try:
                        with open(target_save_path, "wb") as output_file:
                            pickle.dump(vessels, output_file)
                    except Exception as e:
                        error_message = (
                            f"Error saving file {target_save_path}: {str(e)}"
                        )
                        log_error(error_message)

            except Exception as e:
                error_message = f"Error processing file {file_name}: {str(e)}\n{traceback.format_exc()}"
                log_error(error_message)

            pbar.update(1)


if __name__ == "__main__":
    main()
