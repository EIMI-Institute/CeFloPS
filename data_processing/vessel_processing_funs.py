from tqdm.auto import tqdm
from joblib import Parallel
import os
import sys

module_path = os.path.abspath(os.path.join("./../"))
if module_path not in sys.path:
    sys.path.append(module_path)
from joblib import Parallel, delayed
import argparse
import glob
import re
import math
import os
import numpy as np
import trimesh
import scipy
import random
import sys, os, argparse
import CeFloPS.simulation.settings as settings

__author__ = "Tobias Hengsbach"
__email__ = "thengsba@uni-muenster.de"

from CeFloPS.simulation.common.functions import *
from CeFloPS.simulation.common.vessel_functions import *
import CeFloPS.simulation.settings as settings
import CeFloPS.simulation.simsetup as simsetup


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


# creating vessels from submeshes
def create_vessels_from_dir():
    min_dinstance_search = settings.GUIDE_POINT_DISTANCE
    distance_between_pathpoints = settings.DISTANCE_BETWEEN_PATHPOINTS
    save_folder = settings.PATH_TO_STLS + "/vessels_split"
    try:
        os.mkdir(save_folder)
    except OSError as error:
        print(error)
    for name in glob.glob(str(settings.PATH_TO_STLS + "/submeshes_split/*.pickle")):
        print(name)
        with open(
            name,
            "rb",
        ) as input_file:
            submeshes, names = pickle.load(input_file)
        print(len(submeshes), names)
        m = names[0][0:-2].replace("\\", "/").split("/")[-1]

        if not f"{save_folder}/{m}_vessels.pickle" in list(
            [
                name.replace("\\", "/")
                for name in glob.glob(
                    str(settings.PATH_TO_STLS + "/vessels_split/*.pickle")
                )
            ]
        ):
            vessels = process_submeshes(
                submeshes,
                [name[0:-2] for name in names],
                min_dinstance_search,
                distance_between_pathpoints,
                parallel=True,
                del_submeshes=True,
            )

            print(names[0][0:-2], m)
            vessels_and_names = vessels, names
            with open(f"{save_folder}/{m}_vessels.pickle", "wb") as handle:
                pickle.dump(vessels_and_names, handle, protocol=pickle.HIGHEST_PROTOCOL)


# craetion of vessels
# loading and saving submeshes
def load_submeshes_from_dir(ignore_string=None):
    path_saved_vessels = settings.PATH_TO_VESSELOUTPUT
    try:
        os.mkdir(path_saved_vessels)
    except OSError as error:
        print(error)

    debug = True
    file_or_folder_name = settings.PATH_TO_STLS + "/*"  
    print("---------")
    mesh_names = []
    print("Found stl Files in ", file_or_folder_name)
    print(
        "Checking against identifiers",
        settings.VESSEL_ARTERYLIKE,
        settings.VESSEL_VEINLIKE,
    )
    for name in glob.glob(str(file_or_folder_name)):
        print(name)
        stlRegex = re.compile(r".stl$")
        mo1 = stlRegex.search(name)
        if mo1 != None:
            # if("pulmonary" not in name):
            if (
                settings.VESSEL_VEINLIKE
                in name[len(file_or_folder_name) - 1 : len(file_or_folder_name) + 4]
                or settings.VESSEL_ARTERYLIKE
                in name[len(file_or_folder_name) - 1 : len(file_or_folder_name) + 4]
            ):
                if ignore_string is not None:
                    if (
                        ignore_string
                        not in name[
                            len(file_or_folder_name) - 1 : len(file_or_folder_name) + 4
                        ]
                    ):
                        mesh_names.append(name)
                else:
                    mesh_names.append(name)

    for name in mesh_names:
        print(name)
    if not mesh_names:
        print("No matching files found, aborting")

    with ProgressParallel(n_jobs=-2) as parallel:
        submeshes_names = parallel(
            delayed(load_submeshes)(
                meshname, settings.PATH_TO_STLS + "/submeshes_split"
            )
            for meshname in mesh_names  # if "heart" in meshname
        )
    return submeshes_names


def load_submeshes_meshnames_vessels():
    all_vessels = []
    for n in list(
        [
            name.replace("\\", "/")
            for name in glob.glob(
                str(settings.PATH_TO_STLS + "/vessels_split/*.pickle")
            )
        ]
    ):
        with open(
            n,
            "rb",
        ) as input_file:
            vessels, names = pickle.load(input_file)
        all_vessels += vessels
    vessels = all_vessels
    len(vessels)
    submeshes = []
    meshnames = []
    for name in glob.glob(str(settings.PATH_TO_STLS + "/submeshes_split/*.pickle")):
        print(name)
        with open(
            name,
            "rb",
        ) as input_file:
            l = pickle.load(input_file)
            submeshes += l[0]
            meshnames += l[1]
    return submeshes, meshnames, vessels


# ----------------------------------------
from trimesh.collision import CollisionManager


def check_intersection(mesh1, mesh2, return_intersection_points=False):
    # Check bounding box overlap first for early exit
    bbox1 = mesh1.bounds
    bbox2 = mesh2.bounds

    if not ((bbox1[1] >= bbox2[0]).all() and (bbox2[1] >= bbox1[0]).all()):
        if return_intersection_points:
            return False, []
        return False

    # Create a Collision Manager for mesh1
    manager1 = CollisionManager()
    manager1.add_object("mesh1", mesh1)

    # Check collision using the manager
    collided_initial = manager1.in_collision_single(mesh2)
    collided = False
    intersection_points = []

    if collided_initial:
        # Compute the intersection mesh to verify
        intersection_mesh = mesh1.intersection(mesh2)
        if not intersection_mesh.is_empty:
            collided = True
            intersection_points = intersection_mesh.vertices.tolist()

    if return_intersection_points:
        return collided, intersection_points
    return collided


def convert_extralinks_for_vessels(vessels, conn):
    extra = []
    for v, link, tag in conn:
        new_con = [None, None, tag]
        new_con[0] = [vessel for vessel in vessels if vessel.volume == v.volume][0]
        link.source_vessel = new_con[0]
        link.target_vessel = [
            vessel for vessel in vessels if vessel.volume == link.target_vessel.volume
        ][0]
        new_con[1] = link
        extra.append(new_con)
    return extra


def remove_all_link_vessels(dlink):
    to_rem = [
        link
        for link in dlink.source_vessel.links_to_path
        if link.target_vessel == dlink.target_vessel
    ]
    for l in to_rem:
        dlink.source_vessel.links_to_path.remove(l)
    to_rem = [
        link
        for link in dlink.target_vessel.links_to_path
        if link.target_vessel == dlink.source_vessel
    ]
    for l in to_rem:
        dlink.target_vessel.links_to_path.remove(l)


import json


def export_vessel_to_json(): ...
def export_to_json_vis(vessels):
    def convert_to_idlinks(linklist):
        idlinks = []
        for link in linklist:
            idlinks.append(
                (
                    link.source_vessel.id,
                    link.target_vessel.id,
                    link.source_index,
                    link.target_index,
                )
            )
        return idlinks

    def convert_to_idvoilinks(linklist):
        idlinks = []
        for link in linklist:
            idlinks.append(
                (
                    link.source_vessel.id,
                    link.target_tissue.name,
                    link.source_index,
                    link.target_index,
                )
            )
        return idlinks

    def vessel_to_vessel_dict(vessel):
        self.speed_function = speed_function
        self.path = path
        self.fully_profiled = False
        self.diameter_a = diameter_a
        self.diameter_b = diameter_b
        self.avg_diameter = avg_diameter
        self.volumes = []
        self.diameters = diameters
        self.associated_vesselname = associated_vesselname

        self.length = sum(
            [
                calculate_distance_euclidian(self.path[i], self.path[i + 1])
                for i in range(len(self.path) - 1)
            ]
        )
        self.links_to_path = []
        self.tags = []
        self.cuts = None
        self.type = "artery"
        self.reachable_rois = []  # [[lower_index, roi], ...]
        if settings.VESSEL_VEINLIKE in self.associated_vesselname:
            self.type = "vein"
        self.times = self.calculate_times()
        self.distances = [
            calculate_distance_euclidian(self.path[i], self.path[i + 1])
            for i in range(len(self.path) - 1)
        ]
        self.id = f"{id(self)}_{self.associated_vesselname[0:15]}_{len(self.path)}"

        self.profiles = dict()
        self.links_to_vois = []

        return {
            "vertices": [tuple(p) for p in vessel.path],
            "diameters": list(vessel.diameters),
            "vessname": vessel.associated_vesselname,
            "type": "red" if vessel.type == "artery" else "blue",
            "id": vessel.id,
            "times": [float(sum(vessel.times))],
            "links": convert_to_idlinks(vessel.links_to_path),
            "voilinks": convert_to_idvoilinks(vessel.links_to_vois),
            "volumes": convert_volumes(vessel.volumes_to_path),
            "speed_funs": [],
        }

    data = []
    for vessel in vessels:
        data.append(vessel_dict(vessel))
    return data
    with open("data.json", "w") as f:
        json.dump(export_to_json_vis(vessels), f)


def import_vessel_from_json(path_to_file): ...


def export_vessels_to_json(): ...
def import_vessels_from_json(path_to_folder):
    for vesselfile in glob.glob(path_to_folder + "*_vessel.json"):
        ...
