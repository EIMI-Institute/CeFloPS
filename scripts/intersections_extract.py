# script to calculate the intersection points between trimesh loadable meshfiles.
# saves a file: intersections.yaml with the meshfiles name as a key, the intersected meshfiles name and centroid of the intersection

import trimesh
import glob
import json
import pickle
import os
import sys
from tqdm import tqdm
import numpy as np

module_path = os.path.abspath(os.path.join("./../.."))
if module_path not in sys.path:
    sys.path.append(os.path.join("./../.."))
    sys.path.append(os.path.join("./.."))

import CeFloPS.simulation.settings as settings
from CeFloPS.data_processing.vessel_processing_funs import check_intersection


def export_intersect_json(pairs, points, path):
    intersections = dict()
    assert len(pairs) == len(points)
    for pair, ppoints in zip(pairs, points):
        points, average = ppoints
        # Check if the string version of the tuple of average is in intersections
        avg_str = str(tuple(average))
        if avg_str not in intersections:
            intersections[avg_str] = set()
        intersections[avg_str].add(pair[0])
        # should collect multicollisions
        intersections[avg_str].add(pair[1])
    for se in intersections:
        intersections[se] = list(intersections[se])
    with open(path + "/intersections.json", "w") as f:
        json.dump(intersections, f)


if __name__ == "__main__":
    path_to_pickles = str(settings.PATH_TO_STLS) + "/submeshes_split/*.pickle"
    path_to_vesselsplit = str(settings.PATH_TO_STLS) + "/vessels_split"
    submeshes = []

    for file_name in glob.glob(path_to_pickles):
        print(file_name)
        with open(
            file_name,
            "rb",
        ) as input_file:
            subs, names = pickle.load(
                input_file, fix_imports=True, encoding="ASCII", errors="strict"
            )
            subs = list(subs) 
            for i, sub in enumerate(subs):
                new_value = (
                    sub.metadata.get("file_name") or
                    sub.metadata.get("name") or
                    sub.metadata.get("node") or
                    "unknown"
                ) + "_" + str(i)
                # seems like trimesh removed file name in favor of name
                if "file_name" in sub.metadata:
                    sub.metadata["file_name"] = new_value
                elif "name" in sub.metadata:
                    sub.metadata["name"] = new_value
                elif "node" in sub.metadata:
                    sub.metadata["node"] = new_value
                else:
                    sub.metadata["name"] = new_value
            print(subs)
            submeshes += subs

    print("#Submeshes: %s" % len(submeshes))
    print(submeshes[0].__dict__)
    meshes = submeshes
    intersecting_pairs = []
    intersection_points = []
    for i in tqdm(range(len(meshes))):
        for j in range(i + 1, len(meshes)):
            collided = False
            collided, points_of_col = check_intersection(
                meshes[i], meshes[j], return_intersection_points=True
            )
            if collided:
                intersecting_pairs.append(
                    (meshes[i].metadata["file_name"], meshes[j].metadata["file_name"])
                )
                avg = np.average(np.asarray(points_of_col), axis=0)
                assert len(avg) == 3, (
                    meshes[i].metadata["file_name"],
                    meshes[j].metadata["file_name"],
                )
                intersection_points.append((points_of_col, avg))

    export_intersect_json(intersecting_pairs, intersection_points, path_to_vesselsplit)
