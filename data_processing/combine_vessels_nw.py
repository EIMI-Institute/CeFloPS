# %%
import trimesh
import numpy as np
from scipy.spatial import cKDTree
import os
import sys

module_path = os.path.abspath(os.path.join("./../.."))
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
from CeFloPS.data_processing.vessel_processing_funs import *
import visualization as vis
from CeFloPS.data_processing.submesh_processing import (
    extract_centerline,
    extract_diameter_and_centre_from_cut,
    extract_diameter_and_centre_from_cut_two_sided,
    filter_centerline_by_distance,
)
from CeFloPS.simulation.common.vessel2 import Vessel,Link


# %%
old_vessels = simsetup.load_vessels()

# %%
# take all new scaled brain vessels, omit these from the old ones. delete heart vessel substitute for chamber
old_vessels_trimmed = [
    v
    for v in old_vessels
    if "jump" not in v.associated_vesselname and "brain" not in v.associated_vesselname
]
# new_vessels_trimmed=[v for v in new_vessels]


# %%
import numpy as np


def find_similar_sublist(list_a, list_b, checking_reverse=False, max_threshold=20):
    """
    Checks which list contains a subsequence with low distance to the other list, considering the inverted version of list A.
    Returns:
    - Tuple (matched_list_name, start_index, end_index, avg_distance, is_reversed) or None if no match
    """

    def get_best_match(short_list, long_list):
        min_distance = float("inf")
        best_index = -1
        len_short = len(short_list)
        len_long = len(long_list)

        if len_short == 0 or len_long < len_short:
            return None, None

        short_arr = np.array(short_list)
        long_arr = np.array(long_list)

        for i in range(len_long - len_short + 1):
            window = long_arr[i : i + len_short]
            distances = np.linalg.norm(short_arr - window, axis=1)
            avg_distance = np.mean(distances)

            if avg_distance < min_distance:
                min_distance = avg_distance
                best_index = i

        return best_index, min_distance

    a = np.array(list_a)
    b = np.array(list_b)
    results = []

    # Check if list_a is a subsequence of list_b
    if len(a) <= len(b):
        idx, dist = get_best_match(a, b)
        if idx is not None:
            end_idx = idx + len(a) - 1
            results.append(("A in B", idx, end_idx, dist, 0))

    # Check if list_b is a subsequence of list_a
    if len(b) <= len(a):
        idx, dist = get_best_match(b, a)
        if idx is not None:
            end_idx = idx + len(b) - 1
            results.append(("B in A", idx, end_idx, dist, 0))

    if not checking_reverse:
        # Create a reversed copy of list_a to avoid modifying the original
        reversed_a = list_a[::-1]
        reversed_result = find_similar_sublist(
            reversed_a, list_b, checking_reverse=True, max_threshold=max_threshold
        )
        if reversed_result is not None:
            dire, rev_start, rev_end, rev_dist, _ = reversed_result
            # Check if the reversed result was A in B or B in A and adjust direction
            if dire == "A in B":
                new_dire = "A reversed in B"
                # Indices are in B, no adjustment needed
                results.append((new_dire, rev_start, rev_end, rev_dist, 1))
            elif dire == "B in A":
                new_dire = "B in A reversed"
                # Convert indices from reversed_a to original list_a's indices
                n = len(reversed_a)
                start_original = n - 1 - rev_end
                end_original = n - 1 - rev_start
                results.append((new_dire, start_original, end_original, rev_dist, 1))

    if not results:
        return None

    best_match = min(results, key=lambda x: x[3])

    if best_match[3] <= max_threshold:
        return (
            best_match[0],
            best_match[1],
            best_match[2],
            best_match[3],
            best_match[4],
        )

    return None


list_a = [
    np.array([206.71289634, 273.66202685, 334.59037629]),
    np.array([205.90078897, 273.26953801, 334.1585961]),
    np.array([205.12363785, 272.86419674, 333.6772079]),
]

list_b = [
    np.array([206.71289634, 272.66202685, 334.59037629]),
    np.array([206.90078897, 273.26953801, 334.1585961]),
    np.array([205.12363785, 272.86419674, 333.6772079]),
    np.array([204.37935233, 272.42774637, 333.17168823]),
]
# list_a.reverse()
# list_b.reverse()

result = find_similar_sublist(list_a, list_b)
print("Best match:", result)

# %%
def find_matching_vessel(vesspath1, vessradii1, vesspath2):
    """find_matching_vessel Checks wether a second path of 3d points lies within the region defined by vesspath1 and vessradii1 as the radial space around the points that define cylinder spaces connected as a list.
        Gives back the amount of coverage(how many points of path2 lie inside of the first paths and radii defined cylinder volumes)

    Args:
        vesspath1 (_type_): list of 3d points
        vessradii1 (_type_): list of matching radii for those points, orthohonal to the vector pointing to the following points
        vesspath2 (_type_): list of 3d points
    """
    coverage = 0
    for p in vesspath2:
        p = np.array(p)
        # Check if the point is within any of the spheres
        for i in range(len(vesspath1)):
            A = np.array(vesspath1[i])
            r = vessradii1[i]
            if np.linalg.norm(p - A) <= r:
                coverage += 1
                break
        else:
            # Check if the point is within any of the cylinders
            for i in range(len(vesspath1) - 1):
                A = np.array(vesspath1[i])
                B = np.array(vesspath1[i + 1])
                r = vessradii1[i]
                vector_AB = B - A
                vector_AP = p - A
                len_sq_AB = np.dot(vector_AB, vector_AB)
                if len_sq_AB == 0:
                    continue  # Already checked as a sphere
                t = np.dot(vector_AP, vector_AB) / len_sq_AB
                t_clamped = max(0.0, min(1.0, t))
                closest_point = A + t_clamped * vector_AB
                distance = np.linalg.norm(p - closest_point)
                if distance <= r:
                    coverage += 1
                    break
    return (
        coverage,
        np.linalg.norm(np.asarray(vesspath1[0]) - np.asarray(vesspath2[0])),
        np.linalg.norm(np.asarray(vesspath1[0]) - np.asarray(vesspath2[-1])),
    )

# %%
path_to_vesselsplit = r"pathto\vesssplit\vessels_combined.pickle"

new_vessels = []
for filepath in glob.glob(path_to_vesselsplit):
    with open(
        filepath,
        "rb",
    ) as input_file:
        new_vessels += pickle.load(input_file)

# %%
def find_string_match(stringlist, substrings):
    matches = []
    for string in stringlist:
        for substring in substrings:
            app = True
            if substring not in string:
                app = False
        if app:
            matches.append(string)
    return matches

# %%
import visualization as vis
if False:
    vis.show(
        [(v.path, [200, 100, 100, 100]) for v in new_vessels if v is not None]
        + [(v.path, [100, 100, 200, 100]) for v in old_vessels]
    )


# %%
# load intersections

import json

with open(r"pathto\exported_intersections_added_connection_edit_brain2_isolated_leg.json", "r") as f:
    intersection_data = json.load(f)

# %% [markdown]
# Apply and load splits, create links from intersections:

# %%
import json

with open(
    r"pathto\micreduced\mic_vesselgraph\export_vessel_data.json", "r"
) as f:
    split_data = json.load(f)

# %%
def blender_name(vessel):
    return vessel.associated_vesselname + "_" + str(vessel.integer)

# %%
def create_dia_a_b(diameters):
    scope_i = round(len(diameters) // 10)
    avg_diameter = np.mean(diameters)

    ausschnitt = diameters[:scope_i]
    x = 1
    while len(ausschnitt) < 2 and len(ausschnitt) != len(diameters):
        x += 1
        scope_i = round((len(diameters) // 10) + x)
        ausschnitt = diameters[:scope_i]
    path_diameter_a = np.mean(ausschnitt)

    ausschnitt = diameters[len(diameters) - scope_i :]
    x = 1
    while len(ausschnitt) < 1 and len(ausschnitt) != len(diameters):
        x += 1
        scope_i = round((len(diameters) // 10) + x)
        ausschnitt = diameters[:scope_i]
    path_diameter_b = np.mean(ausschnitt)
    return path_diameter_a, path_diameter_b

# %%
def apply_splits(vessels, split_data):
    new_vessels = []
    for vessel in vessels:
        for split in split_data["vessels"]:
            if (
                blender_name(vessel) == split["name"]
                and len(split["split_indices"]) != 0
            ):
                assert len(split["split_indices"]) <= 1, (
                    "More split points not implemented " + str(split["split_indices"])
                )
                vessel.split_point = split["split_indices"]
                # pulmonary omits the smalller part
                new_vessel = None
                if "sys_pul_art_0" in vessel.associated_vesselname:
                    print("pulmon", vessel.associated_vesselname)
                    if (
                        len(vessel.path) - split["split_indices"][0]
                        < len(vessel.path) / 2
                    ):
                        vessel.reduce(0, split["split_indices"][0] + 1)
                    else:
                        vessel.reduce(split["split_indices"][0], len(vessel.path) + 1)
                # coronary sets connected to flow toward the splitting point
                elif "cor_vein" in vessel.associated_vesselname:
                    print("coron", vessel.associated_vesselname)
                    other_part_path, other_part_dia = (
                        vessel.path[split["split_indices"][0] :],
                        vessel.diameters[split["split_indices"][0] :],
                    )
                    other_a, other_b = create_dia_a_b(other_part_dia)
                    vessel.reduce(0, split["split_indices"][0] + 1)
                    new_vessel = Vessel(
                        other_part_path,
                        other_a,
                        other_b,
                        np.mean(other_part_dia),
                        other_part_dia,
                        vessel.associated_vesselname,
                        vessel.speed_function,
                    )
                    new_vessel.integer = vessel.integer
                    vessel.integer = -1
                    # set directions (split at upper side of both plus a marker)
                    new_vessel.reverse()
                    vessel.tags.append("coronary_directed")
                    new_vessel.tags.append("coronary_directed")
                    #also craete a link between them
                    vessel.add_link(Link(vessel,len(vessel.path)-1,new_vessel,len(new_vessel.path)-1))
                    new_vessel.add_link(Link(new_vessel,len(new_vessel.path)-1,vessel,len(vessel.path)-1))
                    #vessel.add_link(Link(vessel,len(vessel.path)-1,new_vessel,0))
                    #new_vessel.add_link(Link(new_vessel,0,vessel,len(vessel.path)-1))
                else:
                    print("other", vessel.associated_vesselname)
                    # rest just divides the vessel
                    other_part_path, other_part_dia = (
                        vessel.path[split["split_indices"][0] :],
                        vessel.diameters[split["split_indices"][0] :],
                    )

                    other_a, other_b = create_dia_a_b(other_part_dia)
                    vessel.reduce(0, split["split_indices"][0] + 1)
                    new_vessel = Vessel(
                        other_part_path,
                        other_a,
                        other_b,
                        np.mean(other_part_dia),
                        other_part_dia,
                        vessel.associated_vesselname,
                        vessel.speed_function,
                    )
                    new_vessel.integer = vessel.integer
                    vessel.tags.append("reduced");new_vessel.tags.append("reduced")
                if new_vessel is not None:
                    new_vessels.append(new_vessel)
    return new_vessels

# %%
split_vessels = apply_splits(new_vessels, split_data)

# %%
len(split_vessels)

# %%
import visualization as vis

if False:
    vis.show(
        [(v.path, [200, 100, 100, 200]) for v in split_vessels]
        + [(v.path, [100, 100, 100, 100]) for v in new_vessels]
    )

# %%
#new_vessels += split_vessels

# %%
vessels = new_vessels+split_vessels
# %% [markdown]
# set links  afterwards obtain volumes with frustum

# %%
def closest_pathindex(points, target_point):
    """
    Find the index of the point in the list that is closest to the given target point.

    Parameters:
    points (list of tuples): A list of tuples where each tuple represents a 3D point (x, y, z).
    target_point (tuple): The target 3D point (x, y, z).

    Returns:
    int: The index of the point in the list that is closest to the target point.
    """

    # Convert the list of points and target point to numpy arrays
    points_array = np.array(points)
    target_array = np.array(target_point)

    # Calculate the Euclidean distances
    distances = np.linalg.norm(points_array - target_array, axis=1)

    # Find the index of the minimum distance
    closest_index = np.argmin(distances)

    return closest_index

# %%
def is_index_used_in_links(vessel, start_index, links, check_to_end=True):
    """
    Checks if any target index from either start_index to the end or start_index to the beginning
    is used in the links list for the given vessel.

    Parameters:
    vessel (Vessel): The vessel to check.
    start_index (int): The starting index to check from.
    links (list of Link): The list of links to search through.
    check_to_end (bool): If True, check from start_index to path end; if False, check to path start.

    Returns:
    bool: True if any target index in the defined range is used; False otherwise.
    """
    path_length = len(vessel.path)

    for link in links:
        if link.target_vessel == vessel:
            if check_to_end:
                if start_index <= link.target_index < path_length:
                    return True
            else:
                if 0 <= link.target_index <= start_index:
                    return True
    return False


# %% [markdown]
# Find all intersection point connections present

# %%
# Start with an empty list of sets
sets = []

for intersect_group in intersection_data.values():
    # This will track which sets need to be merged
    merging_sets = []
    
    # Iterate through current sets to find intersections
    for s in sets:
        if s.intersection(intersect_group):
            merging_sets.append(s)
    
    # If we have sets to merge, merge them along with the current group
    if merging_sets:
        # Create a new set that combines all intersecting sets and the new group
        new_set = set(intersect_group)
        for s in merging_sets:
            new_set.update(s)
            # Remove merged sets from the original list
            sets.remove(s)
        sets.append(new_set)
    else:
        # No intersections found, so add the group as a new set
        sets.append(set(intersect_group))

# At this point, `sets` contains the maximum-size sets possible given the intersections

# %%
""" with open("pathto\\exported_intersections_added_connection.json", 'w') as outfile:
        json.dump(intersection_data, outfile, indent=4) """

# %%
vess_sets=[]
for s in sets:
    it=[]
    for vessel in vessels:
        if blender_name(vessel) in s:
            it.append(vessel)
    vess_sets.append(it)

# %%
if False:
    ...
    #vis.show([(v.path, [200*i%255, 60*i%255, 20*i%255, 200]) for i,x in enumerate(vess_sets) for v in x])

# %%
def connection_exists(source_vessel, target_vessel, links):
    for link in links:
        if (link.source_vessel == source_vessel and link.target_vessel == target_vessel) or \
           (link.source_vessel == target_vessel and link.target_vessel == source_vessel):
            return True
    return False

def calculate_directionality(vessel, point_index):
    # Calculate approximate direction by looking at a few points before and after the index
    pre_index = max(0, point_index - 1)
    post_index = min(len(vessel.path) - 1, point_index + 1)
    
    # Vector from previous point to next to establish directionality
    direction_vector = [
        vessel.path[post_index][k] - vessel.path[pre_index][k] for k in range(3)
    ]
    return direction_vector

def truncate_conditionally(vessel, intersection_index):
    # Determine if truncation is needed based on direction and geometry
    path_length = len(vessel.path)
    if intersection_index > 0 and intersection_index < path_length - 1:
        # Do not truncate if the intersection is a meaningful part of the vessel
        return vessel.path
    # If it's more about connecting ends, consider retaining meaningful points
    return vessel.path

def find_closest_index(vessel, reference_point):
    # Efficiently find the path index with the nearest point to the reference
    return closest_pathindex(vessel.path, reference_point)
import numpy as np
from scipy.spatial import KDTree

class UnionFind:
    """Union-Find data structure for grouping connected components."""
    def __init__(self):
        self.parent = {}
    
    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        fx = self.find(x)
        fy = self.find(y)
        if fx != fy:
            self.parent[fy] = fx

def combine_intersections(intersection_data, spatial_threshold=4):
    """
    Processes intersection data by combining points that are:
    1. Within spatial_threshold distance
    2. Connected through shared vessels (transitively)
    
    Returns a modified copy of the input data with merged points.
    """
    # Create working copy of data
    modified_data = intersection_data.copy()
    
    # Parse all points and their vessel sets
    points = []
    vessel_sets = []
    original_keys = []
    
    for key in list(modified_data.keys()):
        try:
            point = np.array(eval(key))
            points.append(point)
            vessel_sets.append(set(modified_data[key]))
            original_keys.append(key)
        except:
            del modified_data[key]  # Remove invalid entries
    
    if not points:
        return modified_data
    
    # Build spatial index
    kdtree = KDTree(points)
    
    # Find spatial neighborhoods
    clusters = kdtree.query_ball_tree(kdtree, spatial_threshold)
    
    # Union-Find for grouping connected components
    uf = UnionFind()
    for i in range(len(points)):
        uf.add(i)
    
    # Connect points that share vessels in spatial neighborhoods
    for i, neighbors in enumerate(clusters):
        for j in neighbors:
            if i != j and not vessel_sets[i].isdisjoint(vessel_sets[j]):
                uf.union(i, j)
    
    # Group points into final clusters
    clusters = defaultdict(list)
    for i in range(len(points)):
        clusters[uf.find(i)].append(i)
    
    # Process clusters and update data
    new_entries = []
    
    for cluster in clusters.values():
        if len(cluster) < 2:
            continue  # Keep single points as-is
            
        # Calculate cluster centroid
        cluster_points = [points[i] for i in cluster]
        centroid = np.mean(cluster_points, axis=0)
        
        # Combine all vessels from cluster
        combined_vessels = set()
        for i in cluster:
            combined_vessels.update(vessel_sets[i])
        
        # Create new key with controlled precision
        centroid_key = f"({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})"
        
        # Remove old keys and add new entry
        for i in cluster:
            del modified_data[original_keys[i]]
        new_entries.append((centroid_key, sorted(combined_vessels)))
    
    # Add new merged entries
    for key, vessels in new_entries:
        modified_data[key] = vessels
    
    return modified_data
def link_vessels_with_intersections(vessels, intersection_mean_point, links):
    closest_indices = [find_closest_index(v, intersection_mean_point) for v in vessels]
    linklength=lambda l: np.linalg.norm(np.asarray(l.source_vessel.path[l.source_index])-np.asarray(l.target_vessel.path[l.target_index]))
    multi_one_mode=False
    if len(vessels)>2:
        multi_one_mode=True
    if multi_one_mode:
        #connect all to the closest point available, the one available not to self!
        source_vessel_index = np.argmin([np.linalg.norm(v.path[closest_indices[i]] - intersection_mean_point) for i, v in enumerate(vessels)])
        source_vessel = vessels[source_vessel_index]
        source_index = closest_indices[source_vessel_index]
        for j, target_vessel in enumerate(vessels):
                if source_vessel != target_vessel and not connection_exists(source_vessel, target_vessel, links):#only one link between 2 vessels
                    target_index = closest_indices[j]
                    if "reduced" in source_vessel.tags or "reduced" in target_vessel.tags:
                        if linklength(Link(source_vessel, source_index, target_vessel, target_index))>5:
                            continue
                    links.add(Link(source_vessel, source_index, target_vessel, target_index))
                    links.add(Link(target_vessel, target_index, source_vessel, source_index))
        return
    #if multiple then connect all to the closest point available, the one available not to self!
    for i, source_vessel in enumerate(vessels):
        for j, target_vessel in enumerate(vessels):
            if i != j and not connection_exists(source_vessel, target_vessel, links):#only one link between 2 vessels
                source_index = closest_indices[i]
                target_index = closest_indices[j]

                # Truncate based on conditions and directionality if necessary
                source_vessel.path = truncate_conditionally(source_vessel, source_index)
                target_vessel.path = truncate_conditionally(target_vessel, target_index)

                # Create bidirectional links
                if "reduced" in source_vessel.tags or "reduced" in target_vessel.tags:
                        if linklength(Link(source_vessel, source_index, target_vessel, target_index))>5:
                            continue
                links.add(Link(source_vessel, source_index, target_vessel, target_index))
                links.add(Link(target_vessel, target_index, source_vessel, source_index))

def process_intersections(vessels, intersection_data):
    links = set()
    
    for intersection in intersection_data:
        participating_vessels = [
            vess for vess in vessels if blender_name(vess) in intersection_data[intersection]
        ]
        intersection_mean_point = eval(intersection)
        link_vessels_with_intersections(participating_vessels, intersection_mean_point, links)
    
    return links



import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict
from math import acos, degrees

class UnionFind:
    """Union-Find structure with safe initialization"""
    def __init__(self, size):
        self.parent = list(range(size))  # Initialize for all possible indices
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        fx = self.find(x)
        fy = self.find(y)
        if fx != fy:
            self.parent[fy] = fx

def vector_angle(v1, v2):
    """Calculate angle between two vectors in degrees"""
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return degrees(acos(np.clip(cos_theta, -1.0, 1.0)))

def compute_vessel_direction(vessel, num_points=3):
    """Compute average direction vector from vessel path"""
    path = np.array(vessel.path)
    if len(path) < 2:
        return np.zeros(3)
    
    # Use first and last segments for direction
    vectors = []
    for i in range(min(num_points, len(path)-1)):
        vectors.append(path[i+1] - path[i])
    return np.mean(vectors, axis=0)

import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        fx = self.find(x)
        fy = self.find(y)
        if fx != fy:
            self.parent[fy] = fx

def fuse_intersections_by_vessel_diameter(intersection_data, vessels, spatial_buffer=.45):
    """
    Merge intersections that fall within a sphere defined by:
    - Center: Intersection point
    - Radius: spatial_buffer * max_diameter (from participating vessels' closest points)
    """
    # Create working copy and parse data
    modified_data = intersection_data.copy()
    vessel_map = {v.name: v for v in vessels}
    entries = []
    
    # Parse and validate intersections
    valid_keys = []
    for key in list(modified_data.keys()):
        try:
            point = np.array(eval(key))
            vessel_names = modified_data[key]
            if len(vessel_names) < 2:
                del modified_data[key]
                continue
                
            valid_keys.append(key)
            entries.append({
                'point': point,
                'vessels': vessel_names,
                'radius': None
            })
        except:
            del modified_data[key]
    
    if not entries:
        return modified_data
    
    # Precompute radii for all entries
    for i, entry in enumerate(entries):
        max_diameter = 0
        for vessel_name in entry['vessels']:
            vessel = vessel_map.get(vessel_name)
            if not vessel:
                continue
                
            # Find closest path index
            distances = np.linalg.norm(vessel.path - entry['point'], axis=1)
            closest_idx = np.argmin(distances)
            
            # Get diameter at closest index
            if closest_idx < len(vessel.diameters):
                max_diameter = max(max_diameter, vessel.diameters[closest_idx])
        
        # Store radius with buffer
        entries[i]['radius'] = max_diameter * spatial_buffer
    
    # Build spatial index
    points = np.array([e['point'] for e in entries])
    kdtree = KDTree(points)
    
    # Find candidate clusters
    uf = UnionFind(len(entries))
    for i, entry in enumerate(entries):
        # Query all points within this entry's radius
        neighbors = kdtree.query_ball_point(entry['point'], 
                                           r=entry['radius'], 
                                           return_sorted=False)
        for j in neighbors:
            if j != i:
                # Check reciprocal inclusion
                distance = np.linalg.norm(entries[j]['point'] - entry['point'])
                if distance <= entries[j]['radius']:
                    uf.union(i, j)
    
    # Group entries into clusters
    clusters = defaultdict(list)
    for i in range(len(entries)):
        clusters[uf.find(i)].append(i)
    
    # Process clusters and create merged points
    new_entries = []
    for cluster in clusters.values():
        if len(cluster) == 1:
            continue  # Keep single points as-is
        
        # Calculate merged point as centroid
        cluster_points = [entries[i]['point'] for i in cluster]
        centroid = np.mean(cluster_points, axis=0)
        
        # Combine all vessels and calculate precision
        combined_vessels = list({
            v 
            for i in cluster 
            for v in entries[i]['vessels']
        })
        
        # Create new key with controlled precision
        centroid_key = f"({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})"
        
        # Remove old entries
        for i in cluster:
            del modified_data[valid_keys[i]]
        
        # Add new entry
        new_entries.append((centroid_key, combined_vessels))
    
    # Add merged entries to output
    for key, vessels in new_entries:
        modified_data[key] = vessels
    
    return modified_data
for vessel in vessels:
    vessel.name=blender_name(vessel)
intersection_data_close_combined=fuse_intersections_by_vessel_diameter(intersection_data,vessels)
# Run through intersections to establish links
links = process_intersections(vessels, intersection_data_close_combined)
#%%
vis.show_linked(vessels[::],None)


# %%
""" for vessel in vessels:
    if vessel.links
    vessel.links_to_path=[] """
#apply links and fix parallel parts
for link in links:
        link.source_vessel.add_link(link)
#show connections
##vis.show_linked(vessels[::], None)


# %% [markdown]
# check if unconnected ends are free or end in other vessels meshes

# %%
import numpy as np
import numpy as np
from scipy.spatial import KDTree

def precompute_segment_data(vessels):
    """Precompute segment metadata and bounding spheres."""
    segments = []
    for vessel in vessels:
        path = vessel.path
        diameters = vessel.diameters
        for i in range(len(path) - 1):
            start = np.array(path[i])
            end = np.array(path[i + 1])
            start_diam = diameters[i]
            end_diam = diameters[i + 1]
            
            # Bounding sphere parameters
            midpoint = (start + end) / 2
            segment_length = np.linalg.norm(end - start)
            max_radius = max(start_diam, end_diam) / 2
            sphere_radius = segment_length / 2 + max_radius
            
            segments.append({
                'start': start,
                'end': end,
                'start_diam': start_diam,
                'end_diam': end_diam,
                'midpoint': midpoint,
                'sphere_radius': sphere_radius,
                'vessel': vessel
            })
    return segments

def build_spatial_index(segments):
    """Build a KD-Tree for segment midpoints."""
    midpoints = np.array([s['midpoint'] for s in segments])
    radii = np.array([s['sphere_radius'] for s in segments])
    return KDTree(midpoints), radii

def find_nearby_segments(query_point, kdtree, radii, segments, max_distance=50.0):
    """Find segments whose bounding spheres are near the query point."""
    dists, indices = kdtree.query(query_point, k=100, distance_upper_bound=max_distance)
    nearby = []
    for idx in indices:
        if idx >= len(segments):
            continue
        s = segments[idx]
        if np.linalg.norm(query_point - s['midpoint']) <= s['sphere_radius']:
            nearby.append(s)
    return nearby

def point_within_frustum_optimized(point, segment):
    """Vectorized frustum check for a precomputed segment."""
    start = segment['start']
    end = segment['end']
    start_diam = segment['start_diam']
    end_diam = segment['end_diam']
    
    frustum_vector = end - start
    frustum_length = np.linalg.norm(frustum_vector)
    if frustum_length == 0:
        return False
    
    frustum_dir = frustum_vector / frustum_length
    point_vector = point - start
    proj_length = np.dot(point_vector, frustum_dir)
    
    if proj_length < 0 or proj_length > frustum_length:
        return False
    
    relative_pos = proj_length / frustum_length
    interpolated_diam = start_diam + (end_diam - start_diam) * relative_pos
    closest_point = start + frustum_dir * proj_length
    distance = np.linalg.norm(point - closest_point)
    
    return distance <= (interpolated_diam / 2)

def check_unconnected_end_optimized(vessel, segments, kdtree, radii, links):
    """Check ends using spatial indexing."""
    if vessel.tags:
        return {'start': 'outside', 'end': 'outside'},{'start': [], 'end': []}
    path = vessel.path
    ends_to_check = [0, -1]
    result = {'start': 'outside', 'end': 'outside'}
    indices_to_delete = {'start': [], 'end': []}

    # Precompute connected indices
    connected_indices = set()
    for link in links:
        if link.source_vessel == vessel:
            connected_indices.add(link.source_index)
        if link.target_vessel == vessel:
            connected_indices.add(link.target_index)
    connected_indices = sorted(connected_indices)

    for i, end_label in zip([0, -1], ['start', 'end']):
        end_point = np.array(path[i])
        
        # Skip connected ends
        is_connected = (i == 0 and 0 in connected_indices) or (i == -1 and (len(path)-1) in connected_indices)
        if is_connected:
            continue
        
        # Find nearby segments using spatial index
        nearby_segments = find_nearby_segments(end_point, kdtree, radii, segments)
        for seg in nearby_segments:
            if seg['vessel'] == vessel:
                continue
            if point_within_frustum_optimized(end_point, seg):
                # Determine overhang region
                if connected_indices:
                    if end_label == 'start':
                        min_idx = min(connected_indices)
                        indices_to_delete[end_label] = (0, min_idx)
                    else:
                        max_idx = max(connected_indices)
                        indices_to_delete[end_label] = (max_idx, len(path)-1)
                else:
                    indices_to_delete[end_label] = (0, len(path)-1)
                result[end_label] = 'inside'
                break

    return result, indices_to_delete
#%%
def optimize_linklength_ends(vessels):
    """Optimize connection points while maintaining vessel link references."""
    for vessel in vessels:
        if not hasattr(vessel, 'links_to_path') or not vessel.links_to_path:
            continue

        # Get extreme links from original list
        vessel.links_to_path.sort(key=lambda x: x.source_index)
        lower_link = vessel.links_to_path[0]
        upper_link = vessel.links_to_path[-1]

        # Process lower end
        if lower_link.source_index != 0:
            current_length = np.linalg.norm(
                vessel.path[lower_link.source_index] - 
                lower_link.target_vessel.path[lower_link.target_index]
            )
            
            # Find optimal index moving towards start
            best_idx = lower_link.source_index
            for candidate_idx in range(lower_link.source_index-1, -1, -1):
                candidate_point = vessel.path[candidate_idx]
                new_target_idx = closest_pathindex(
                    lower_link.target_vessel.path, 
                    candidate_point
                )
                new_length = np.linalg.norm(
                    candidate_point - 
                    lower_link.target_vessel.path[new_target_idx]
                )
                
                if new_length < current_length:
                    best_idx = candidate_idx
                    current_length = new_length
                else:
                    break

            # Update in original links list
            if best_idx != lower_link.source_index:
                # Store original indices
                original_source_idx = lower_link.source_index
                original_target_idx = lower_link.target_index
                
                # Update main link
                lower_link.source_index = best_idx
                lower_link.target_index = new_target_idx

                # Update reciprocal links in source vessel's links_to_path
                for i in range(len(vessel.links_to_path)):
                    t_link = vessel.links_to_path[i]
                    if (t_link.target_vessel == lower_link.target_vessel and 
                        t_link.target_index == original_target_idx):
                        vessel.links_to_path[i].source_index = best_idx
                        vessel.links_to_path[i].target_index = new_target_idx
                        break
                
                # Update reciprocal links in target vessel's links_to_path
                for i in range(len(lower_link.target_vessel.links_to_path)):
                    t_link = lower_link.target_vessel.links_to_path[i]
                    if (t_link.target_vessel == vessel and 
                        t_link.target_index == original_source_idx):
                        lower_link.target_vessel.links_to_path[i].source_index = new_target_idx
                        lower_link.target_vessel.links_to_path[i].target_index = best_idx
                        
                        break

        # Process upper end 
        if upper_link.source_index != len(vessel.path)-1:
            current_length = np.linalg.norm(
                vessel.path[upper_link.source_index] - 
                upper_link.target_vessel.path[upper_link.target_index]
            )
            
            # Find optimal index moving towards end
            best_idx = upper_link.source_index
            for candidate_idx in range(upper_link.source_index+1, len(vessel.path)):
                candidate_point = vessel.path[candidate_idx]
                new_target_idx = closest_pathindex(
                    upper_link.target_vessel.path, 
                    candidate_point
                )
                new_length = np.linalg.norm(
                    candidate_point - 
                    upper_link.target_vessel.path[new_target_idx]
                )
                
                if new_length < current_length:
                    best_idx = candidate_idx
                    current_length = new_length
                else:
                    break

            # Process upper end update
            if best_idx != upper_link.source_index:
                # Store original indices
                original_source_idx = upper_link.source_index
                original_target_idx = upper_link.target_index
                
                # Update main link
                upper_link.source_index = best_idx
                upper_link.target_index = new_target_idx

                # Update reciprocal links in source vessel's links_to_path
                for i in range(len(vessel.links_to_path)):
                    t_link = vessel.links_to_path[i]
                    if (t_link.target_vessel == upper_link.target_vessel and 
                        t_link.target_index == original_target_idx):
                        vessel.links_to_path[i].source_index = best_idx
                        vessel.links_to_path[i].target_index = new_target_idx
                        break
                
                # Update reciprocal links in target vessel's links_to_path
                for i in range(len(upper_link.target_vessel.links_to_path)):
                    t_link = upper_link.target_vessel.links_to_path[i]
                    if (t_link.target_vessel == vessel and 
                        t_link.target_index == original_source_idx):
                        upper_link.target_vessel.links_to_path[i].source_index = new_target_idx
                        upper_link.target_vessel.links_to_path[i].target_index = best_idx
                        break

def closest_pathindex(path, point):
    """Find index of closest path point using spatial indexing."""
    return np.argmin([np.linalg.norm(np.array(p)-np.array(point)) for p in path])
#vis.show_linked(vessels[::], None)

optimize_linklength_ends(vessels)
# %%
#vis.show_linked(vessels[::], None)
#%%
def closest_pathindex(path, point):
    """Find index of closest path segment endpoint to point."""
    distances = [np.linalg.norm(np.array(p) - np.array(point)) for p in path]
    return np.argmin(distances)

#%%
# Precompute data once
segments = precompute_segment_data(vessels)
kdtree, radii = build_spatial_index(segments)

# Main loop
vesselends_to_delete = []
for vessel in vessels:
    result, indices = check_unconnected_end_optimized(vessel, segments, kdtree, radii, links)
    print(f"Vessel {vessel.associated_vesselname}: start={result['start']}, end={result['end']}")
    if indices['start']:
        vesselends_to_delete.append((vessel, indices['start']))
    if indices['end']:
        vesselends_to_delete.append((vessel, indices['end']))
vesselends_to_delete

# %%
to_show=[(v.path[ind[0]:ind[1]],[200,100,100,200]) for v,ind in vesselends_to_delete]+[(v.path[:],[100,100,100,100]) for v in vessels]
#vis.show(to_show)

# %% [markdown]
# remove overhang

# %%
#remove the part thats overstanding inside
for vessel, indices in vesselends_to_delete:
    index_end_excluded=len(vessel.path) if indices[0]==0 else indices[0]+1 
    index_start=indices[1] if indices[0]==0 else 0
    vessel.reduce(index_start,index_end_excluded)
# %%
#vis.show_linked(vessels[::], None)
#%%
def detect_short_overhangs(vessel, segments, kdtree, radii, links, threshold=0.03):
    """Detect vessel ends that:
    1. Are behind a link connection
    2. Extend beyond the link for <5% of total path length
    3. End outside other vessels
    Returns deletion ranges in format {'start': (from, to), 'end': (from, to)}
    """
    if "sys_pul_art" in vessel.associated_vesselname:
        threshold=0.15
    if "sigmoid_sinus_right_1" in vessel.associated_vesselname:
        threshold=0.1
        
    marked = {'start': None, 'end': None}
    path = vessel.path
    path_len = len(path)
    
    if path_len < 20:  # Minimum points for meaningful percentage calculation
        return marked
    
    # Get all connection points from links
    connected_indices = set()
    for link in links:
        if link.source_vessel == vessel:
            connected_indices.add(link.source_index)
        if link.target_vessel == vessel:
            connected_indices.add(link.target_index)
    
    if not connected_indices:
        return marked
    
    # Check start end
    min_connected = min(connected_indices)
    if min_connected > 0:
        start_segment_length = min_connected
        if start_segment_length/path_len <= threshold:
            start_point = np.array(path[0])
            if not is_point_inside_any_vessel(start_point, vessel, segments, kdtree, radii):
                marked['start'] = (0, min_connected)
    
    # Check end
    max_connected = max(connected_indices)
    if max_connected < path_len-1:
        end_segment_length = path_len-1 - max_connected
        if end_segment_length/path_len <= threshold:
            end_point = np.array(path[-1])
            if not is_point_inside_any_vessel(end_point, vessel, segments, kdtree, radii):
                marked['end'] = (max_connected, path_len-1)
    
    return marked

def is_point_inside_any_vessel(point, current_vessel, segments, kdtree, radii):
    """Check if point is inside any other vessel's structure"""
    nearby_segments = find_nearby_segments(point, kdtree, radii, segments)
    for seg in nearby_segments:
        if seg['vessel'] != current_vessel:
            if point_within_frustum_optimized(point, seg):
                return True
    return False

short_overhangs_to_delete = []
for vessel in vessels:
    # Get existing unconnected end deletions
    result, _ = check_unconnected_end_optimized(vessel, segments, kdtree, radii, links)
    
    # Get short overhang deletions
    overhangs = detect_short_overhangs(vessel, segments, kdtree, radii, links,threshold=0.04)
    
    # Combine results
    for end in ['start', 'end']:
        if overhangs[end]:
            short_overhangs_to_delete.append((vessel, overhangs[end]))
#%%
to_show=[(v.path[ind[0]:ind[1]+1],[200,100,100,200]) for v,ind in short_overhangs_to_delete]+[(v.path[:],[100,100,100,100]) for v in vessels]
#vis.show(to_show)

#%%
# Then process deletions
for vessel, indices in short_overhangs_to_delete:
    index_end_excluded=len(vessel.path) if indices[0]==0 else indices[0]+1 
    index_start=indices[1] if indices[0]==0 else 0
    vessel.reduce(index_start,index_end_excluded)  
    
# %%
to_show=[(v.path[:],[100,100,100,100]) for v in vessels]
#vis.show_linked(vessels[::], None)




# %% [markdown]
# split vessels into uninterrupted parts, using intra vessel links

# %% [markdown]
# remove overlaps (there are no more)

# #vis.show(points)

# %%
hash(vessels[0])

# %%
#vis.show_linked(vessels, None)


# %%
def vessel_split_by_links(vessel, links):
    # Collect indices from links relevant to this vessel (ignoring index 0 for splitting)
    split_indices = sorted({link.target_index for link in links if link.target_vessel == vessel and link.target_index != 0})

    new_paths, new_dias = [], []
    last_index = 0

    # Add segment splits based on identified split_indices
    for split_index in split_indices:
        if last_index < split_index:
            new_paths.append(vessel.path[last_index:split_index])
            new_dias.append(vessel.diameters[last_index:split_index])
            last_index = split_index

    # Add remaining part as last segment
    if last_index < len(vessel.path):
        new_paths.append(vessel.path[last_index:])
        new_dias.append(vessel.diameters[last_index:])

    new_vessels = []
    # Create new Vessel objects from the segments
    for i, dia in enumerate(new_dias):
        other_a, other_b = create_dia_a_b(dia)
        nv = Vessel(
            new_paths[i],
            other_a,
            other_b,
            np.mean(dia),
            dia,
            vessel.associated_vesselname,
            vessel.speed_function,
        )
        nv.orig = hash(vessel)
        nv.iteration_number = i
        new_vessels.append(nv)

    new_links = []
    
    # Connect consecutive new vessel segments at endpoints
    for j in range(len(new_vessels) - 1):
        source_vessel = new_vessels[j]
        target_vessel = new_vessels[j+1]
        new_links.append(Link(source_vessel, len(source_vessel.path) - 1, target_vessel, 0))
        new_links.append(Link(target_vessel, 0, source_vessel, len(source_vessel.path) - 1))

    return new_vessels, new_links

""" def create_split_links(singled_vessels_split, original_vessels, original_links):
    vessel_map = {hash(original): [] for original in original_vessels}
    for split_vessel in singled_vessels_split:
        for original in original_vessels:
            if split_vessel.orig == hash(original):
                vessel_map[hash(original)].append(split_vessel)
                break

    new_inter_segment_links = []

    for link in original_links:
        source_segments = vessel_map[hash(link.source_vessel)]
        target_segments = vessel_map[hash(link.target_vessel)]

        # We assume here that the connecting logic follows original indices appropriately
        source_cumulative_length = 0
        target_cumulative_length = 0

        for source_segment in source_segments:
            segment_length = len(source_segment.path)
            if source_cumulative_length <= link.source_index < source_cumulative_length + segment_length:
                new_source_index = link.source_index - source_cumulative_length
                break
            source_cumulative_length += segment_length

        for target_segment in target_segments:
            segment_length = len(target_segment.path)
            if target_cumulative_length <= link.target_index < target_cumulative_length + segment_length:
                new_target_index = link.target_index - target_cumulative_length
                break
            target_cumulative_length += segment_length

        # Double-check bounds before adding
        if new_source_index < len(source_segments[source_segments.index(source_segment)].path) and \
           new_target_index < len(target_segments[target_segments.index(target_segment)].path):
            new_inter_segment_links.append(Link(source_segment, new_source_index, target_segment, new_target_index))
            new_inter_segment_links.append(Link(target_segment, new_target_index, source_segment, new_source_index))

    return new_inter_segment_links  """
""" def vessel_split_by_links(vessel, links):
    # Identify split indices, excluding index 0 to avoid initial split
    split_indices = sorted({link.target_index for link in links if link.target_vessel == vessel and link.target_index != 0})
    
    new_paths, new_dias = [], []
    last_index = 0

    for split_index in split_indices:
        if last_index < split_index and split_index < len(vessel.path):
            new_paths.append(vessel.path[last_index:split_index])
            new_dias.append(vessel.diameters[last_index:split_index])
            last_index = split_index

    # Append the final remaining segment
    if last_index < len(vessel.path):
        new_paths.append(vessel.path[last_index:])
        new_dias.append(vessel.diameters[last_index:])

    new_vessels = []
    for i, dia in enumerate(new_dias):
        if len(new_paths[i]) > 1:  # Expects segments with more than one point
            other_a, other_b = create_dia_a_b(dia)
            nv = Vessel(
                new_paths[i],
                other_a,
                other_b,
                np.mean(dia),
                dia,
                vessel.associated_vesselname,
                vessel.speed_function,
            )
            nv.orig = hash(vessel)
            nv.iteration_number = i
            new_vessels.append(nv)

    new_links = []
    # Link consecutive segments in the vessel
    for j in range(len(new_vessels) - 1):
        source = new_vessels[j]
        target = new_vessels[j + 1]
        new_links.append(Link(source, len(source.path) - 1, target, 0))
        new_links.append(Link(target, 0, source, len(source.path) - 1))

    return new_vessels, new_links """
def create_split_links(singled_vessels_split, original_vessels, original_links):
    # Map original vessels to the new split segments
    vessel_map = {hash(original): [] for original in original_vessels}
    for split_vessel in singled_vessels_split:
        vessel_map[split_vessel.orig].append(split_vessel)

    new_inter_segment_links = []

    for link in original_links:
        source_segments = vessel_map.get(hash(link.source_vessel), [])
        target_segments = vessel_map.get(hash(link.target_vessel), [])

        new_source_index = None
        new_target_index = None
        source_segment = None
        target_segment = None

        # Translate original link's source index to the new segment
        source_cumulative_length = 0
        for segment in source_segments:
            segment_length = len(segment.path)
            if source_cumulative_length <= link.source_index < source_cumulative_length + segment_length:
                source_segment = segment
                new_source_index = link.source_index - source_cumulative_length
                break
            source_cumulative_length += segment_length

        # Translate original link's target index to the new segment
        target_cumulative_length = 0
        for segment in target_segments:
            segment_length = len(segment.path)
            if target_cumulative_length <= link.target_index < target_cumulative_length + segment_length:
                target_segment = segment
                new_target_index = link.target_index - target_cumulative_length
                break
            target_cumulative_length += segment_length

        # Ensure valid segments and indices before creating the link
        if source_segment and target_segment and new_source_index is not None and new_target_index is not None:
            new_inter_segment_links.append(Link(source_segment, new_source_index, target_segment, new_target_index))
            new_inter_segment_links.append(Link(target_segment, new_target_index, source_segment, new_source_index))

    return new_inter_segment_links
links=[]
for v in vessels:
    for link in v.links_to_path:
        links.append(link)
# Sample processing logic with vessels and existing links
splits = [vessel_split_by_links(vessel, links) for vessel in vessels]

# Gather split vessels and new links
singled_vessels_split, new_links_from_split = [], []
for new_vessels, new_links in splits:
    singled_vessels_split.extend(new_vessels)
    new_links_from_split.extend(new_links)

# Add the newly created links based on split logic
additional_new_links = create_split_links(singled_vessels_split, vessels, links)
new_links_from_split.extend(additional_new_links)

# Assign links back to the vessel paths
for vessel in singled_vessels_split:
    vessel.links_to_path = []

# Apply links to the vessels
for link in new_links_from_split:
    link.source_vessel.add_link(link)

#vis.show_linked(singled_vessels_split, None)

#export to load in the graph script
with open(f"./vessels_split_reduced_pre_graph.pickle", "wb") as handle:
        pickle.dump(singled_vessels_split, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%
