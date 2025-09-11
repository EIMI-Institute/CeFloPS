# %%
import os
import sys

module_path = os.path.abspath(os.path.join("./../.."))
if module_path not in sys.path:
    sys.path.append(module_path)
import CeFloPS.simulation.common.shared_geometry as sg
import CeFloPS.simulation.common.functions as cf
import numpy as np

import trimesh
import pickle
import CeFloPS.simulation.settings as settings
import trimesh
import pickle
import CeFloPS.simulation.simsetup as simsetup
from CeFloPS.data_processing.vesselvoxel import *

# %%
rate_constants = settings.rate_constants
region_remapping = settings.region_remapping
print(settings.ROI_VOXEL_PITCH)

# %%
from CeFloPS.simulation.simsetup import *
from scipy.ndimage import distance_transform_edt
blood_only = False
names_only = False
vessel_not_allowed_in_path = True
exclude_strings = []
# start with all stls files in spec folder
all_files = [x for x in glob.glob(settings.PATH_TO_STLS + "/*.stl")]
if blood_only:
    all_files = []
print("remaining:", len(all_files))
vessels = []
rois = dict()
to_remove = []
for name in all_files:
    if (
        settings.VESSEL_VEINLIKE
        in name  # [len(settings.PATH_TO_STLS) : len(settings.PATH_TO_STLS) + 5]
        or settings.VESSEL_ARTERYLIKE
        in name  # [len(settings.PATH_TO_STLS) : len(settings.PATH_TO_STLS) + 5]
    ):
        vessels.append(name)
        to_remove.append(name)
[all_files.remove(name) for name in to_remove]
# filter out vessellike names
print("removed", vessels)  # exclude vesselfiles
print("remaining after excluding vessels:", len(all_files))
extravessels = []
to_remove = []
# filter out vessellike names that are not marked as vessellike as well as negatively defined objects
for name in all_files:
    print(name)
    if (
        "veins" in name
        or "arte" in name
        or "Vein" in name
        or "Arte" in name
        or ("vessel" in name and vessel_not_allowed_in_path)
        or any([n in name for n in settings.NEGATIVE_SPACES])
        or any([n in name for n in exclude_strings])
        #or "organ" in name or not "leg" in name
    ):
        extravessels.append(name)
        to_remove.append(name)

[all_files.remove(name) for name in to_remove]
print(
    "remaining after excluding negspaces and vesselidents unspecified:",
    len(all_files),
)

vessels += extravessels

# sort rois
# define blood roi and subtract negative stls as well as vesselstls
blood_roi = Blood_roi("blood", settings.PATH_TO_VESSELS, 1)
voi_addition = blood_roi.additional_vois
for x in voi_addition:
    x.reload(blood_roi)
blood_roi.additional_vois = None
try:
    subs = settings.NEGATIVE_SPACES
    prefix = settings.PATH_TO_STLS
    subs_meshes = [trimesh.load(prefix + "/" + name) for name in subs]
    print(subs_meshes)
    if settings.SUBTRACT_VESSELMESHES:
        subs_meshes += [trimesh.load(path) for path in settings.VESSELPATHS]
except Exception as e:
    print(f"Couldn't load negative Spaces, Error: {e}")

    from scipy.ndimage import distance_transform_edt

# Modified negative grid processing
negative_data = []
""" for sub in subs_meshes:
    try:
        safe_grid = create_safe_negative_grid(sub, settings.ROI_VOXEL_PITCH)
        negative_data.append(safe_grid)
    except Exception as e:
        print(f"Error processing {sub}: {e}") """


print("negative spaces", negative_data)

# load meshes and map them to a region, load the rate constants and create rois, also filter that only specified geometries within all_files are loaded

identifiers = [
    item
    for sublist in region_remapping.values()
    for item in (sublist if isinstance(sublist, list) else [sublist])
]
distilled_all_names = [
    name
    for name in all_files
    if any([identifier in name for identifier in identifiers]) #and not "musc" in name#TODO rem again
]
#%%
import CeFloPS.visualization as vis
def full_model_connection_visualization(vessels,vois,mark_vesseltypes=True):
    inner_voi_paths=[]
    voi_to_vessels_paths=[]
    vessel_vessel_path=[]
    colors = {
    "vessel_general": [128, 128, 128, 255],  # Gray
    "vein": [0, 0, 255, 255],               # Blue
    "artery": [255, 0, 0, 255],             # Red
    "tissue": [255, 182, 193, 255],         # Light Pink
    "capillary_vein": [100, 149, 237, 255], # Cornflower Blue
    "capillary_artery": [255, 99, 71, 255], # Tomato
    "venol": [70, 130, 180, 255],           # Steel Blue
    "arteriol": [255, 165, 0, 255]          # Orange
    }
    def create_path(positions,color):
        assert len(positions)>1
        print(positions)
        showable_path = trimesh.load_path(
                                positions
                            )
        if len(showable_path.entities) == 0:
            #print("No entities were created. Please check the positions input.",positions)
            positions[1]=[positions[1][0],positions[1][1],positions[1][2]+0.1]

            showable_path = trimesh.load_path(
                                positions
                            )
        #print(len(positions),len(showable_path.entities))
        showable_path.colors = [color for _ in range(len(showable_path.entities))]
        return showable_path
    total_inner_paths,total_outer_paths=[],[]
    for voi in vois:
        veinpoints_outside=[[vv[0].path[vv[1]],voi.geometry.get_points()[vv[2]]] for vv in voi.outlets]
        veinpoints_outside=[[vv[0].path[vv[1]],voi.geometry.get_points()[vv[2]]] for vv in voi.inlets]
        #generate paths from all entries to outlets
        color=colors["venol"]
        if not mark_vesseltypes:
            color=colors["vessel_general"]
        outer_paths=[create_path(v,color) for v in veinpoints_outside  ]

        color=colors["arteriol"]
        if not mark_vesseltypes:
            color=colors["vessel_general"]
        outer_paths+=[create_path(v,color) for v in veinpoints_outside  ]

        color=colors["capillary_artery"]
        #print("TAKE")
        #paths inside of voi
        veinpoints_artpoints=[[voi.geometry.get_points()[vv[2]], voi.geometry.get_points()[vvv[2]]] for vv in voi.inlets for vvv in voi.outlets]
        inner_paths=[create_path(v,color) for v in veinpoints_artpoints if not all([k == kk for k in v[0]  for kk in v[1]])]
        total_inner_paths+= inner_paths;total_outer_paths+=outer_paths
    to_show=[(v.path,colors["vein"] if v.type=="vein" else colors["artery"]) for v in vessels]
    to_show+=total_outer_paths+total_inner_paths
    vis.show(to_show)

# %%
vessels=blood_roi.geometry


vessels_reconnected = reconnect(vessels)

# %%

extracted_layer_skeletons=[generate_layered_skeleton(v.path,v.diameters,settings.ROI_VOXEL_PITCH) for v in vessels_reconnected]

# %%
all_points = np.vstack([v.path for v in vessels_reconnected])
global_bounds = (np.min(all_points, axis=0), np.max(all_points, axis=0))

#%%

sparse_skel_set, skel_origin, skel_pitch, skel_shape = create_sparse_skeleton(
    extracted_layer_skeletons,
    pitch=settings.ROI_VOXEL_PITCH,
    global_bounds=global_bounds
)



#%%
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import lil_matrix


import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.spatial import KDTree
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
def init_vessel_node_tags(vessels):
    return (dict((vessel.id,["B" for i in range(len(vessel.path))]) for vessel in vessels),dict((vessel.id,[np.inf for i in range(len(vessel.path))]) for vessel in vessels))
def set_vessel_node_tag(vessel,index,nametag,meshvolume,vesseltags,tagvolumes):
    if tagvolumes[vessel.id][index]>meshvolume:
        vesseltags[vessel.id]=nametag
        tagvolumes[vessel.id][index]=meshvolume
def safe_thicken(vox, thickness=4):
    """3D thickening with proper broadcasting and component connection"""
    matrix = vox.matrix.copy()
    struct = generate_binary_structure(3, 3)  # 3D connectivity

    # Create initial shell
    core = binary_erosion(matrix, structure=struct)
    shell = matrix & ~core

    # Dilate outward while maintaining connectivity
    for _ in range(thickness - 1):
        shell = binary_dilation(shell, structure=struct)

    # Combine with original matrix using broadcasting
    thickened = np.logical_or(matrix, shell)

    return trimesh.voxel.VoxelGrid(thickened, transform=vox.transform)

def find_connected_components(points, pitch, origin):
    """3D connected components with proper type handling"""
    # Ensure proper array types and shapes
    points = np.asarray(points, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64).reshape(1, 3)

    # Convert to grid coordinates
    indices = np.round((points - origin) / pitch).astype(int)
    unique_indices, inverse = np.unique(indices, axis=0, return_inverse=True)

    # Find neighbors with KDTree
    tree = KDTree(unique_indices)
    pairs = tree.query_pairs(r=1.0, p=np.inf)  # 26-connectivity

    # Build adjacency matrix
    adj = lil_matrix((len(unique_indices), len(unique_indices)), dtype=bool)
    for i, j in pairs:
        adj[i, j] = adj[j, i] = True

    # Find components
    n_components, labels = connected_components(adj)
    return [points[labels[inverse] == label] for label in range(n_components)]
def find_voxel_components(points, pitch, origin):
    """Find 26-connected components"""
    indices = np.round((points - origin) / pitch).astype(int)
    unique_indices, inverse = np.unique(indices, axis=0, return_inverse=True)

    # Chebyshev distance for 26-neighbor connectivity
    tree = KDTree(unique_indices)
    pairs = tree.query_pairs(r=1.0, p=np.inf)

    # Build adjacency matrix
    adj = lil_matrix((len(unique_indices), len(unique_indices)), dtype=bool)
    for i, j in pairs:
        adj[i, j] = adj[j, i] = True

    # Find components
    _, labels = connected_components(adj)

    return [points[labels[inverse] == label] for label in np.unique(labels)]

import numpy as np
from scipy.spatial import KDTree

def map_vessel_points_to_voxelgrid(vessels, voxel_grid):
    """
    Accurately maps vessel points to voxel grid using spatial queries.

    Parameters:
        vessels (list): List of vessel objects with `path` and `id`.
        voxel_grid (trimesh.voxel.VoxelGrid): Target voxel grid.

    Returns:
        dict: Mapping of vessel IDs to contained point indices.
    """
    contained_points = {}

    # Get voxel centers and pitch from the grid
    voxel_centers = voxel_grid.points
    voxel_pitch = voxel_grid.transform[0, 0]

    # Create KDTree for efficient spatial queries
    tree = KDTree(voxel_centers)

    # Maximum distance to consider as inside (half voxel diagonal)
    max_distance = voxel_pitch * np.sqrt(3)/2 * 1.0001  # Small buffer for FP errors

    for vessel in vessels:
        # Query all points within voxel boundaries
        distances, indices = tree.query(
            vessel.path,
            distance_upper_bound=max_distance,
            workers=-1
        )

        # Find valid points within voxels
        valid = np.where(distances <= max_distance)[0].tolist()

        if valid:
            contained_points[vessel.id] = valid

    return contained_points
def map_vessel_points_to_voxelgrid(vessels, voxel_grid):
    """
    Accurately maps vessel points to voxel grid using dual verification:
    1. KDTree for proximity search
    2. Exact bounds check for containment
    """
    contained_points = {}

    # Get voxel grid properties
    voxel_centers = voxel_grid.points
    pitch = voxel_grid.transform[0, 0]
    half_pitch = pitch / 2.0
    max_distance = half_pitch * np.sqrt(3)  # Max center-to-corner distance

    # Create KDTree for efficient proximity search
    tree = KDTree(voxel_centers)

    for vessel in vessels:
        valid_indices = []
        path = vessel.path

        # Find all candidate voxels for each point
        candidates = tree.query_ball_point(
            path,
            r=max_distance * 1.0001,  # Small buffer for FP errors
            workers=-1,
            return_sorted=False
        )

        # Vectorized bounds checking
        for i, (point, cand_indices) in enumerate(zip(path, candidates)):
            if not cand_indices:
                continue

            # Get candidate voxel centers
            cand_centers = voxel_centers[cand_indices]

            # Calculate bounds for all candidates simultaneously
            mins = cand_centers - half_pitch
            maxs = cand_centers + half_pitch

            # Check containment in all dimensions
            in_x = (point[0] >= mins[:, 0]) & (point[0] < maxs[:, 0])
            in_y = (point[1] >= mins[:, 1]) & (point[1] < maxs[:, 1])
            in_z = (point[2] >= mins[:, 2]) & (point[2] < maxs[:, 2])

            # Combine results across candidates
            if np.any(in_x & in_y & in_z):
                valid_indices.append(i)

        if valid_indices:
            contained_points[vessel.id] = valid_indices

    return contained_points
#omitting outer layer:
import numpy as np
from scipy.spatial import KDTree

def map_vessel_points_to_voxelgrid(vessels, voxel_grid):
    """
    Maps vessel points to inner voxels only (excluding outer layer),
    using precise boundary checks.
    """
    contained_points = {}

    # Get all voxel centers and grid properties
    all_centers = voxel_grid.points
    pitch = voxel_grid.transform[0, 0]
    origin = voxel_grid.transform[:3, 3]
    half_pitch = pitch / 2.0

    # Calculate voxel indices for each center
    indices = ((all_centers - origin) / pitch).round().astype(int)

    # Find min/max indices to identify outer layers
    min_idx = indices.min(axis=0)
    max_idx = indices.max(axis=0)

    # Mask for inner voxels (not on outer faces)
    is_inner = np.all(
        (indices > min_idx) & (indices < max_idx),
        axis=1
    )
    inner_centers = all_centers[is_inner]

    if len(inner_centers) == 0:
        return contained_points  # No inner voxels to check

    # Create KDTree only for inner voxels
    tree = KDTree(inner_centers)
    max_distance = half_pitch * np.sqrt(3) * 1.0001  # Max center-to-corner

    for vessel in vessels:
        valid_indices = []
        path = vessel.path

        # Find candidates in inner voxels only
        dist, idx = tree.query(
            path,
            distance_upper_bound=max_distance,
            workers=-1
        )

        # Vectorized bounds check for inner voxels
        for i, (d, point) in enumerate(zip(dist, path)):
            if d > max_distance:
                continue

            # Get the candidate inner voxel's center
            center = inner_centers[idx[i]]

            # Precise bounds check (exclusive of outer edges)
            if np.all(
                (point >= center - half_pitch) &
                (point < center + half_pitch)
            ):
                valid_indices.append(i)

        if valid_indices:
            contained_points[vessel.id] = valid_indices

    return contained_points
def process_mesh_memopt(filepath, sparse_skel_set, skel_origin, skel_pitch, vessels):
    try:
        mesh = trimesh.load(filepath)

        if not isinstance(mesh, trimesh.Trimesh):
            return []

        results = []
        name = os.path.basename(filepath)
        is_roi = any(i in name for i in settings.ROI_2_5D)

        # Align voxelization with skeleton grid
        voxel_args = {
            'pitch': skel_pitch
        }

        if mesh.is_watertight:
            contained_points = dict()
            # Check where vessels lie inside the one mesh:
            for vessel in vessels:
                for i, val in enumerate(mesh.contains(vessel.path)):
                    if val:
                        if vessel.id not in contained_points:
                            contained_points[vessel.id] = []
                        contained_points[vessel.id].append(i)

            for idx, submesh in enumerate(mesh.split()):
                try:
                    # Voxelize using skeleton-aligned grid
                    vox = submesh.voxelized(**voxel_args).fill()

                    if is_roi:
                        print(f"Before thickening: {len(vox.points)}")
                        vox = vox.hollow()
                        vox = safe_thicken(vox)
                        print(f"After thickening: {len(vox.points)}")

                    # Find non-skeleton voxels
                    remaining = subtract_sparse_voxels(
                        vox,
                        sparse_skel_set,
                        skel_origin,
                        skel_pitch
                    )
                    print(f"Non-skeleton voxels: {len(remaining)}")

                    if len(remaining) > 0:
                        if len(remaining) > 10:
                            results.append((
                                f"{name}_{idx}_comp{0}",
                                remaining,
                                len(remaining),
                                np.mean(remaining, 0),
                                submesh,
                                contained_points
                            ))

                except Exception as e:
                    print(f"Submesh {idx} error: {str(e)}")

        else:
            idx = 0
            try:
                vox = mesh.voxelized(**voxel_args).fill()

                if is_roi:
                    print(f"Before thickening: {len(vox.points)}")
                    vox = vox.hollow()
                    vox = safe_thicken(vox)
                    print(f"After thickening: {len(vox.points)}")

                remaining = subtract_sparse_voxels(
                    vox,
                    sparse_skel_set,
                    skel_origin,
                    skel_pitch
                )
                print(f"Non-skeleton voxels: {len(remaining)}, type{type(remaining)}")

                if len(remaining) > 0:
                    contained_points = map_vessel_points_to_voxelgrid(vessels, vox)
                    components = find_voxel_components(
                        remaining,
                        skel_pitch,
                        skel_origin
                    )
                    for ci, comp in enumerate(components):
                        # Filter and store results
                        if len(comp) >= 10:
                            results.append((
                                f"{name}_{idx}_comp{ci}",
                                comp,
                                len(comp),
                                np.mean(comp, 0),
                                mesh,
                                contained_points
                            ))

            except Exception as e:
                print(f"Error processing {filepath}: {str(e)}")
                return []

        return results

    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return []

def process_all_meshes(filepaths, sparse_skel_set, skel_origin, skel_pitch, vessels):
    all_results = Parallel(n_jobs=-1, verbose=1)(
        delayed(process_mesh_memopt)(
            fp,
            sparse_skel_set,
            skel_origin,
            skel_pitch,
            vessels
        ) for fp in filepaths
    )
    # Flatten the list if you want a single list with all sublists combined
    flat_results = [result for mesh_results in all_results for result in mesh_results]
    return flat_results

#%%


import numpy as np
from scipy.spatial import KDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import lil_matrix

#%%
base_path = r"pathto\test\heart/"
final_results = process_all_meshes(
    filepaths=[n for n in distilled_all_names+[base_path + "beating_heart.systole.pericardium.stl"] if "intest" in n ],
    sparse_skel_set=sparse_skel_set,
    skel_origin=skel_origin,
    skel_pitch=skel_pitch,
    vessels=vessels
)
#%%
import pickle
filename = 'pathto/final_results_backup.pkl'
with open(filename, 'rb') as file:
        final_results = pickle.load(file)

#%%
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
import trimesh



# Group entries by original mesh and process
import numpy as np
from collections import defaultdict

def merge_contained_points(dict1, dict2):
    merged_dict = dict1.copy()  # Start with a copy of the first dictionary

    for key, indices in dict2.items():
        if key in merged_dict:
            # Add indices to existing list, ensuring uniqueness
            merged_dict[key] = list(set(merged_dict[key] + indices))
        else:
            # If the key is not present, simply add it
            merged_dict[key] = indices

    return merged_dict

def merge_components_by_intersection(entries):
    if len(entries) <= 1:
        return entries

    # Sort entries based on the size of their voxel sets
    entries.sort(key=lambda x: len(x[1]), reverse=True)

    merged_entries = []
    non_mergable_entries = []
    while entries:
        to_merge = [entries.pop(0)]  # Start with the largest remaining
        combined_voxels = set(map(tuple, to_merge[0][1]))
        merged_contained_points = to_merge[0][5]

        # Attempt to merge other entries
        i = 0
        while i < len(entries):
            current_voxels = set(map(tuple, entries[i][1]))
            if combined_voxels.intersection(current_voxels):
                # If there is an intersection, merge the voxel sets and contained points
                combined_voxels.update(current_voxels)
                merged_contained_points = merge_contained_points(merged_contained_points, entries[i][5])
                to_merge.append(entries.pop(i))  # Remove and process this entry
            else:
                i += 1

        if len(to_merge) > 1:
            # If more than one entry was mergeable, add the merged result
            combined_tracked_array = np.array(list(combined_voxels))
            new_mean_position = np.mean(combined_tracked_array, axis=0)
            base_name = '_'.join(to_merge[0][0].split('_')[:-2]) + '_merged'

            merged_entry = (
                base_name,
                combined_tracked_array,
                len(combined_tracked_array),
                new_mean_position,
                to_merge[0][4],  # Assuming the mesh is the same for the group
                merged_contained_points
            )
            merged_entries.append(merged_entry)
        else:
            # If no other entry was mergeable, simply keep this entry
            non_mergable_entries.append(to_merge[0])

    # Return combined results of merged and non-merged entries
    return merged_entries + non_mergable_entries

# Example usage:
original_groups = defaultdict(list)
for entry in final_results:
    base_name = '_'.join(entry[0].split('_')[:-2])  # Remove "_compX" suffix
    original_groups[base_name].append(entry)

combined_results = []
for original_name, entries in original_groups.items():
    if "intest" in original_name:
        merged = merge_components_by_intersection(entries)
    else:
        merged = entries  # Keep original entries unmerged if not applicable
    combined_results.extend(merged)

#%%
filename = './combined_results_intest.pkl'
with open(filename, 'rb') as file:
        comb_intestine = pickle.load(file)
#%%
final_results = [entry for entry in final_results if "intest" not in entry[0]]
final_results.extend(comb_intestine)
#%%
for i, entry in enumerate(final_results):
    base_name = entry[0].split('.stl')[0]
    entry_list = list(entry)
    entry_list[0] = base_name
    final_results[i] = tuple(entry_list)
#%%

[(f[0],len(f[1])) for f in combined_results]
to_show=[]
for i,entry in enumerate(combined_results):
    if "intest" in entry[0]:
        to_show.append((list(entry[1]),[(100+10*i)%250,100,100,100]))
vis.show(to_show)


#%%
def create_tag_results(final_results):
    tag_results = []

    for i,entry in enumerate(final_results):
            name,points,volume,meanpoint,mesh,vessdict=entry
            for vessel_id, indices in vessdict.items():
                vessel = next((v for v in vessels if vessel_id == v.id), None)
                if vessel:
                    # Validate indices in terms of vessel path
                    for idx in indices:
                        assert idx < len(vessel.path), f"Index {idx} out of range for vessel path"

                    # Append the tag results as a tuple of vessel_id, indices, and mesh_name
                    tag_results.append((vessel_id, indices, name,volume))

    print("Tag Results:")
    for tag in tag_results:
        print(tag)
    return tag_results

vessel_tags=create_tag_results(final_results)
#%%
def create_vesselnodetags(data):
    data.sort(key=lambda x: x[-1])
    # Final dictionary to store the results
    result = {}
    for vessel_id,indices, mesh_name, volume in data:
            # Fetch the relevant vessel
            vessel = [v for v in vessels if v.id == vessel_id][0]
            if vessel_id not in result:
                result[vessel_id] = {}

            # Keep track of taken indices
            claimed_indices = set()

            # If this vessel already has assignments, collect the claimed indices
            for m_name, m_indices in result[vessel_id].items():
                claimed_indices.update(m_indices)

            # Filter out indices that are already claimed
            new_indices = [index for index in indices if index not in claimed_indices]

            # If there are new indices to add, store them
            if new_indices:
                result[vessel_id][mesh_name] = new_indices
    return result
#%%
#create muscle and nonmuscle version
muscle_vessdata=create_vesselnodetags(vessel_tags)
muscleless_vessdata=create_vesselnodetags([entry for entry in vessel_tags if ("musc" not in entry[-2] or "pericard" in entry[-2])])
#%%
set([tuple(muscle_vessdata[entry].keys()) for entry in muscle_vessdata])
set([tuple(muscleless_vessdata[entry].keys()) for entry in muscleless_vessdata])
#%%
# Print the resulting mapping
import pickle
filename = settings.vessel_dir+'/vessel_nodetag_data_musc.pkl'
with open(filename, 'wb') as file:
    pickle.dump(muscle_vessdata, file)
filename = settings.vessel_dir+'/vessel_nodetag_data_no_musc.pkl'
with open(filename, 'wb') as file:
    pickle.dump(muscleless_vessdata, file)
#%%
def load_vesseltags(vessels,final_rois):
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
            for mesh_name, indices in loaded_data[vessel.id].items():  # <- Ensure vessel.id is used here
                if mesh_name in loaded_roinames or True:
                    for index in indices:
                        # Debug: print to help verify what is being processed
                        print(f"Processing Vessel ID: {vessel.id}, Mesh: {mesh_name}, Index: {index}")

                        # Only try to replace if 'index' is within bounds
                        if index < len(vessel.node_tags) and vessel.node_tags[index] == "B":
                            vessel.node_tags[index] = mesh_name
    for vessel in vessels:
        for i in range(len(vessel.path)):
            if vessel.node_tags[i]=="B" and vessel.type=="vein":
                vessel.node_tags[i]="VB"
load_vesseltags(vessels,[])
#%%
for v in vessels:
    if "aorta" or "arm" in v.associated_vesselname:
        print(v.associated_vesselname,v.node_tags)
#%%
import matplotlib.pyplot as plt
import random

mesh_names = {tag for vessel in vessels for tag in vessel.node_tags}
mesh_colors = {mesh: plt.cm.viridis(random.random()) for mesh in mesh_names}

path_visualization = []

for vessel in vessels:
    for idx, node_tag in enumerate(vessel.node_tags):
        # Use node_tag to get color
        color = mesh_colors.get(node_tag, (0, 0, 0, 1))  # Default to black if not found

        # Check if index is within bounds
        if idx < len(vessel.path):
            point = vessel.path[idx]
            path_visualization.append(([point], color))

vis.show(path_visualization)
#%%
import matplotlib.pyplot as plt
all_mesh_names = set()
for vessel_id, meshes in loaded_data.items():
    all_mesh_names.update(meshes.keys())
mesh_names=list(all_mesh_names)
mesh_colors = {mesh: plt.cm.viridis(random.random()) for mesh in mesh_names}
path_visualization = []

for vessel in vessels:
    if vessel.id in loaded_data:
        for mesh_name, indices in loaded_data[vessel.id].items():
            color = mesh_colors.get(mesh_name, (0, 0, 0, 1))  # Default black if not found
            points = [vessel.path[idx] for idx in indices if idx < len(vessel.path)]
            path_visualization.append((points, color))
vis.show(path_visualization)

#%%
#save raw filtered and load for all loadedin finrois.
#group the final_results by name and try to combine the ones with the same name! add their volumes, recalculate the center and the points as well as update the tuple with the new information!

# %%
identifiers_meshes=[]
for entry in final_results:

        name, points,volume3, center,mesh,_=entry#entry["name"], entry["points"], entry["volume"], entry["center"], entry["mesh"]
        identifiers_meshes.append( (settings.PATH_TO_STLS+"/"+name, points, volume3, center, mesh))

# %%
#identifiers_meshes

# %%
for k in results:
    if type(k)==list:
        print(k[0])

# %%
for iss in identifiers_meshes:
    assert len(iss[1]) > 0, iss

# %%

# %%
settings.f_settings = settings.settings_from_files

# %%
rate_constants, region_remapping

# %%
def create_region_mappings(
    identifiers_meshes, blood_roi, rate_constants, region_remapping
):
    # Preserve original case but use lowercase for matching
    region_patterns = {}
    for orig_region, patterns in region_remapping.items():
        key = orig_region.lower()
        region_patterns[key] = {
            "original": orig_region,  # Keep original casing
            "patterns": [patterns] if isinstance(patterns, str) else patterns,
            "rate_constants": rate_constants.get(orig_region, []),
        }
        # Special handling for muscle
        if key == "muscle":
            region_patterns[key]["patterns"].extend(["musc", "muscle"])

    # Initialize with original region names
    rois = {k: [] for k in region_remapping.keys()}
    rois["All"] = []

    for mesh_name, geometry, x, y, mesh in identifiers_meshes:
        if len(geometry) == 0:
            print("TOTALLY NEGATED", mesh_name)
            continue

        ident, roi = create_roi(mesh_name, (geometry, x, y), blood_roi,store_loc=True)
        roi.blood_roi = blood_roi
        roi.recreate_compartments()
        roi.temp_mesh = mesh

        clean_name = mesh_name.lower()
        matched = False

        for norm_region, data in region_patterns.items():
            for pattern in data["patterns"]:
                if str(pattern).lower() in clean_name:
                    # Use original casing from region_remapping
                    rois[data["original"]].append(roi)
                    roi.rate_constants = data["rate_constants"]
                    matched = True
                    break
            if matched:
                break

        if not matched:
            rois["All"].append(roi)
            roi.rate_constants = rate_constants.get("All", [])

    final_rois = [
        roi
        for region in region_remapping
        if region != "Lll"
        for roi in rois.get(region, [])
    ]

    return rois, final_rois, region_patterns


rois, final_rois, roi_mapping_str = create_region_mappings(
    [x for x in identifiers_meshes if "musc" not in x[0] and "pericard" not in x[0]], blood_roi, rate_constants, region_remapping
)

#%%
#rename duplicate names
seen_names=set()
idx=10
for roi in final_rois:
    if roi.name in seen_names:
        roi.name=roi.name+"_"+str(idx)
        idx+=1
    seen_names.add(roi.name)

#%%
for vv in final_rois:
    for vvv in final_rois:
        if vv!=vvv:
            assert vv.name!=vvv.name, (vv.name,vvv.name,vv.volume,vvv.volume)


# %%
if False:
    to_show=[(v.path,[200,100,100,100]) for v in vessels]
    for voi in final_rois[:-1]:
        source_indices=[vv[2] for vv in voi.inlets]
        target_indices=[vv[2] for vv in voi.outlets]
        if any([i in target_indices for i in source_indices]):
            print(voi.name)
            to_show.append(([voi.geometry.get_points()[i] for i in source_indices],[200,100,100,200]))
            to_show.append(([voi.geometry.get_points()[i] for i in target_indices],[100,100,200,200]))
        to_show.append((voi.geometry.get_points(),[100,100,100,20]))
    vis.show(to_show)

# %%
len(final_rois)

# %%
for roi in final_rois:
    blood_roi.register_connected_roi_comp(roi)
final_rois.append(blood_roi)
# removes=[]
problems = []
for vessel in blood_roi.geometry:
    vessel.register_volumes()
for i, vessel in enumerate(blood_roi.geometry):
    try:
        traverse_vessel(vessel, 0, [], [])
    except Exception as e:
        problems.append(vessel)
        print("Error:", e, i, vessel.associated_vesselname[-20::])
""" for v in problems:
    # removes.append(v)
    # print("removed not connected from: ", v.associated_vesselname)
    blood_roi.geometry.remove(v) """
# remove profiles as they cant be pickled
for v in blood_roi.geometry:
    v.unregister_functions()  # as llambda functions cant be pickled
vessels=blood_roi.geometry
print("blood_roi_vessels", len(blood_roi.geometry))

# %%
for v in vessels:
    if "rca_vess2" in v.associated_vesselname:
        print(v.links_to_vois)

# %%
def create_vessel_mappings(vessel_list):
            points = []
            mappings = []
            for vessel in vessel_list:
                for i, pt in enumerate(vessel.path):
                    points.append(pt)
                    mappings.append((vessel, i))
            return np.array(points), mappings

# %%
def get_bbox_from_voxels(voxel_points, voxel_pitch):
    """Calculate bounding box from voxel centers"""
    # Convert to numpy array if not already
    min_coords = [float("inf")] * 3
    max_coords = [-float("inf")] * 3

    # Single pass through voxel points
    for point in voxel_points:
        for i in range(3):
            coord = point[i]
            if coord < min_coords[i]:
                min_coords[i] = coord
            if coord > max_coords[i]:
                max_coords[i] = coord

    # Convert to numpy arrays and expand by pitch
    min_coords = np.array(min_coords) - voxel_pitch / 2
    max_coords = np.array(max_coords) + voxel_pitch / 2

    return min_coords, max_coords


def ends_in_mesh(vessel, voi):
    """Improved version using calculated bbox and ray casting"""
    point = vessel.path[0] if vessel.type == "vein" else vessel.path[-1]
    is_inside = voi.temp_mesh.contains([point])
    return is_inside[0]
def not_in_mesh_indices(vessel, voi):
    """Improved version using calculated bbox and ray casting"""
    points = vessel.path
    is_inside = voi.temp_mesh.contains(points)
    return [i for i in range(len(is_inside)) if not is_inside[i]]




def is_inside_voxelgrid(point, voxel_set, voxel_pitch, bbox_min, bbox_max):
    """
    Check if a point is inside a voxel grid with potential holes using ray-casting.

    Args:
        point: 3D coordinate (x, y, z)
        voxel_set: Set of (x, y, z) tuples representing occupied voxels
        voxel_pitch: Voxel grid resolution
        bbox_min: Minimum bounding box corner (x, y, z)
        bbox_max: Maximum bounding box corner (x, y, z)

    Returns:
        bool: True if point is inside the solid voxel grid (excluding holes)
    """
    # Cast ray along positive X-axis
    ray_dir = np.array([1, 0, 0])
    current_voxel = tuple(np.floor((point - bbox_min) / voxel_pitch).astype(int))
    crossings = 0

    # March through voxels until exiting bounding box
    while True:
        # Check if current voxel is occupied
        if current_voxel in voxel_set:
            crossings += 1

        # Move to next voxel along ray
        current_voxel = (
            current_voxel[0] + ray_dir[0],
            current_voxel[1] + ray_dir[1],
            current_voxel[2] + ray_dir[2],
        )

        # Check if outside bounding box
        if not (
            bbox_min[0] <= current_voxel[0] * voxel_pitch < bbox_max[0]
            and bbox_min[1] <= current_voxel[1] * voxel_pitch < bbox_max[1]
            and bbox_min[2] <= current_voxel[2] * voxel_pitch < bbox_max[2]
        ):
            break

    return crossings % 2 == 1

def no_voi_links_to(v,vois):
    return all([all([v not in k for k in voi.inlets]) and all([v not in k for k in voi.outlets]) for voi in vois])
def connect_vessels_to_vois(final_rois, vessels):
    """connect_vessels_to_vois Connect all innerlaying vessels, rest of vessels to fullfill bidirectional rule and rest with threshold

    Args:
        final_rois (_type_): _description_
        vessels (_type_): _description_
    """
    roi_data_cache = dict(
        (roi.name, []) for roi in final_rois if "blood" not in roi.name
    )

    available_vessels = vessels.copy()
    connected_vessels = set()
    vesselends_in = dict()

    # Sort ROIs by size
    roi_arrays = [
        (roi, roi.geometry.get_points())
        for roi in final_rois
        if "blood" not in roi.name
    ]
    roi_arrays = sorted(roi_arrays, key=lambda l: len(l[1]))
    meshes = [set([tuple(e) for e in roi_array[1]]) for roi_array in roi_arrays]

    # Cache checking logic
    use_cached = True
    lookup_names = []
    for roi in final_rois:
        if "blood" not in roi.name:
            vessel_hash = consistent_hash(settings.PATH_TO_VESSELS)
            voxeldistance = settings.ROI_VOXEL_PITCH
            string_to_hash = (
                str(vessel_hash)
                + "_"
                + str(voxeldistance)
                + "_"
                + roi.name
                + "_"
                + str(len(final_rois))
                + "_"
                + str(settings.NEGATIVE_SPACES)
            )

            if settings.SUBTRACT_VESSELMESHES:
                string_to_hash = string_to_hash + "subtract_vessels"
            roi_ident = consistent_hash(string_to_hash)

            lookupname = "connections" + str(roi_ident) + ".cached_connections"
            roi.cache_name = lookupname  # settings.create_roi_hash(roi.name)
            if not os.path.isfile(settings.cache_dir + "/" + lookupname):
                use_cached = False
    if use_cached and False:#TODO rem
        for roi in final_rois:
            if "blood" not in roi.name:
                # load data and restore connections with it
                with open(
                    settings.cache_dir + "/" + roi.cache_name,
                    "rb",
                ) as input_file:
                    connections = pickle.load(input_file)
                restore_connections(connections, vessels, final_rois)
    else:
        # Main processing logic
        from scipy.spatial import KDTree


        def find_closest_vessel(roi_points, tree, mappings, vessel_type):
            if tree is None or len(roi_points) == 0:
                return None, None, None

            min_dist = float("inf")
            closest_vessel = None
            closest_vessel_idx = None
            closest_roi_idx = None

            for roi_idx, roi_pt in enumerate(roi_points):
                dist, idx = tree.query(roi_pt)
                if dist < min_dist:
                    min_dist = dist
                    closest_vessel, closest_vessel_idx = mappings[idx]
                    closest_roi_idx = roi_idx

            if closest_vessel:
                print(f"  Found closest {vessel_type} at distance {min_dist:.2f}")
            return closest_vessel, closest_vessel_idx, closest_roi_idx
        unconnected_paths_from_end_in_relation=[]
        available_vessel_ends = []
        for v in vessels:
            # add only those with free ends:
            if v.type == "vein":
                if len(v.links_at(0, ignore_tlinks=False)) == 0 and no_links_to(v,vessels) and no_voi_links_to(v,final_rois[:-1]):
                    available_vessel_ends.append(v)
            if v.type != "vein":
                if len(v.links_at(len(v.path) - 1, ignore_tlinks=False)) == 0 and no_voi_links_to(v,final_rois[:-1]):
                    available_vessel_ends.append(v)
        to_rem=set()
        for v in voi_addition:
            for vv in v.inlets+v.outlets:
                if vv[0] in available_vessel_ends:
                    to_rem.add(vv[0])
        for v in to_rem:
            available_vessel_ends.remove(v)
        for m_index, voi_geom in enumerate(roi_arrays ):
            # len(l) is proportional to volume of mesh thus lower meshes get iterated over first
            voi,geom=voi_geom
            to_add_to_mesh = []  # collect vessels that lay in mesh
            for vessel in available_vessel_ends:
                if ends_in_mesh(vessel, voi):  # end is in meshvoxels
                    # find which indices are to be connected
                    v = vessel
                    index_v = 0
                    if v.type != "vein":
                        index_v = len(v.path) - 1
                    index_m = get_closest_index_to_point(
                        v.path[index_v], roi_arrays[m_index][1]
                    )
                    print(v.type,index_m)
                    to_add_to_mesh.append((v, index_v, index_m))
                    #also track which part ist still available for other connections
                    available_indices=not_in_mesh_indices(vessel, voi)
                    if len(available_indices)>0:unconnected_paths_from_end_in_relation.append((v,[v.path[i] for i in available_indices]))


            for vessel, index_v, index_m in to_add_to_mesh:
                if "musc" not in voi.name:
                    available_vessel_ends.remove(vessel)  # remove from  available vessels
                if m_index not in vesselends_in:
                    vesselends_in[m_index] = []
                vesselends_in[m_index].append(
                    (vessel, index_v, index_m)
                )  # only end in one, priority of smallest volume

        for m_index, values in vesselends_in.items():
            voi = roi_arrays[m_index][0]#the voi reference

            for vessel, index_v, index_m in values:
                store_info(
                    index_v, vessel, voi, index_m, connected_vessels, roi_data_cache
                )  # voi gets vein and index, vessel gets voi and voiindex and startindex
        return available_vessel_ends,roi_data_cache,unconnected_paths_from_end_in_relation


vessels_left,roi_data_cache,vesselpoints_left = connect_vessels_to_vois(final_rois, blood_roi.geometry)

# %%
for v in vessels:
    if "rca_vess2" in v.associated_vesselname:
        print([l.target_tissue.name for l in v.links_to_vois])#[<CeFloPS.simulation.common.vessel2.TissueLink object at 0x00000153B4A57310>]

# %%
for v in final_rois[0:-1]+voi_addition:
    print("....")
    print(v.name,"in",v.inlets)
    print(v.name,"out",v.outlets)
    print("....")


# %%
""" points=None
all_points=None
final_results=None
results=None
for v in final_rois:
    v.temp_mesh=None
import gc
gc.collect()
 """
vesselpoints_left

# %%
 # rest of vessels is available for connections
def distance_to_heart(point):
            return min(np.linalg.norm(point - coord) for coord in HEART_COORDS)

# Modified connection logic
def find_optimal_vessel_pair(roi_points, art_tree, vein_tree, art_mappings, vein_mappings):
    """Find artery-vein pair with max distance between them and vein close to heart"""
    # Find all candidate points within threshold
    candidates = []
    threshold = 10.0  # Adjust based on your spatial scale

    # Find nearby artery points (flatten indices)
    art_indices = art_tree.query_ball_point(roi_points, threshold)
    flat_art_indices = [idx for sublist in art_indices for idx in sublist]  # Flatten nested lists
    art_candidates = [(art_mappings[i][0], art_mappings[i][1], art_tree.data[i])
                    for i in flat_art_indices]

    # Find nearby vein points (flatten indices)
    vein_indices = vein_tree.query_ball_point(roi_points, threshold)
    flat_vein_indices = [idx for sublist in vein_indices for idx in sublist]  # Flatten nested lists
    vein_candidates = [(vein_mappings[i][0], vein_mappings[i][1], vein_tree.data[i])
                    for i in flat_vein_indices]

    # Score all possible pairs
    best_score = -np.inf
    best_pair = None

    for a_vessel, a_idx, a_pt in art_candidates:
        for v_vessel, v_idx, v_pt in vein_candidates:
            # Calculate spatial separation score
            separation = np.linalg.norm(a_pt - v_pt)
            heart_dist = distance_to_heart(v_pt)

            # Combined score (weight factors as needed)
            score = separation * 0.7 - heart_dist * 0.3

            if score > best_score:
                best_score = score
                best_pair = (a_vessel, a_idx, v_vessel, v_idx)

    return best_pair

from scipy.spatial import KDTree
kdtree_cache = {}

def point_near_points(point, voi, threshold):
    # Check if the KDTree for this VOI is already cached
    if voi not in kdtree_cache:
        # Extract the points from the ROI geometry and create a KDTree
        points = np.array(voi.geometry.get_points())
        kdtree_cache[voi] = KDTree(points)

    # Retrieve the KDTree from the cache
    tree = kdtree_cache[voi]

    # Query for the closest point in the KDTree
    distance, index = tree.query(point)

    # Check if the distance is within the threshold
    in_threshold = distance < threshold

    return index, in_threshold,distance
available_vessels=vessels_left
# name that is allowed to connect in this pass
voi_substring_names_allowed=["liver","_int"]
vessel_substring_names_allowed=["port"]
threshold=20
vessels_connected=set()



for v in available_vessels:
    if any([a in v.associated_vesselname for a in vessel_substring_names_allowed]):
        #only allow matched vessels and check against allowed vois
        for voi in final_rois[:-1]:
            if any([a in voi.name for a in voi_substring_names_allowed]):
                #check if there is a point in voi.geometry.get_points() that does have a distance lower than threshold
                p=v.path[-1] if v.type!="vein" else v.path[0]
                v_idx=len(v.path)-1 if v.type!="vein" else 0
                closest_voiindex,in_threshold,distance=point_near_points(p,voi,threshold)
                if in_threshold:
                    print(v.associated_vesselname,distance,voi.name)
                    vessels_connected.add(v)
                    store_info(
                        v_idx, v, voi, closest_voiindex, set(), roi_data_cache
                    )
for v in vessels_connected:
    available_vessels.remove(v)
#%%
#full_model_connection_visualization(vessels,final_rois[0:-1]+voi_addition)
#%%
#vis.show([(v.path,[200,100,100,100]) for v in available_vessels])
#%%
available_vessels+=[vessel for vessel in vessels if "arm" in vessel.associated_vesselname or "leg" in vessel.associated_vesselname]
available_vessels=list(set(available_vessels))
# %%
def get_potential_links_vois(final_rois, available_vessels):
    potential_paths = []
    initial_threshold = 20
    threshold_increment = 5
    max_threshold = 200

    # Precompute vessel data
    art_vessels = [v for v in available_vessels if v.type != "vein"]
    vein_vessels = [v for v in available_vessels if v.type == "vein"]
    art_points, art_mappings = create_vessel_mappings(art_vessels)
    vein_points, vein_mappings = create_vessel_mappings(vein_vessels)

    # Build KD-trees
    art_tree = KDTree(art_points) if len(art_points) > 0 else None
    vein_tree = KDTree(vein_points) if len(vein_points) > 0 else None

    for m_index, voi in enumerate(final_rois[:-1]):
        voi_points = voi.geometry.get_points()
        print(f"\nProcessing VOI {voi.name} (Index {m_index})")

        current_threshold = initial_threshold
        found = False

        # Get existing connections if available
        existing_art = voi.inlets[0] if voi.inlets else None
        existing_vein = voi.outlets[0] if voi.outlets else None

        while current_threshold <= max_threshold and not found:
            # Determine which connections to search for
            search_art = existing_art is None
            search_vein = existing_vein is None

            # Find nearby indices based on needs
            art_ids = np.array([], dtype=int)
            vein_ids = np.array([], dtype=int)

            if search_art and art_tree:
                art_ids = np.unique(np.concatenate(
                    art_tree.query_ball_point(voi_points, r=current_threshold)
                )).astype(int)

            if search_vein and vein_tree:
                vein_ids = np.unique(np.concatenate(
                    vein_tree.query_ball_point(voi_points, r=current_threshold)
                )).astype(int)

            # Handle different search scenarios
            if search_art and search_vein:
                # Original logic for finding both connections
                if len(art_ids) > 0 and len(vein_ids) > 0:
                    art_pts = art_points[art_ids]
                    vein_pts = vein_points[vein_ids]
                    distances = np.sqrt(np.sum((art_pts[:, None] - vein_pts[None, :])**2, axis=2))
                    i, j = np.unravel_index(np.argmax(distances), distances.shape)
                    best_pair = (
                        art_mappings[art_ids[i]],
                        art_ids[i],
                        vein_mappings[vein_ids[j]],
                        vein_ids[j]
                    )
                    found = True

            elif search_art and existing_vein:
                # Find artery connection using existing vein position
                vein_pos = vein_points[existing_vein[1]]  # Use stored vein points
                if len(art_ids) > 0:
                    art_pts = art_points[art_ids]
                    distances = np.linalg.norm(art_pts - vein_pos, axis=1)
                    best_idx = np.argmax(distances)
                    best_pair = (
                        art_mappings[art_ids[best_idx]],
                        art_ids[best_idx],
                        existing_vein[0],  # Existing vein vessel
                        existing_vein[1]   # Existing vein index
                    )
                    found = True

            elif search_vein and existing_art:
                # Find vein connection using existing artery position
                art_pos = art_points[existing_art[1]]  # Use stored artery points
                if len(vein_ids) > 0:
                    vein_pts = vein_points[vein_ids]
                    distances = np.linalg.norm(vein_pts - art_pos, axis=1)
                    best_idx = np.argmax(distances)
                    best_pair = (
                        existing_art[0],  # Existing artery vessel
                        existing_art[1],  # Existing artery index
                        vein_mappings[vein_ids[best_idx]],
                        vein_ids[best_idx]
                    )
                    found = True

            if not found:
                current_threshold += threshold_increment

        if found :
            # Use either found pair or existing connections
            potential_paths.append(
                (voi, best_pair[0], best_pair[1], best_pair[2], best_pair[3])
            )

    return potential_paths
""" def get_potential_links_vois(final_rois, available_vessels):
    potential_paths = []
    initial_threshold = 20
    threshold_increment = 5
    max_threshold = 200

    # Precompute vessel data
    art_vessels = [v for v in available_vessels if v.type != "vein"]
    vein_vessels = [v for v in available_vessels if v.type == "vein"]
    art_points, art_mappings = create_vessel_mappings(art_vessels)
    vein_points, vein_mappings = create_vessel_mappings(vein_vessels)

    # Build KD-trees
    art_tree = KDTree(art_points) if len(art_points) > 0 else None
    vein_tree = KDTree(vein_points) if len(vein_points) > 0 else None

    for m_index, voi in enumerate(final_rois[:-1]):
        voi_points = voi.geometry.get_points()
        voi_points_np = np.array(voi_points)  # Precompute VOI points array

        print(f"\nProcessing VOI {voi.name} (Index {m_index})")

        current_threshold = initial_threshold
        found = False

        # Get existing connections if available
        existing_art = voi.inlets[0] if voi.inlets else None
        existing_vein = voi.outlets[0] if voi.outlets else None

        while current_threshold <= max_threshold and not found:
            # Determine which connections to search for
            search_art = existing_art is None
            search_vein = existing_vein is None

            # Find nearby indices based on needs
            art_ids = np.array([], dtype=int)
            vein_ids = np.array([], dtype=int)

            if search_art and art_tree:
                art_ids = np.unique(np.concatenate(
                    art_tree.query_ball_point(voi_points, r=current_threshold)
                )).astype(int)

            if search_vein and vein_tree:
                vein_ids = np.unique(np.concatenate(
                    vein_tree.query_ball_point(voi_points, r=current_threshold)
                )).astype(int)

            # Handle different search scenarios
            if search_art and search_vein:
                # Find both artery and vein connections with priority on VOI distances <60
                if len(art_ids) > 0 and len(vein_ids) > 0:
                    art_pts = art_points[art_ids]
                    vein_pts = vein_points[vein_ids]

                    # Compute distances from VOI to each art_pt and vein_pt
                    distances_art_to_voi = np.linalg.norm(art_pts[:, np.newaxis, :] - voi_points_np[np.newaxis, :, :], axis=2)
                    d_art = np.min(distances_art_to_voi, axis=1)

                    distances_vein_to_voi = np.linalg.norm(vein_pts[:, np.newaxis, :] - voi_points_np[np.newaxis, :, :], axis=2)
                    d_vein = np.min(distances_vein_to_voi, axis=1)

                    # Compute pairwise distances between art and vein points
                    D = np.sqrt(np.sum((art_pts[:, None] - vein_pts[None, :])**2, axis=2))

                    # Create mask for pairs where both distances are under 60
                    valid_pairs_mask = (d_art[:, None] < 40) & (d_vein[None, :] < 40)

                    if np.any(valid_pairs_mask):
                        # Find max distance in valid pairs
                        masked_D = np.where(valid_pairs_mask, D, -np.inf)
                        max_idx = np.argmax(masked_D)
                    else:
                        # Fallback to original max distance
                        max_idx = np.argmax(D)

                    i, j = np.unravel_index(max_idx, D.shape)
                    best_pair = (
                        art_mappings[art_ids[i]],
                        art_ids[i],
                        vein_mappings[vein_ids[j]],
                        vein_ids[j]
                    )
                    found = True

            elif search_art and existing_vein:
                # Find artery connection using existing vein position with VOI distance <60 priority
                vein_pos = vein_points[existing_vein[1]]
                if len(art_ids) > 0:
                    art_pts = art_points[art_ids]

                    # Compute distances from VOI to each art_pt
                    distances_art_to_voi = np.linalg.norm(art_pts[:, np.newaxis, :] - voi_points_np[np.newaxis, :, :], axis=2)
                    d_art = np.min(distances_art_to_voi, axis=1)

                    # Filter arteries with VOI distance <60
                    valid_art_mask = d_art < 60
                    valid_art_indices = np.where(valid_art_mask)[0]

                    if len(valid_art_indices) > 0:
                        # Use valid arteries to find max distance to vein
                        valid_art_pts = art_pts[valid_art_indices]
                        distances = np.linalg.norm(valid_art_pts - vein_pos, axis=1)
                        best_sub_idx = np.argmax(distances)
                        best_idx = valid_art_indices[best_sub_idx]
                    else:
                        # Fallback to all arteries
                        distances = np.linalg.norm(art_pts - vein_pos, axis=1)
                        best_idx = np.argmax(distances)

                    best_pair = (
                        art_mappings[art_ids[best_idx]],
                        art_ids[best_idx],
                        existing_vein[0],
                        existing_vein[1]
                    )
                    found = True

            elif search_vein and existing_art:
                # Find vein connection using existing artery position with VOI distance <60 priority
                art_pos = art_points[existing_art[1]]
                if len(vein_ids) > 0:
                    vein_pts = vein_points[vein_ids]

                    # Compute distances from VOI to each vein_pt
                    distances_vein_to_voi = np.linalg.norm(vein_pts[:, np.newaxis, :] - voi_points_np[np.newaxis, :, :], axis=2)
                    d_vein = np.min(distances_vein_to_voi, axis=1)

                    # Filter veins with VOI distance <60
                    valid_vein_mask = d_vein < 60
                    valid_vein_indices = np.where(valid_vein_mask)[0]

                    if len(valid_vein_indices) > 0:
                        # Use valid veins to find max distance to artery
                        valid_vein_pts = vein_pts[valid_vein_indices]
                        distances = np.linalg.norm(valid_vein_pts - art_pos, axis=1)
                        best_sub_idx = np.argmax(distances)
                        best_idx = valid_vein_indices[best_sub_idx]
                    else:
                        # Fallback to all veins
                        distances = np.linalg.norm(vein_pts - art_pos, axis=1)
                        best_idx = np.argmax(distances)

                    best_pair = (
                        existing_art[0],
                        existing_art[1],
                        vein_mappings[vein_ids[best_idx]],
                        vein_ids[best_idx]
                    )
                    found = True

            if not found:
                current_threshold += threshold_increment

        if found:
            potential_paths.append(
                (voi, best_pair[0], best_pair[1], best_pair[2], best_pair[3]))

    return potential_paths """
potpaths=get_potential_links_vois(final_rois,available_vessels)
potpaths

# %%
to_show=[]
def get_closest_connection(vessel_point, voi_points):
        distances = np.linalg.norm(voi_points - vessel_point, axis=1)
        closest_idx = np.argmin(distances)
        return voi_points[closest_idx],closest_idx
def create_path(positions,color):
        assert len(positions)>1
        print(positions)
        showable_path = trimesh.load_path(
                                positions
                            )
        if len(showable_path.entities) == 0:
            #print("No entities were created. Please check the positions input.",positions)
            positions[1]=[positions[1][0],positions[1][1],positions[1][2]+0.1]
            showable_path = trimesh.load_path(
                                positions
                            )
        #print(len(positions),len(showable_path.entities))
        showable_path.colors = [color for _ in range(len(showable_path.entities))]
        return showable_path
#%%
for entry in potpaths:
    voi, a, aindex, b, bindex = entry
    voi_points = voi.geometry.get_points()
    # Plot artery connection
    if isinstance(a, tuple):
        artery_vessel, artery_path_idx = a
        artery_point = artery_vessel.path[artery_path_idx]
        voi_connection = get_closest_connection(artery_point, voi_points)
        to_show.append(create_path([artery_point, voi_connection[0]], [255, 0, 0, 255]))

    # Plot vein connection
    if isinstance(b, tuple):
        vein_vessel, vein_path_idx = b
        vein_point = vein_vessel.path[vein_path_idx]
        voi_connection = get_closest_connection(vein_point, voi_points)
        to_show.append(create_path([vein_point, voi_connection[0]], [0, 0, 255, 255]))

#vis.show(to_show)
#%%
import numpy as np

def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))

for entry in potpaths:
    voi, a, aindex, b, bindex = entry
    voi_points = voi.geometry.get_points()

    # Calculate artery connection path length
    if isinstance(a, tuple):
        artery_vessel, artery_path_idx = a
        artery_point = artery_vessel.path[artery_path_idx]
        voi_connection = get_closest_connection(artery_point, voi_points)
        artery_path_length = calculate_distance(artery_point, voi_connection[0])

        if artery_path_length>70:
            print(f"Artery Path Length: {artery_path_length}")
            print(artery_vessel.associated_vesselname)
    # Calculate vein connection path length
    if isinstance(b, tuple):
        vein_vessel, vein_path_idx = b
        vein_point = vein_vessel.path[vein_path_idx]
        voi_connection = get_closest_connection(vein_point, voi_points)
        vein_path_length = calculate_distance(vein_point, voi_connection[0])
        #print(f"Vein Path Length: {vein_path_length}")
# %%
#apply the potential links to vois:
for entry in potpaths:
    voi, a, aindex, b, bindex = entry

    voi_points = voi.geometry.get_points()
    if isinstance(a, tuple):
        artery_vessel, artery_path_idx = a
        artery_point = artery_vessel.path[artery_path_idx]
        voi_connection,v_idx = get_closest_connection(artery_point, voi_points)
        store_info(
                        artery_path_idx, artery_vessel, voi, v_idx, set(), roi_data_cache
                    )
    if isinstance(b, tuple):
        vein_vessel, vein_path_idx = b
        vein_point = vein_vessel.path[vein_path_idx]
        voi_connection,v_idx  = get_closest_connection(vein_point, voi_points)
        store_info(
                        vein_path_idx, vein_vessel, voi, v_idx, set(), roi_data_cache
                    )



# %%
def voi_traversable(voi):
    """Check if all inlets and outlets are connected through neighboring voxels in the VOI."""
    # Get all VOI points and convert to coordinate tuples
    voi_points = voi.geometry.get_points()
    if not len(voi_points):
        return False
    voi_coords = {tuple(p) for p in voi_points}

    # Collect all connection points from inlets and outlets
    connections = []

    # Process inlets and outlets
    for connections_list in [voi.inlets, voi.outlets]:
        for conn in connections_list:
            if conn[2] < 0 or conn[2] >= len(voi_points):
                return False  # Invalid index
            connections.append(tuple(voi_points[conn[2]]))

    # Edge cases: no connections or only one type
    if not connections or not voi.inlets or not voi.outlets:
        return False

    # Check if all connections exist in VOI
    if not all(c in voi_coords for c in connections):
        return False

    # BFS setup
    from collections import deque
    visited = set()
    q = deque()

    # Start from first connection point
    start = connections[0]
    q.append(start)
    visited.add(start)

    # 6-connectivity directions (adjacent voxels)
    directions = [(dx, dy, dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1) if (dx, dy, dz) != (0,0,0)]

    # Perform BFS
    while q:
        current = q.popleft()

        # Explore neighbors
        for dx, dy, dz in directions:
            neighbor = (current[0]+dx, current[1]+dy, current[2]+dz)

            if neighbor in voi_coords and neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)

    # Check if all connection points are reachable
    return all(c in visited for c in connections)
not_traversable_vois=[]
for voi in final_rois[:-1]:
    if voi_traversable(voi):
        ...#print(f"VOI {voi.name} is traversable")
    else:
        print(f"VOI {voi.name} is NOT traversable")
        not_traversable_vois.append(voi)

# %%
len(not_traversable_vois)

#%%
# %%
#create a visualization for the vois, their direct connection vectors between inlets and outlets as well as connections to the vesselsystem:

#full_model_connection_visualization(vessels,final_rois[0:-1]+voi_addition)
#%%
import importlib
import CeFloPS.simulation.settings as settings

# Reload the settings module
importlib.reload(settings)
#%%
for v in [s for s in final_rois[:-1] if s not in voi_addition]:
    v.create_flow_vectorfield(store_loc=True)
#%%
for v in  [v for v in final_rois[:-1] if v not in voi_addition]:
    fit_capillary_speed([v ],visual=False,plot=False,store_loc=True)
#%%
for roi in final_rois:
            if "blood" not in roi.name:
                with open(settings.cache_dir + "/" + roi.cache_name, "wb") as handle:
                    pickle.dump(
                        roi_data_cache[roi.name],
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
# %%
# Instead of pickle, use dill


# In your saving code:
import dill
with open(settings.cache_dir +"final_rois.dill", "wb") as f:
    dill.dump((final_rois+voi_addition,blood_roi,roi_mapping_str), f)
#%%
assert vessels==blood_roi.geometry
for vessel in vessels:
    #fix radii
    vessel.pre_cached_profiles=False
    for vol in vessel.volumes:
        vol.radius=vol.radius/1000
        vol.length=vol.length/1000
    vessel.register_volumes()
    vessel.unregister_functions()
#%%
import pickle
with open(settings.cache_dir + "/final_rois.pickle", "wb") as handle:
        pickle.dump((final_rois,blood_roi,roi_mapping_str), handle, protocol=pickle.HIGHEST_PROTOCOL)


#%%
"""
import pickle
with open(
    r"D:\final_rois.pickle",
    "rb",
) as input_file:
    loaded_rois = pickle.load(input_file) """

#%%
final_rois
"""
# %%
to_show=[(v.path,[200,100,100,100]) for v in vessels]
for voi in final_rois[:-1]:
    source_indices=[vv[2] for vv in voi.inlets]
    target_indices=[vv[2] for vv in voi.outlets]
    if any([i in target_indices for i in source_indices]):
        print(voi.name)
        to_show.append(([voi.geometry.get_points()[i] for i in source_indices],[200,100,100,200]))
        to_show.append(([voi.geometry.get_points()[i] for i in target_indices],[100,100,200,200]))
    to_show.append((voi.geometry.get_points(),[100,100,100,20]))
vis.show(to_show)

# %%
to_show=[]
for voi in not_traversable_vois:
    source_indices=[vv[2] for vv in voi.inlets]
    target_indices=[vv[2] for vv in voi.outlets]
    print(voi.name)
    to_show.append(([voi.geometry.get_points()[i] for i in source_indices],[200,100,100,200]))
    to_show.append(([voi.geometry.get_points()[i] for i in target_indices],[100,100,200,200]))
    to_show.append((voi.geometry.get_points(),[100,100,100,20]))
vis.show(to_show)
#%%
while(any([v.volume<10 for v in not_traversable_vois])):
    for v in not_traversable_vois:
        if v.volume<10:
            if v in final_rois:final_rois.remove(v)
            not_traversable_vois.remove(v)
#%%
#now just connect the rest of the arm to the closest, make all that are connected to other arms available too
to_show=[]
for v in available_for_rest:
    to_show.append((v.path,[200,200,200,200]))
vis.show(to_show)

#%%
available_for_rest=list(set(available_vessels+[v for v in vessels if "arm" in v.associated_vesselname]))
potpaths=get_potential_links_vois(not_traversable_vois,available_for_rest)
potpaths
#%%
#apply the potential links to vois:
for entry in potpaths:
    voi, a, aindex, b, bindex = entry

    voi_points = voi.geometry.get_points()
    if isinstance(a, tuple):
        artery_vessel, artery_path_idx = a
        artery_point = artery_vessel.path[artery_path_idx]
        voi_connection,v_idx = get_closest_connection(artery_point, voi_points)
        store_info(
                        artery_path_idx, artery_vessel, voi, v_idx, set(), roi_data_cache
                    )
    if isinstance(b, tuple):
        vein_vessel, vein_path_idx = b
        vein_point = vein_vessel.path[vein_path_idx]
        voi_connection,v_idx  = get_closest_connection(vein_point, voi_points)
        store_info(
                        vein_path_idx, vein_vessel, voi, v_idx, set(), roi_data_cache
                    )

#%%
#show missing ones
still_left= [v for v in not_traversable_vois if len(v.inlets)==0]
to_show=[(v.path,[10,10,10,10]) for v in vessels]+[(v.path,[200,10,10,100]) for v in available_for_rest]
for v in still_left:
    to_show.append((v.geometry.get_points(),[100,100,100,100]))
vis.show(to_show)
#%%
import dill
sys.setrecursionlimit(10000)
dill.dump_session('notebook_env.db')

#%%
import dill
dill.load_session('notebook_env.db') """

# %%
def reset_voi_connections(vessels,frois,voi_addition):
    for vessel in vessels:
        to_rem=set()
        for l in vessel.links_to_vois:
            if l.target_tissue in frois:
                to_rem.add(l)
        for l in to_rem:
            l.source_vessel.links_to_vois.remove(l)
    for v in frois[:-1]:
        v.inlets,v.outlets=[],[]

# %%
#reset_voi_connections(vessels,final_rois,voi_addition)
