# vessel creation methods that are based on a newer approach for less stringend assumptions about a meshesh structure.
# The methods combine a voxel approach for extracting a centerline for a mesh, a graph approach for deciding which voxels are relevant if more then 2 are adjacent to each other and tangential mesh slicing along a plane in predetermined deltas. 
# The cut is analyzed to a point on its convex hull to find which points belong to the closest "meshwall". Finally an elllipsis is fitted onto the projected points to extract a diameter at that point and also correct the voxelcoordinate to be at the "real" center instead of a discrete centerpoint.
import numpy as np
from scipy.spatial.distance import pdist
from CeFloPS.simulation.common.vessel_functions import calculate_centered_point, points_from_ring,standard_speed_function
from CeFloPS.simulation.common.vessel2 import Vessel
from CeFloPS.data_processing.voxel_fun import  get_adjacent_indices,Graph,longest_path_from_node
from skimage.morphology import skeletonize
import networkx as nx 
from CeFloPS.data_processing.geometric_fun import get_area_from_ring, tilt_vector, PI, get_center_ellipsis,translate_2d_to_3d,project_3d_points_to_2d
import trimesh
from skimage.measure import EllipseModel 



import networkx as nx
from scipy.spatial.distance import pdist, squareform
import logging
from CeFloPS.logger_config import setup_logger 
logger = setup_logger(__name__, level=logging.CRITICAL)

def calculate_closest_ring(cut, origin, max_jump_distance=2.5, all_rings=False, debug=False):
    """
    Identify and return the closest set of points ('ring') from a cut based on a given origin.
    """
    assert len(origin) == 3, "Origin must be a 3D coordinate."
 
    if len(cut)==0:
        return []

    # sort by distance
    cutpoints = np.unique(np.array(cut), axis=0)

    # return if too few points
    if len(cutpoints) < 2:
        return cutpoints.tolist()

    # Calculate pairwise distances and find max distance pair
    distances = pdist(cutpoints)
    max_dist_pair_indices = np.unravel_index(distances.argmax(), (len(cutpoints),) * 2)
    outer_point = cutpoints[max_dist_pair_indices[0]]

    # Sort points by distance from the outermost point
    distances_to_outer = np.linalg.norm(cutpoints - outer_point, axis=1)
    sorted_indices = np.argsort(distances_to_outer)
    sorted_cutpoints = cutpoints[sorted_indices]
    logger.print("min distances", min(distances_to_outer),"max distances", max(distances_to_outer),"avg distances", np.mean(distances_to_outer))
    if debug:
        logger.print("Distances sorted from farthest:", distances_to_outer[sorted_indices])

    # Identify rings by looking for significant jumps in sorted distances
    distance_diffs = np.diff(distances_to_outer[sorted_indices])
    split_indices = np.where(distance_diffs > max_jump_distance)[0] + 1

    ring_collection = np.split(sorted_cutpoints, split_indices)

    if not all_rings:
        # Find and return the ring closest to the origin
        centers_of_rings = [np.mean(ring, axis=0) for ring in ring_collection]
        origin_distances = [np.linalg.norm(center - origin) for center in centers_of_rings]
        
        if debug:
            logger.print("Origin distances to ring centers:", origin_distances)

        closest_ring_idx = np.argmin(origin_distances)
        return ring_collection[closest_ring_idx] if ring_collection else []

    return [ring for ring in ring_collection]

import matplotlib.pyplot as plt
def calculate_closest_ring_no_threshold(cut, origin, all_rings=False, debug=False):
    """
    Identify and return the closest set of points ('ring') from a cut based on a given origin,
    ensuring at least 25% of points are included and using the maximum of 25% distance and std deviation as threshold.
    """
    assert len(origin) == 3, "Origin must be a 3D coordinate."

    if len(cut) == 0:
        return []

    cutpoints = np.unique(np.array(cut), axis=0)

    if len(cutpoints) < 2:
        return cutpoints.tolist()

    # Calculate pairwise distances and find max distance pair
    distances = pdist(cutpoints)
    max_dist_pair_indices = np.unravel_index(distances.argmax(), (len(cutpoints),) * 2)
    max_dist_points = cutpoints[list(max_dist_pair_indices)]

    # Determine the point from the max distance pair that is closest to the origin
    origin_distances_to_max_dist_points = [np.linalg.norm(point - origin) for point in max_dist_points]
    closest_to_origin_idx = np.argmin(origin_distances_to_max_dist_points)
    reference_point = max_dist_points[closest_to_origin_idx]

    # Sort points by distance from the reference point
    distances_to_reference = np.linalg.norm(cutpoints - reference_point, axis=1)
    sorted_indices = np.argsort(distances_to_reference)
    sorted_cutpoints = cutpoints[sorted_indices]
    sorted_distances = distances_to_reference[sorted_indices]

    # Calculate the first derivative of sorted distances
    distance_diffs = np.diff(sorted_distances)
    plt.plot(distance_diffs)
    # Determine the threshold based on the maximum of std deviation and 25% percentile
    std_diff = np.std(distance_diffs)
    percentile_25 = np.percentile(sorted_distances, 25)
    distance_threshold = max(std_diff, percentile_25)
    logger.print(distance_threshold)
    # Determine significant jumps > threshold
    significant_jumps = np.where(distance_diffs > distance_threshold)[0] + 1
    
    # Use indices to split points into rings
    ring_collection = np.split(sorted_cutpoints, significant_jumps)

    if not all_rings:
        # Ensure at least 25% of points are included in some split
        min_points = max(1, int(0.25 * len(sorted_cutpoints)))
        ring_collection = [ring for ring in ring_collection if len(ring) >= min_points]
        if not ring_collection:
            ring_collection = [sorted_cutpoints]  # Fallback to include all

        # Find and return the ring closest to the origin
        centers_of_rings = [np.mean(ring, axis=0) for ring in ring_collection]
        origin_distances = [np.linalg.norm(center - origin) for center in centers_of_rings]
        
        if debug:
            logger.print("Origin distances to ring centers:", origin_distances)

        closest_ring_idx = np.argmin(origin_distances)
        return ring_collection[closest_ring_idx] if ring_collection else []

    return [ring for ring in ring_collection] 





def get_endpoints(skeleton_voxels, skeleton_shape):
    """Identify all endpoints in the skeleton."""
    endpoints = set()
    for voxel in skeleton_voxels:
        x, y, z = voxel
        adjacent_coords = get_adjacent_indices(voxel, skeleton_shape)
        filled_neighbors = sum((adj in skeleton_voxels) for adj in adjacent_coords)
        if filled_neighbors == 1:  # An endpoint has exactly one neighbor, multiple can exist in a skeleton but at least 2
            endpoints.add(voxel)
    return endpoints

def calculate_path_length(path):
    """Calculate Euclidean length of a path."""
    length = 0
    for i in range(1, len(path)):
        v1 = np.array(path[i-1])
        v2 = np.array(path[i])
        length += np.linalg.norm(v2 - v1)
    return length

def find_longest_shortest_path(skeleton_graph, endpoints):
    longest_distance = 0
    best_path = []

    for endpoint in endpoints:
        for target in endpoints:
            if endpoint == target:
                continue
            try:
                # Find shortest path in terms of steps
                path = nx.shortest_path(skeleton_graph, source=endpoint, target=target)
                # calc pathlen (in this case we use discrete steps and no euclidian distance)
                distance = len(path)
                if distance > longest_distance:
                    longest_distance = distance
                    best_path = path
            except nx.NetworkXNoPath:
                pass

    return best_path

def build_full_graph(voxel_set, skeleton_shape):
    #build nxgraph from voxeladjacency
    graph = nx.Graph()

    for voxel in voxel_set:
        adjacent_coords = get_adjacent_indices(voxel, skeleton_shape)
        for adj in adjacent_coords:
            if adj in voxel_set:
                graph.add_edge(voxel, adj)

    return graph
 

def extract_centerline(trimesh_object,step_size=1):
    voxelgrid=trimesh_object.voxelized(pitch=step_size)#use step size for pitch
    logger.print("Voxel grid created")
    voxelgrid = voxelgrid.fill()
    logger.print("Voxel grid filled")
    if np.all(voxelgrid.matrix == 0):
        logger.print("Voxel grid matrix is all zeros after fill()")
    if(len(voxelgrid.matrix))<50:
        step_size=.1
        voxelgrid=trimesh_object.voxelized(pitch=step_size)#use step size for pitch 
        voxelgrid = voxelgrid.fill()
    voxelgrid_skeleton = skeletonize(voxelgrid.matrix)
    still_set_voxels = set(tuple(entry) for entry in np.argwhere(voxelgrid_skeleton))
    skeleton = voxelgrid_skeleton
    skeleton_voxels = still_set_voxels 
    graph = build_full_graph(skeleton_voxels, skeleton.shape)
    endpoints = get_endpoints(skeleton_voxels, skeleton.shape)

    # find the longest shortest path through the skeleton
    shortest_path = find_longest_shortest_path(graph, endpoints)    
     
    #voxel_indices=nx.shortest_path(G, l1, l2)
    voxel_indices=shortest_path
    logger.print(voxel_indices)
    #voxel_indices=extend_skeleton_to_edge(voxel_indices,voxelgrid.matrix)
    parts=list(extend_skeleton_to_edge(voxel_indices[::2],voxelgrid.matrix))#pass every second point to alleviate ruggedness
    # for the partial extention, also calculate the longest shortest path and then concat everything together
    for i,skeleton_voxels in enumerate(parts):
        graph = build_full_graph(skeleton_voxels, skeleton.shape)
        endpoints = get_endpoints(skeleton_voxels, skeleton.shape)

        # find the longest shortest path through the skeleton
        shortest_path = find_longest_shortest_path(graph, endpoints)  
        if i==0:
            #beginning segment
            voxel_indices=shortest_path+voxel_indices
            logger.print(shortest_path,"I ZERO")
        else:
            voxel_indices=voxel_indices+shortest_path
             
        
    output = []
    for coords in voxel_indices: 
        # Get voxel center
        voxel_center = voxelgrid.transform.dot(np.append(coords, 1))[:3]

        # Generate points to check: center + 8 corners, as some do not lie within mesh which leads to bad plane positions
        points = [voxel_center]
        half_step = step_size / 2
        for dx in (-half_step, half_step):
            for dy in (-half_step, half_step):
                for dz in (-half_step, half_step):
                    points.append(voxel_center + np.array([dx, dy, dz]))

        # Batch check inside attribute
        is_inside = trimesh_object.contains(points)

        # use first point found inside
        for pt, inside in zip(points, is_inside):
            if inside:
                output.append(pt)
                break
        else:
            # Fallback: closest surface point
            closest = trimesh.proximity.closest_point(trimesh_object, [voxel_center])[0]
            output.append(closest)
    output=[np.asarray(e) if len(e)==3 else np.asarray(e[0]) for e in output]

    return output#,parts#[output[0]]+output[1:-1:10]+[output[-1]]
 
def calculate_tangent_with_weights(points, is_end, num_points=10, weight_at_end=1.25, weight_inside=0.8):
    """
    Calculate a tangent vector at an endpoint using a weighted average of normalized vectors
    from each point in the segment to the endpoint, with more weight on earlier points.
    """
    logger.print(len(points))
    if is_end:
        # Endpoint is the last point in the list
        endpoint = points[-1]
        relevant_points = points[max(-num_points-1, -len(points)):-1]
    else:
        # Endpoint is the first point in the list
        endpoint = points[0]
        relevant_points = points[1:num_points+1]

    if len(relevant_points) < 1:
        return np.array([0, 0, 0])

    # Calculate and weight vectors from each point to the endpoint
    direction_vectors = []
    weights = np.linspace(weight_at_end, weight_inside, num=len(relevant_points))  # Linearly decreasing weights

    for i, point in enumerate(relevant_points):
        vector = np.array(endpoint) - np.array(point)
        norm = np.linalg.norm(vector)
        if norm > 0:
            normalized_vector = vector / norm
            direction_vectors.append(normalized_vector * weights[i])

    # Compute the weighted average of the vectors
    tangent = np.sum(direction_vectors, axis=0)
    final_norm = np.linalg.norm(tangent)
    
    return tangent / final_norm if final_norm > 0 else np.array([0, 0, 0])

def append_voxel_extension(start_idx, tangent, grid, max_steps=100):
    """
    Extend in the given direction from a start point, skipping duplicates.
    """
    current_point = np.array(start_idx, dtype=float)
    newly_added_points = []
    previous_point = None  # Track the last added point to avoid duplicates

    for _ in range(max_steps):
        current_point += tangent
        next_voxel = np.round(current_point).astype(int)
        next_tuple = tuple(next_voxel)

        # Check boundaries and voxel status
        if (0 <= next_voxel[0] < grid.shape[0] and
            0 <= next_voxel[1] < grid.shape[1] and
            0 <= next_voxel[2] < grid.shape[2]):
            
            if grid[next_tuple]:  # Part of mesh
                if previous_point == next_tuple:
                    continue  # Skip duplicates and continue
                newly_added_points.append(next_tuple)
                previous_point = next_tuple  # Update last added point
            else:
                break  # Stop when exiting the mesh boundary
        else:
            break  # Stop when out of bounds

    return newly_added_points

def extend_skeleton_to_edge(voxel_indices, voxel_grid):
    """
    Extend the path in the direction anchored by calculated tangents.
    """
    # Calculate tangents at ends
    start_tangent = calculate_tangent_with_weights(voxel_indices, False)
    end_tangent = calculate_tangent_with_weights(voxel_indices, True)
    logger.print(start_tangent, end_tangent)
    # Generate extensions
    start_extension = append_voxel_extension(voxel_indices[0], start_tangent, voxel_grid)
    end_extension = append_voxel_extension(voxel_indices[-1], end_tangent, voxel_grid)

    return start_extension, end_extension

def filter_centerline_by_distance(centerline, min_distance):
    """
    Remove points from the centerline so that each point is at least min_distance from the previous one.
    The first and last points will not be removed.
    
    :param centerline: List of points (each point is an array-like of coordinates).
    :param min_distance: Minimum distance required between consecutive points.
    :return: A filtered list of centerline points.
    """
    if len(centerline) < 2:
        return centerline  # No filtering needed for zero or one point

    filtered_centerline = [centerline[0]]  # Start with the first point

    for i in range(1, len(centerline) - 1):
        last_kept_point = filtered_centerline[-1]
        current_point = centerline[i]
        
        # Calculate the Euclidean distance
        distance = np.linalg.norm(current_point - last_kept_point)
        
        if distance >= min_distance:
            filtered_centerline.append(current_point)

    filtered_centerline.append(centerline[-1])  # Always include the last point

    return filtered_centerline
def compute_centerline_tangent(centerline, i, window_size=5):
    assert 0 <= i < len(centerline), "Index i must be within the valid range of centerline points"
    centerline=filter_centerline_by_distance(centerline,1)
    num_points = len(centerline)
    half_window = max(1, window_size // 2)
    
    # Determine start and end indices for the window
    start_idx = max(0, i - half_window)
    end_idx = min(num_points, i + half_window + 1)

    if end_idx <= start_idx:
        raise ValueError("Window size too small or no valid window found around the point")

    #a vector to accumulate the tangential vectors
    tangent_vector = np.zeros(3)

    # Add vectors pointing to the current point
    for j in range(start_idx, i):
        vector_to_current = centerline[i] - centerline[j]
        if np.linalg.norm(vector_to_current) > 0:
            tangent_vector += vector_to_current / np.linalg.norm(vector_to_current)

    # Subtract vectors pointing away from the current point
    for j in range(i + 1, end_idx):
        vector_from_current = centerline[j] - centerline[i]
        if np.linalg.norm(vector_from_current) > 0:
            tangent_vector -= vector_from_current / np.linalg.norm(vector_from_current)

    # Normalize the average tangential vector
    norm = np.linalg.norm(tangent_vector)
    return tangent_vector / norm if norm > 0 else tangent_vector

def change_i_on_closeness(position_vector, centerline_points, current_i):
    """
    Find the index of the closest point in centerline_points to the given position_vector. Enforce that it is not the previous one
    """ 
    if current_i==len(centerline_points)-1:
        return current_i
    position = np.array(position_vector)
    points = np.array(centerline_points[current_i+1::]) 
    distances = np.linalg.norm(points - position, axis=1)
    
    # find index of the smallest distance
    closest_index = np.argmin(distances)
    
    return closest_index+current_i+1
def threshold_close(position_vector, next_vector,threshold=1):
    position_vector = np.array(position_vector)
    next_vector = np.array(next_vector) 
    distance = np.linalg.norm(position_vector - next_vector)
     
    return distance < threshold
    
def compute_normal(centerline_points,i):
    normal= np.array(centerline_points[i+1])-np.array(centerline_points[i]) if i<len(centerline_points)-1 else np.array(centerline_points[-1])-np.array(centerline_points[-2])
    norm = np.linalg.norm(normal)
    return normal / norm if norm > 0 else normal

def compute_normal_from_previous_centres(previouspoints):
    normal= np.array(previouspoints[-1])-np.array(previouspoints[-2])
    norm = np.linalg.norm(normal)
    return normal / norm if norm > 0 else normal


def c_normal(A,toB):
    normal= -np.array(A)+np.array(toB)
    norm = np.linalg.norm(normal)
    return normal / norm if norm > 0 else normal

from sklearn.cluster import DBSCAN
import trimesh

def identify_closest_ring(cut, origin, eps=3.6, min_samples=3): 
    
    if cut is None or len(cut) < 2:
        return []

    # Convert to numpy array
    cut_points = np.array(cut)

    # Use DBSCAN to identify clusters (rings) in the cut points
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(cut_points)
    labels = clustering.labels_

    # Extract unique labels representing different rings
    unique_labels = set(labels)
    
    if -1 in unique_labels:
        unique_labels.remove(-1)  # Remove noise or outliers

    # Calculate centers and find the ring closest to the origin
    min_distance = float('inf')
    closest_ring = []

    for label in unique_labels:
        ring_points = cut_points[labels == label]
        center_of_ring = np.mean(ring_points, axis=0)
        #center_of_ring=ring_points[0]
        distance_to_origin = np.linalg.norm(center_of_ring - origin)

        if distance_to_origin < min_distance:
            min_distance = distance_to_origin
            closest_ring = ring_points

    return closest_ring

def minimize_cut_area(mesh, initial_normal, position_vector, tolerance=1e-3,debug=False, max_iter=100, max_angle_degrees=30):
    # Normalize the initial normal vector and save the original
    normal_vector = np.array(initial_normal, dtype=np.float64)
    normal_vector /= np.linalg.norm(normal_vector)
    original_initial_normal = normal_vector.copy()
    
    previous_area = float('inf')
    area_not_minimal = True
    base_perturbations = np.identity(3) * 0.4
    perturbations = []
    for base in base_perturbations:
        perturbations.append(base)
        perturbations.append(-base)
    perturbations = np.array(perturbations)
    iterations = 1
    best_cut = None

    # precompute cosine of maximum allowed angle if provided
    cos_max_angle = None
    if max_angle_degrees is not None:
        max_angle_rad = np.deg2rad(max_angle_degrees)
        cos_max_angle = np.cos(max_angle_rad)

    while area_not_minimal and iterations <= max_iter:
        best_normal = normal_vector.copy()
        smallest_area = previous_area
        current_perturbations = perturbations / iterations  # Scale perturbations by iteration

        for perturbation in current_perturbations:
            # generate test normal
            test_normal = normal_vector + perturbation
            test_norm = np.linalg.norm(test_normal)
            if test_norm == 0:
                continue
            test_normal /= test_norm

            # Check angular constraint against original initial normal
            if cos_max_angle is not None:
                dot_product = np.dot(test_normal, original_initial_normal)
                dot_product = np.clip(dot_product, -1.0, 1.0)  # for numerical inaccuracies
                if dot_product < cos_max_angle:
                    continue  # Skip if angle exceeds max angle degrees parameter

            # Compute intersection with the test plane
            try:
                cut_result = trimesh.intersections.mesh_plane(
                    mesh, test_normal, position_vector,
                    return_faces=False, local_faces=None, cached_dots=None
                )
                cut_result = [x for sub in cut_result for x in sub]
            except Exception:
                cut_result = []

            # Calculate cut area if valid
            total_area = 0
            if len(cut_result)>6:
                #print(cut_result)
                r = calculate_cut_area(np.asarray(cut_result), position_vector, test_normal,eps=max(mesh.edges_unique_length.mean()*1.1,1.2),debug=debug)
                #print(r)
                if r is None or r==np.inf or r[0] is None:
                    continue
                area_ellipsis, ring = r
                xc, yc, a, b, theta = area_ellipsis.params
                area = a * b * PI
                total_area = area

            # Update best normal and area if improvement found
            if 0 < total_area < smallest_area:
                smallest_area = total_area
                best_normal = test_normal.copy()
                best_cut = (xc, yc, area, ring, best_normal) 


        # Convergence Check
        if abs(previous_area - smallest_area) <= tolerance:
            area_not_minimal = False
        else:
            previous_area = smallest_area
            normal_vector = best_normal.copy()
            iterations += 1

    return best_cut
def minimize_cut_areass(mesh, initial_normal, position_vector, tolerance=1e-3, max_iter=100):
    normal_vector = np.array(initial_normal, dtype=np.float64)
    normal_vector /= np.linalg.norm(normal_vector)
    previous_area = float('inf')
    area_not_minimal = True
    base_perturbations = np.identity(3) * 0.4
    perturbations = []
    for base in base_perturbations:
        perturbations.append(base)
        perturbations.append(-base)
    perturbations = np.array(perturbations)
    iterations = 1
    best_cut = None

    while area_not_minimal and iterations <= max_iter:
        best_normal = normal_vector.copy()
        smallest_area = previous_area
        current_perturbations = perturbations / iterations

        for perturbation in current_perturbations:
            test_normal = normal_vector + perturbation
            test_norm = np.linalg.norm(test_normal)
            if test_norm == 0:
                continue
            test_normal /= test_norm  # Ensure unit length

            # Compute intersection with the test plane
            try:
                cut_result = (
                    trimesh.intersections.mesh_plane(
                        mesh,
                        test_normal,
                        position_vector,
                        return_faces=False,
                        local_faces=None,
                        cached_dots=None,
                    ),
                    position_vector,
                    )
                cut_result = [x for sub in cut_result[0] for x in sub]
            except:
                cut_result = []
            total_area=0
            if cut_result: 
                    r = calculate_cut_area(np.asarray(cut_result), position_vector, test_normal)
                    if r==np.inf or r is None:
                        return np.inf
                    area_ellipsis,ring=r
                    xc, yc, a, b, theta = area_ellipsis.params
                    area = a * b * PI
                    total_area =area

            if total_area > 0 and total_area < smallest_area:
                smallest_area = total_area
                best_normal = test_normal.copy()
                best_cut=(xc, yc,area,ring,best_normal) 

        # Check convergence
        if abs(previous_area - smallest_area) <= tolerance:
            area_not_minimal = False
        else:
            previous_area = smallest_area
            normal_vector = best_normal.copy()
            iterations += 1

    return best_cut
def minimize_cut_areas(mesh, initial_normal, position_vector, tolerance=1e-3):
    normal_vector = np.array(initial_normal)
    step_size = 0.1
    previous_area = float('inf')
    area_not_minimal = True

    # Constant small perturbation vectors
    base_perturbations=np.identity(3) * .4
    perturbations = []
    for base in base_perturbations:
            perturbations.append(base)   # Positive direction
            perturbations.append(-base)  # Negative direction

    perturbations=np.asarray(perturbations, dtype=np.float64)
    blocked=set()
    iterations=1 
    while area_not_minimal:
        best_normal = normal_vector
        smallest_area = previous_area
        #logger.print("smallest",smallest_area)
        iterations+=.1
        perturbations=perturbations/iterations
        for xx,perturbation in enumerate(perturbations):
            if xx not in blocked:
                for direction in [-1, 1]:  # Positive and negative directions
                    test_normal = normal_vector + direction * perturbation
                    test_normal /= np.linalg.norm(test_normal)  # Ensure it remains a unit vector

                    # Perform the cut with the test normal
                    #cut_result = trimesh.intersections.mesh_plane(mesh, normal=test_normal, origin=position_vector)
                    cut_result = (
                    trimesh.intersections.mesh_plane(
                        mesh,
                        test_normal,
                        position_vector,
                        return_faces=False,
                        local_faces=None,
                        cached_dots=None,
                    ),
                    position_vector,
                    )
                    cut_result = [x for sub in cut_result[0] for x in sub]
                    if cut_result is not None:
                        cut_points = np.array(cut_result)
                        r = calculate_cut_area(cut_points, position_vector, test_normal)
                        if r==np.inf:
                            return r
                        area_ellipsis,ring=r
                        xc, yc, a, b, theta = area_ellipsis.params
                        area = a * b * PI
                        if area < smallest_area:
                            smallest_area = area
                            best_normal = test_normal
                            best_cut=(xc, yc,area,ring,best_normal)
                        else:
                            #block direction if it didnt decrease area size
                            blocked.add(xx)

        # Check for termination
        area_not_minimal = abs(previous_area - smallest_area) > tolerance
        previous_area = smallest_area
        normal_vector = best_normal

    return best_cut

def calculate_cut_area(cut_points, position_vector, normal_vector,eps,debug=False):
    if not len(cut_points):
        return float('inf')

    ring = identify_closest_ring(cut_points, position_vector,eps=eps)
    if len(ring) > 6:
        points_to_fit = project_3d_points_to_2d(ring, position_vector, normal_vector)
        ellipsis = EllipseModel()
        #print("fit",points_to_fit)
        if ellipsis.estimate(points_to_fit): 
            if debug:
                return ellipsis,cut_points
            return ellipsis,ring
    return float('inf')
def average_point_distance(points):
    """
    Calculate the average distance between consecutive points in a list.
    
    Args:
        points (list): A list of points, where each point is an array-like object (e.g., list or numpy array).
        
    Returns:
        float: The average distance between consecutive points.
    """
    # Calculate distances between consecutive points
    distances = []
    for i in range(len(points) - 1):
        p1 = np.array(points[i])
        p2 = np.array(points[i + 1])
        dist = np.linalg.norm(p2 - p1)
        distances.append(dist)
    
    if not distances:
        return 0.0  # If there are no points or only one point, return 0
    
    return np.mean(distances)
def extract_diameter_and_centre_from_cut_two_sided(centerline_points, mesh, step_size=1, window_size=5):
    used_smaller_steps=False
    if "brain" or "pulmonary" in mesh.metadata["file_name"]:
        step_size=0.4
        used_smaller_steps=True
    rings=[]
    ref_indices=[]
    #assert all(mesh.contains(centerline_points)),1
    cp=centerline_points.copy()#[1:-1]
    #assert all(mesh.contains(centerline_points)),2
    if len(centerline_points)>=30:
        logger.print("stays",len(centerline_points))
        centerline_points=filter_centerline_by_distance(centerline_points,1)
    else:
        #use a smaller voxel size to obtain centerline points
        logger.print("pre",len(centerline_points))
        centerline_points = extract_centerline(mesh,step_size=0.2)
        logger.print("post",len(centerline_points))
    first=True 
    normal_vector = None
    
    used_normals=[]
    #we split the whole thing into two parts that are concatenated later. This is due to the fact, that the inner region of a vessel is most often way easier to analyze and has a less complex shape than its ends.
    #Another advantage is, that we do not have to rely on correct ending points as we may use normals from previous centerpoints to cut along the vessels shape based on the initial normal vectors from the skeleton that should be correct at the middle of the mesh.
    centresA,centresB,radiiA,radiiB=[],[],[],[]
    middle = len(centerline_points)//2
    side_A=centerline_points[middle:0:-1]#exludes half point and points towards the first end
    side_B=centerline_points[middle::]
    assert len(side_A)>1 and len(side_B)>1, "sides are too small, total: "+str(len(centerline_points))+" len A: "+str(len(side_A))+", B: "+str(len(side_B))
    assert(tuple(side_A[0])==tuple(side_B[0])), ("sideA and sideB",side_A[0],side_B[0])
    logger.print("sidepath lengths:",len(side_A),len(side_B),"original:",len(centerline_points))
    for side,centres,radii in [(side_A,centresA,radiiA),(side_B,centresB,radiiB)]:
        
        cut=True
        first=True
        i = 0
        logger.print(np.linalg.norm(np.asarray(side[0])-np.asarray(side[1])))
        #TODO fallback if only one point per side! use other point to form first normal eg -2 of prev/1 of next
        first_normal=c_normal(side[0],side[1])#first normal pointing towards end of vessel, taken from shape
        first_center=side[0]
        #assert all(mesh.contains([first_center])), first_center
        mod_normal=None
        logger.print("First:",first_center)
        while(cut):
            careful_append=False
            #logger.print("cut: ",len(centres))
            if first:
                logger.print(first)
                position_vector = np.array(first_center) 
                logger.print(position_vector)
                normal_vector=first_normal
            else:
                #we have previous position and normal vector as well as at least one centered point:
                normal_vector=mod_normal
                position_vector = position_vector + step_size * normal_vector
                if not all(mesh.contains([position_vector])):
                    careful_append=True #not in mesh anymore (extends end or is at beginning)
            
            #create a cut and minimize area:
            """ if first:
                r=minimize_cut_area(mesh, initial_normal=normal_vector, position_vector=position_vector,max_angle_degrees=None)
            else: """
            angle_limit=40
            if first:
                angle_limit = 160
            r=minimize_cut_area(mesh, initial_normal=normal_vector, position_vector=position_vector,max_angle_degrees=angle_limit,debug=first)
            if first:
                logger.print(position_vector)    
            if r==np.inf or r is None:
                logger.print("INF")
                break
            xc, yc, area,ring,mod_normal = r
            #if first:
            rings.append(ring)
            used_normals.append((position_vector,normal_vector))
            c_center=translate_2d_to_3d([[xc, yc]], position_vector, mod_normal)[0] 
            
            position_vector=c_center.copy()#centralise pos vector
            used_normals.append((position_vector,normal_vector))
            
            if careful_append:
                #check if the new center point has a vastly differenct relative location compared to previous ones
                
                avg_dist=average_point_distance(centres)
                if avg_dist!=0:#if 0, append as its the second point
                    if not avg_dist*2 >= np.linalg.norm(c_center - centres[-1]):
                        logger.print("away from avg dist")
                        break#detected a new point with vastly different distance
            logger.print("First append",c_center, position_vector,normal_vector)
            if first:
                logger.print("First append",c_center, position_vector,normal_vector)
            first=False
            centres.append(c_center) 
            ref_indices.append(i)
            radii.append(math.sqrt(area/math.pi))#note that we store the diameters here because later /2 is applied
    #now put centres together:
    logger.print(len(centresA),len(centresB))
    centres = centresA[::-1]+centresB[1::]
    radii=  radiiA[::-1]+radiiB[1::]
    

    if used_smaller_steps:
        centres,radii= fill_interpolate_path(centres,radii) #TODO find which flips the order
    if pairwise_distance_sum(centres)>=1.5*pairwise_distance_sum(cp):
        ...#assert False,"recutting mesh due to low centerline information?"
    #centres,radii= reorder_and_filter_by_closest_points(centres,radii) 
    return centres, radii,rings#,used_normals,centresB,centresA,side_A,side_B

def pairwise_distance_sum(points):
    sum=0
    for i in range(len(points)):
        if i+1<len(points):
            sum+=abs(np.linalg.norm(np.asarray(points[i])-np.asarray(points[i+1])))
    return sum
def extract_diameter_and_centre_from_cut(centerline_points, mesh, step_size=1, window_size=5):
    radii = []
    centres = [] 
    rings=[]
    ref_indices=[]
    first_save=centerline_points[0]
    last_save=centerline_points[-1]
    centerline_points=centerline_points[1:-1]
    i = 0
    centerline_points=filter_centerline_by_distance(centerline_points,2)
    position_vector = centerline_points[i].copy()
    first=True 
    normal_vector = None
    cut=True
    while(cut):
        logger.print(i,len(centerline_points))
        #i = change_i_on_closeness(position_vector,centerline_points,i)
        if i+1<len(centerline_points) and threshold_close(position_vector,centerline_points[i+1],1+0.5):
            i+=1
        # Define the current plane using the position vector and normal vector
        prev_normal=normal_vector
        if first:
            position_vector = np.array(centerline_points[i]) 
        else:
            #move the plane along the shape
            if prev_normal is not None:
                normal_vector=(normal_vector+prev_normal)/2
                normal_vector = normal_vector / np.linalg.norm(normal_vector)
            position_vector = position_vector + step_size * normal_vector
            
        first=False
        if len(centres)<2:
            normal_vector = compute_normal(centerline_points,i)
        else: 
            normal_vector==compute_normal_from_previous_centres(centres)
        
        #create a cut and minimize area:
        r=minimize_cut_area(mesh, initial_normal=normal_vector, position_vector=position_vector)
        if r==np.inf:
            break
        xc, yc, area,ring,mod_normal = r
        rings.append(ring)
        c_center=translate_2d_to_3d([[xc, yc]], position_vector, mod_normal)[0] 
        centres.append(c_center) 
        ref_indices.append(i)
        radii.append(math.sqrt(area/math.pi)/2)
    
    is_inside = mesh.contains(centres)
    assert is_inside[0]
    #for every point thats not inside, instead put in the point from the guiding points with the last valid radius
    last_ref=-1
    for i, val in enumerate(is_inside):
        if not val:
            if ref_indices[i]!=last_ref:
                refpoint=centerline_points[ref_indices[i]]
                last_ref=ref_indices[i]
                centres[i]=refpoint
                radii[i]=radii[i-1]#last valid point
            else:
                del centres[i]
                del radii[i]
    #centres,radii= fill_interpolate_path(centres,radii) #TO DO find which flips the order
    #centres,radii= reorder_and_filter_by_closest_points(centres,radii)
    centres.insert(0,first_save)
    centres.append(last_save)
    return centres, radii,rings
def fill_interpolate_path(centres, radii, steplength=1):
    assert len(centres) == len(radii), "Centres and radii must be lists of the same length."

    centres = np.array(centres)
    radii = np.array(radii)

    interpolated_centres = [centres[0]]
    interpolated_radii = [radii[0]]
    current_position = centres[0].copy()
    current_radius = radii[0]

    for i in range(1, len(centres)):
        next_center = centres[i]
        next_radius = radii[i]
        while True:
            to_next_vector = next_center - current_position
            to_next_length = np.linalg.norm(to_next_vector)
            if to_next_length < 1e-6:  # Avoid division by zero 
                break
            if to_next_length >= steplength:
                direction = to_next_vector / to_next_length
                current_position += direction * steplength
                # Calculate t based on the original segment from i-1 to i
                segment_vector = centres[i] - centres[i-1]
                segment_length = np.linalg.norm(segment_vector)
                if segment_length < 1e-6:
                    t = 1.0  # Avoid division by zero
                else:
                    t = np.linalg.norm(current_position - centres[i-1]) / segment_length
                current_radius = radii[i-1] + t * (radii[i] - radii[i-1])
                interpolated_centres.append(current_position.copy())
                interpolated_radii.append(current_radius)
            else:
                break

    
    if not np.allclose(interpolated_centres[-1], centres[-1]):
         interpolated_centres.append(centres[-1])
         interpolated_radii.append(radii[-1])

    return interpolated_centres, interpolated_radii
import math
def reorder_and_filter_by_closest_points(centres, radii):
    """
    Reorder the centres and radii by selecting the closest points consecutively starting from the first point.
    If a closer point is found before checking all points, the intermediate point(s) are skipped.

    :param centres: List of 3D points representing path centers.
    :param radii: List of radius values corresponding to each center.
    :return: Tuple of filtered and reordered centres and radii.
    """

    def distance(p1, p2):
        return np.linalg.norm(p1 - p2)

    # Initial setup with the first point
    remaining_points = list(zip(centres, radii))
    ordered_centres = [centres[0]]
    ordered_radii = [radii[0]]
    remaining_points.pop(0)  # Remove the first point since it's now in ordered lists

    while remaining_points:
        # Calculate distances from the last point in the ordered list to all remaining points
        last_point = ordered_centres[-1]
        distances = [distance(last_point, centre) for centre, _ in remaining_points]
        
        # Find the index of the closest point
        min_index = np.argmin(distances)
        
        # Retrieve and append the closest point and its radius to our ordered lists
        closest_centre, closest_radius = remaining_points.pop(min_index)
        ordered_centres.append(closest_centre)
        ordered_radii.append(closest_radius)

    return ordered_centres, ordered_radii
def process_single_submesh(trimesh_object,integer):
    assert trimesh_object.is_watertight, "One Submesh is not watertight, aborting"
    cl=extract_centerline(trimesh_object)
    centerline_points = extract_centerline(trimesh_object)
    cl, radii, rings = extract_diameter_and_centre_from_cut_two_sided(cl,trimesh_object)
    return create_vessel(trimesh_object,cl,radii,trimesh_object.metadata["file_name"],del_submesh=True,integer=integer)


def create_vessel(
    submesh,
    path,
    radii,
    name,
    del_submesh=False,
    integer=-1
):
    #function to create a vessel 
    diameters=np.asarray(radii)*2
    vessel_path = path 
    # combine diameters over a window of 10percent length to smoothen out errors
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
    insertions = []

    vessel = Vessel(
        vessel_path,
        path_diameter_a,
        path_diameter_b,
        avg_diameter,
        diameters,
        name,
        standard_speed_function,
    )
    vessel.integer=integer
    vessel.volume = submesh.volume
    vessel.submesh = None
    if not del_submesh:
        vessel.submesh = submesh
    return vessel