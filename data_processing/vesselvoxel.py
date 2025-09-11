# package for layered vesselvoxel extraction

import numpy as np
import math
from collections import deque


def generate_layered_skeleton(points, diameters, pitch=1):
    skeleton = set()
    layers = set()  # Original layers as a set

    def voxelize_line(A, B, pitch):
        scaled_A = np.array(A) / pitch
        scaled_B = np.array(B) / pitch

        direction = scaled_B - scaled_A
        current_voxel = np.floor(scaled_A).astype(int)
        end_voxel = np.floor(scaled_B).astype(int)

        step = np.zeros(3, dtype=int)
        tMax = np.zeros(3)
        tDelta = np.zeros(3)

        for i in range(3):
            if direction[i] > 0:
                step[i] = 1
            elif direction[i] < 0:
                step[i] = -1
            else:
                step[i] = 0

            if step[i] != 0:
                voxel_boundary = current_voxel[i] + (step[i] > 0)
                tMax[i] = (
                    (voxel_boundary - scaled_A[i]) / direction[i]
                    if direction[i] != 0
                    else float("inf")
                )
                tDelta[i] = (
                    1.0 / abs(direction[i]) if direction[i] != 0 else float("inf")
                )
            else:
                tMax[i] = float("inf")
                tDelta[i] = float("inf")

        voxels = set()
        voxels.add(tuple(current_voxel))

        while True:
            if tuple(current_voxel) == tuple(end_voxel):
                break

            min_axis = np.argmin(tMax)
            current_voxel[min_axis] += step[min_axis]
            tMax[min_axis] += tDelta[min_axis]

            voxels.add(tuple(current_voxel))

        return voxels

    def closest_point_on_segment(P, A, B):
        AP = P - A
        AB = B - A
        length_sq = np.dot(AB, AB)
        if length_sq == 0:
            return A, 0.0
        t = np.dot(AP, AB) / length_sq
        t_clamped = np.clip(t, 0.0, 1.0)
        closest = A + t_clamped * AB
        return closest, t_clamped

    def voxelize_layers(A, B, dA, dB, pitch, skeleton):
        A = np.array(A)
        B = np.array(B)
        delta = B - A
        length = np.linalg.norm(delta)
        if length == 0:
            return set()
        direction = delta / length
        max_radius = max(dA, dB) / 2.0

        min_bound = np.minimum(A, B) - max_radius
        max_bound = np.maximum(A, B) + max_radius

        min_voxel = np.floor(min_bound / pitch).astype(int)
        max_voxel = np.ceil(max_bound / pitch).astype(int)

        x = np.arange(min_voxel[0], max_voxel[0] + 1)
        y = np.arange(min_voxel[1], max_voxel[1] + 1)
        z = np.arange(min_voxel[2], max_voxel[2] + 1)

        grid = np.array(np.meshgrid(x, y, z, indexing="ij")).T.reshape(-1, 3)
        if grid.size == 0:
            return set()

        centers = grid * pitch

        AP = centers - A
        t = np.dot(AP, direction) / length
        t_clamped = np.clip(t, 0.0, 1.0)

        closest_points = A + np.outer(t_clamped, delta)
        distances = np.linalg.norm(centers - closest_points, axis=1)
        radii = (dA * (1 - t_clamped) + dB * t_clamped) / 2.0

        mask = distances <= radii
        candidate_voxels = [tuple(voxel) for voxel in grid[mask]]

        layer_voxels = {voxel for voxel in candidate_voxels if voxel not in skeleton}
        return layer_voxels

    # Generate skeleton voxels
    for i in range(len(points) - 1):
        A = points[i]
        B = points[i + 1]
        line_voxels = voxelize_line(A, B, pitch)
        skeleton.update(line_voxels)

    # Generate layer voxels
    for i in range(len(points) - 1):
        A = points[i]
        B = points[i + 1]
        dA = diameters[i]
        dB = diameters[i + 1]
        layer_voxels = voxelize_layers(A, B, dA, dB, pitch, skeleton)
        layers.update(layer_voxels)

    # Split layers into layers based on distance from skeleton using BFS
    distance = {voxel: 0 for voxel in skeleton}
    queue = deque(skeleton)

    # Perform BFS to compute distances from skeleton
    while queue:
        current = queue.popleft()
        current_dist = distance[current]
        # Check all 26 neighboring voxels
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue  # Skip the current voxel itself
                    neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                    if neighbor in layers and neighbor not in distance:
                        distance[neighbor] = current_dist + 1
                        queue.append(neighbor)

    # After BFS and before converting layers to a list, save the original set
    original_layers_set = layers.copy()

    # Organize layer voxels into layers based on their distance
    layers_dict = {}
    for voxel, d in distance.items():
        if voxel in layers:  # Ensure we only process layer voxels
            layer = d - 1  # Subtract 1 because skeleton is at distance 0
            if layer not in layers_dict:
                layers_dict[layer] = set()
            layers_dict[layer].add(voxel)

    # Determine the maximum layer index
    max_layer = max(layers_dict.keys()) if layers_dict else -1

    # Create the layers list, ensuring consecutive layers even if some are empty
    layers_list = []
    for i in range(max_layer + 1):
        layers_list.append(layers_dict.get(i, set()))

    # Check for any layer voxels that couldn't be reached
    unreachable = original_layers_set - set(distance.keys())
    if unreachable:
        print(
            f"Warning: {len(unreachable)} layer voxels are unreachable from the skeleton and are excluded from layers."
        )

    return {"skeleton": skeleton, "layers": layers_list}


import numpy as np
import trimesh
import numpy as np
import trimesh
import numpy as np
from scipy.sparse import dok_matrix

import numpy as np
from collections import defaultdict


def create_sparse_skeleton(skeleton_data, pitch, global_bounds):
    origin = np.array(global_bounds[0])
    span = np.array(global_bounds[1]) - origin
    shape = np.ceil(span / pitch).astype(int)

    sparse_voxels = set()

    for skel in skeleton_data:
        if not isinstance(skel, dict):
            continue

        # Safely collect points
        all_points = []
        if "skeleton" in skel and len(skel["skeleton"]) > 0:
            all_points.extend(skel["skeleton"])
        if "layers" in skel:
            for layer in skel["layers"]:
                if len(layer) > 0:
                    all_points.extend(layer)

        # Skip empty skeletons
        if len(all_points) == 0:
            continue  # <-- CRITICAL FIX

        points = np.array(all_points)
        indices = ((points - origin) / pitch).astype(int)
        valid = np.all((indices >= 0) & (indices < shape), axis=1)

        for idx in indices[valid]:
            sparse_voxels.add(tuple(idx))

    return sparse_voxels, origin, pitch, shape


def subtract_sparse_voxels(mesh_vox, sparse_skel, origin, pitch):
    """Subtract using 3D sparse set"""
    # Convert mesh points to indices
    mesh_indices = ((mesh_vox.points - origin) / pitch).astype(int)

    # Create mask of voxels not in sparse_skel
    keep_mask = [tuple(idx) not in sparse_skel for idx in mesh_indices]

    return mesh_vox.points[keep_mask]


def subtract_sparse_voxels(mesh_vox, sparse_skel, origin, pitch):
    """Subtract sparse skeleton from mesh voxels"""
    # Convert mesh voxels to indices
    mesh_indices = ((mesh_vox.points - origin) / pitch).astype(int)

    # Create mask of voxels to keep
    keep_mask = np.ones(len(mesh_vox.points), dtype=bool)

    # Check each mesh voxel against sparse skeleton
    for i, idx in enumerate(mesh_indices):
        if tuple(idx) in sparse_skel:
            keep_mask[i] = False

    # Apply mask
    remaining = mesh_vox.points[keep_mask]
    return remaining


def process_mesh_memopt(filepath):
    mesh = trimesh.load(filepath)
    if not isinstance(mesh, trimesh.Trimesh):
        return None

    # Voxelize mesh
    mesh_vox = mesh.voxelized(1).fill()

    # Subtract skeleton
    remaining = subtract_sparse_voxels(mesh_vox, sparse_skel, skel_origin, skel_pitch)

    # Return results
    if len(remaining) > 0:
        return (filepath, remaining, len(remaining), None)
    return None


def build_skeleton_lookup(skeleton_list, pitch):
    """Create a global lookup dictionary of skeleton voxels"""
    lookup = {}
    for skel in skeleton_list:
        combined = set(skel["skeleton"])
        for layer in skel["layers"]:
            combined.update(layer)

        for voxel in combined:
            world_pos = tuple(np.array(voxel) * pitch)
            lookup[world_pos] = True  # Store as exact positions
    return lookup


import numpy as np


class CombinedObject:
    def __init__(self, path, diameters, orig):
        self.path = path
        self.diameters = diameters
        self.orig = orig


def reconnect(vessels):
    """Reconnect vessel segments into continuous paths minimizing end-to-start distance, allowing segment reversal."""
    output = []
    visited = set()

    # Build a bidirectional adjacency list based on links
    adjacency = {vessel: set() for vessel in vessels}
    for vessel in vessels:
        for link in vessel.links_to_path:
            target = link.target_vessel
            adjacency[vessel].add(target)
            adjacency[target].add(vessel)  # Ensure bidirectional connection

    def _traverse(current_vessel, current_group):
        if current_vessel in visited:
            return
        visited.add(current_vessel)
        current_group.append(current_vessel)
        for neighbor in adjacency[current_vessel]:
            if neighbor not in visited and neighbor.orig == current_vessel.orig:
                _traverse(neighbor, current_group)

    def reorder_group(group):
        """Reorder vessels to minimize connection distances, allowing reversal."""
        if len(group) <= 1:
            return group, [False] * len(group) if group else []

        best_order = None
        best_reverse = None
        min_total = float("inf")

        # Try each vessel as the starting point
        for start_vessel in group:
            remaining = [v for v in group if v != start_vessel]
            current_order = [start_vessel]
            current_reverse = [False]
            current_end = start_vessel.path[-1]
            total_distance = 0
            temp_remaining = remaining.copy()

            while temp_remaining:
                closest_dist = float("inf")
                closest_vessel = None
                closest_rev = False

                for vessel in temp_remaining:
                    # Distance to vessel's start
                    dist_start = np.linalg.norm(current_end - vessel.path[0])
                    # Distance to vessel's end (requires reversing)
                    dist_end = np.linalg.norm(current_end - vessel.path[-1])

                    if dist_start < closest_dist:
                        closest_dist = dist_start
                        closest_vessel = vessel
                        closest_rev = False
                    if dist_end < closest_dist:
                        closest_dist = dist_end
                        closest_vessel = vessel
                        closest_rev = True

                if closest_vessel is None:
                    break  # No connection found (shouldn't happen)

                total_distance += closest_dist
                current_order.append(closest_vessel)
                current_reverse.append(closest_rev)
                current_end = (
                    closest_vessel.path[0] if closest_rev else closest_vessel.path[-1]
                )
                temp_remaining.remove(closest_vessel)

            # Check if this starting vessel gives a better total distance
            if total_distance < min_total:
                min_total = total_distance
                best_order = current_order
                best_reverse = current_reverse

        return best_order, best_reverse

    for vessel in vessels:
        if vessel not in visited:
            group = []
            _traverse(vessel, group)
            if group:
                # Reorder the group optimally
                ordered_group, reverse_flags = reorder_group(group)

                # Combine paths with reversals as needed
                combined_path = []
                combined_diameters = []
                for v, rev in zip(ordered_group, reverse_flags):
                    if rev:
                        combined_path.append(np.flipud(v.path))
                        combined_diameters.append(np.flip(v.diameters))
                    else:
                        combined_path.append(v.path)
                        combined_diameters.append(v.diameters)

                # Handle edge case where combined_path might be empty
                if not combined_path:
                    continue

                combined_obj = CombinedObject(
                    path=np.vstack(combined_path),
                    diameters=np.concatenate(combined_diameters),
                    orig=group[0].orig,  # All in group share same orig
                )
                output.append(combined_obj)

    return output
