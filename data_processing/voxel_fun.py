import numpy as np
import trimesh
import math


def convert_to_trimesh_voxelgrid(result, voxel_size):
    """
    convert numpy boolean array to voxelgrid in trimesh

    :param result: Boolean NumPy-Array, representing voxelgrid
    :param voxel_size: set size of voxel
    :return: trimesh.voxel.VoxelGrid
    """
    assert len(result.shape) == 3, "Use a 3 dimensional array!"
    # find filled voxels
    filled = np.argwhere(result)

    # create voxelgrid with filled voxels where array is true
    voxel_grid = trimesh.voxel.VoxelGrid(
        encoding=trimesh.voxel.encoding.DenseEncoding(data=result.astype(bool))
    )

    return voxel_grid


def get_adjacent_indices(voxel_index, grid_shape):
    """
    Returns 26 neighbouring indices

    :param voxel_index: tuple or list: (i, j, k), 3d index of voxel in grid
    :param grid_shape: shape of grid: (nx, ny, nz)
    :return: list of neighbouring indices
    """

    i, j, k = voxel_index
    nx, ny, nz = grid_shape

    # list of offsets
    offsets = [
        (di, dj, dk)
        for di in [-1, 0, 1]
        for dj in [-1, 0, 1]
        for dk in [-1, 0, 1]
        if not (di == 0 and dj == 0 and dk == 0)
    ]

    neighbors = []
    for offset in offsets:
        ni, nj, nk = i + offset[0], j + offset[1], k + offset[2]
        # check neighbors in grid
        if (0 <= ni < nx) and (0 <= nj < ny) and (0 <= nk < nz):
            neighbors.append((ni, nj, nk))

    return neighbors


# numba CUDA kernel for distance computation
from numba import jit, cuda

# Check for available GPU backend
has_cuda = False
has_rocm = False

try:
    from numba import cuda

    has_cuda = True
except ImportError:
    pass

if not has_cuda:
    try:
        from numba import roc

        has_rocm = True
    except ImportError:
        pass

if not (has_cuda or has_rocm):
    raise ImportError("No GPU support available. Need either CUDA or ROCm.")

# Alias the GPU module
gpu = cuda if has_cuda else roc

# Conditional kernel definitions
if has_cuda:
    # CUDA version with 3D grid
    @gpu.jit
    def init_outer_distances(grid, distances):
        x, y, z = gpu.grid(3)
        nx, ny, nz = grid.shape

        if x < nx and y < ny and z < nz:
            is_outer = False
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if not (dx == 0 and dy == 0 and dz == 0):
                            ni, nj, nk = x + dx, y + dy, z + dz
                            if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                                if abs(dx) + abs(dy) + abs(dz) <= 1:
                                    if not grid[ni, nj, nk]:
                                        is_outer = True
                            else:
                                is_outer = True

            if is_outer:
                distances[x, y, z] = 1

    @gpu.jit
    def assign_outer_distances(grid, distances, changed):
        x, y, z = gpu.grid(3)
        nx, ny, nz = grid.shape
        if distances[x, y, z] != 0:
            return
        min_near_distance = 0

        if x < nx and y < ny and z < nz:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if not (dx == 0 and dy == 0 and dz == 0):
                            ni, nj, nk = x + dx, y + dy, z + dz
                            if (
                                0 <= ni < nx
                                and 0 <= nj < ny
                                and 0 <= nk < nz
                                and abs(dx) + abs(dy) + abs(dz) <= 1
                            ):
                                if distances[ni, nj, nk] != 0:
                                    if distances[ni, nj, nk] > min_near_distance:
                                        min_near_distance = distances[ni, nj, nk]

            if min_near_distance > 0:
                distances[x, y, z] = min_near_distance + 1
                changed[0] = 1

else:  # ROCm version

    @gpu.jit
    def init_outer_distances(grid, distances):
        idx = gpu.get_global_id(0)
        nx, ny, nz = grid.shape
        total_voxels = nx * ny * nz

        if idx >= total_voxels:
            return

        x = idx // (ny * nz)
        remainder = idx % (ny * nz)
        y = remainder // nz
        z = remainder % nz

        is_outer = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if not (dx == 0 and dy == 0 and dz == 0):
                        ni, nj, nk = x + dx, y + dy, z + dz
                        if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                            if abs(dx) + abs(dy) + abs(dz) <= 1:
                                if not grid[ni, nj, nk]:
                                    is_outer = True
                        else:
                            is_outer = True

        if is_outer:
            distances[x, y, z] = 1

    @gpu.jit
    def assign_outer_distances(grid, distances, changed):
        idx = gpu.get_global_id(0)
        nx, ny, nz = grid.shape
        total_voxels = nx * ny * nz

        if idx >= total_voxels:
            return

        x = idx // (ny * nz)
        remainder = idx % (ny * nz)
        y = remainder // nz
        z = remainder % nz

        if distances[x, y, z] != 0:
            return

        min_near_distance = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if not (dx == 0 and dy == 0 and dz == 0):
                        ni, nj, nk = x + dx, y + dy, z + dz
                        if (
                            0 <= ni < nx
                            and 0 <= nj < ny
                            and 0 <= nk < nz
                            and abs(dx) + abs(dy) + abs(dz) <= 1
                        ):
                            if distances[ni, nj, nk] != 0:
                                if distances[ni, nj, nk] > min_near_distance:
                                    min_near_distance = distances[ni, nj, nk]

        if min_near_distance > 0:
            distances[x, y, z] = min_near_distance + 1
            changed[0] = 1


def get_voxel_distances(grid, max_iterations=1, db=True):
    nx, ny, nz = grid.shape

    # GPU memory management
    d_grid = gpu.to_device(grid)
    d_distances = gpu.device_array_like(np.full(grid.shape, 0, dtype=np.int32))
    host_changed = np.array([1], dtype=np.int32)
    d_changed = gpu.to_device(host_changed)

    # Configure launch parameters
    if has_cuda:
        threads_per_block = (8, 8, 8)
        blocks_per_grid = (
            math.ceil(nx / threads_per_block[0]),
            math.ceil(ny / threads_per_block[1]),
            math.ceil(nz / threads_per_block[2]),
        )
    else:
        total_voxels = nx * ny * nz
        threads_per_block = 256
        blocks_per_grid = (total_voxels + threads_per_block - 1) // threads_per_block

    # Initial distance marking
    init_outer_distances[blocks_per_grid, threads_per_block](d_grid, d_distances)

    # Iterative processing
    while True:
        # Check termination condition
        host_changed = d_changed.copy_to_host()
        if host_changed[0] != 1:
            break

        # Reset changed flag
        host_changed[0] = 0
        d_changed.copy_to_device(host_changed)

        # Run distance assignment
        assign_outer_distances[blocks_per_grid, threads_per_block](
            d_grid, d_distances, d_changed
        )

        # Debug output
        if db:
            print("Distances after change:", d_distances.copy_to_host())
        print("Changed:", d_changed.copy_to_host()[0])

    return d_distances.copy_to_host()


class Graph:
    # a simple graph implementation using a dictionary with its keys for nodes and values for edgesets to other nodes.
    # used to find out which voxels together create a connected centerline
    def __init__(self):
        self.node_to_edges = dict()

    def add_edge(self, edge):
        n1, n2 = edge
        if n1 not in self.node_to_edges:
            self.node_to_edges[n1] = {n2}
        else:
            self.node_to_edges[n1].add(n2)

        if n2 not in self.node_to_edges:
            self.node_to_edges[n2] = {n1}
        else:
            self.node_to_edges[n2].add(n1)

    def half(self):
        """Remove every other node starting from a leaf node where possible."""

        # First pass to determine the nodes to be removed
        nodes_to_remove = set()
        for node in self.node_to_edges:
            if len(self.node_to_edges[node]) == 1:
                leaf = node
                break
        assert leaf != None, "A leaf node is needed for connectivity based halfing"

        # iterate over nodes that are connected, alternating between marking them for deletion
        unvisited = set(self.node_to_edges.keys())
        curr_nodes = set([leaf])
        even = False

        while unvisited:
            next_nodes = set()
            for curr_node in curr_nodes:
                unvisited.remove(curr_node)
                if len(self.node_to_edges[curr_node]) == 2 and even:
                    logger.print("mark for del")
                    nodes_to_remove.add(curr_node)
                even = not even
                # get next nodes
                for other_node in self.node_to_edges[curr_node]:
                    if other_node in unvisited and other_node not in curr_nodes:
                        next_nodes.add(other_node)
            curr_nodes = next_nodes

        # Second pass to process the removal
        for node in nodes_to_remove:
            if node in self.node_to_edges and len(self.node_to_edges[node]) == 2:
                neighbours = list(self.node_to_edges[node])
                n1, n2 = neighbours

                # Update neighbours' sets
                self.node_to_edges[n1].remove(node)
                self.node_to_edges[n2].remove(node)

                # Ensure neighbours are directly connected if not already
                if n2 not in self.node_to_edges[n1]:
                    self.node_to_edges[n1].add(n2)
                if n1 not in self.node_to_edges[n2]:
                    self.node_to_edges[n2].add(n1)

                # Delete the node
                del self.node_to_edges[node]
                logger.print("Removed node:", node)


def longest_path_from_node(node, graph, prior=[], pathlength=0):
    pathlengths = []
    # logger.print(node)
    for edge in graph.node_to_edges[node]:
        if edge not in prior:
            pathlengths.append(
                longest_path_from_node(
                    edge, graph, prior=[node] + prior, pathlength=pathlength + 1
                )
            )
    if all(edge in prior for edge in graph.node_to_edges[node]):
        return pathlength
    return max(pathlengths + [0])
