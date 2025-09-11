import numpy as np
import trimesh
import os
import sys

module_path = os.path.abspath(os.path.join("./../.."))
if module_path not in sys.path:
    sys.path.append(module_path)
from CeFloPS.simulation.common.functions import single_dir_vector
from .visualization import pick_color, show


def show_chances_3d(
    cell_pos, vectors, show_path=False, show_chance=True, show_merged=False
):  # TODO call by reference? WICHTIG MEMORY
    cell_pos = np.asarray(cell_pos)
    vectors = np.asarray(vectors)
    stepchances = np.asarray(
        [
            sum([vector[0] for vector in vectors if vector[0] >= 0] + [0]),
            sum([vector[0] for vector in vectors if vector[0] <= 0] + [0]),
            sum([vector[1] for vector in vectors if vector[1] >= 0] + [0]),
            sum([vector[1] for vector in vectors if vector[1] <= 0] + [0]),
            sum([vector[2] for vector in vectors if vector[2] >= 0] + [0]),
            sum([vector[2] for vector in vectors if vector[2] <= 0] + [0]),
        ]
    )
    pathes = [trimesh.load_path([cell_pos, vector]) for vector in vectors]
    for i, path in enumerate(pathes):
        path.colors = [pick_color(i % 20) + [100]]
    chance_pathes = []
    chance_pathes_merged = []

    """ print([cell_pos,np.append(stepchances[0], cell_pos[1::])])
    print([cell_pos,np.append(stepchances[1], cell_pos[1::])])

    print([cell_pos,np.append(cell_pos[0:1],np.append(stepchances[2] ,cell_pos[2::]))])
    print([cell_pos,np.append(cell_pos[0:1],np.append(stepchances[3],cell_pos[2::]))])

    print([cell_pos,np.append(cell_pos[0:2],stepchances[4])])
    print([cell_pos,np.append(cell_pos[0:2],stepchances[5])])"""

    if sum([vector[0] for vector in vectors] + [0]) != 0:
        chance_pathes_merged.append(
            trimesh.load_path(
                [
                    cell_pos,
                    single_dir_vector(sum([vector[0] for vector in vectors]), 0, 3),
                ]
            )
        )
    if sum([vector[1] for vector in vectors] + [0]) != 0:
        chance_pathes_merged.append(
            trimesh.load_path(
                [
                    cell_pos,
                    single_dir_vector(sum([vector[1] for vector in vectors]), 1, 3),
                ]
            )
        )
    if sum([vector[2] for vector in vectors] + [0]) != 0:
        chance_pathes_merged.append(
            trimesh.load_path(
                [
                    cell_pos,
                    single_dir_vector(sum([vector[2] for vector in vectors]), 2, 3),
                ]
            )
        )

    # color parts that make chancevectors up:
    direction_vectors = np.asarray([cell_pos for i in range(0, 6)])
    for v, vector in enumerate(vectors):
        for i, axis in enumerate(vector):
            # change origin and add colored path according to vector input
            if axis < 0:
                path = trimesh.load_path(
                    [
                        direction_vectors[i],
                        direction_vectors[i] + single_dir_vector(axis, i, 3),
                    ]
                )
                path.colors = [pick_color(v % 20) + [100]]
                chance_pathes.append(path)
                direction_vectors[i] = direction_vectors[i] + single_dir_vector(
                    axis, i, 3
                )  # x,y,z negative

            elif axis > 0:
                path = trimesh.load_path(
                    [
                        direction_vectors[i + 3],
                        direction_vectors[i + 3] + single_dir_vector(axis, i, 3),
                    ]
                )
                path.colors = [pick_color(v % 20) + [100]]
                chance_pathes.append(path)
                direction_vectors[i + 3] = direction_vectors[i + 3] + single_dir_vector(
                    axis, i, 3
                )  # x,y,z positive
    """ 
    print(pathes)
    print(chance_pathes) 
    """
    if show_path:
        show(pathes)
    if show_chance:
        show(chance_pathes)
    if show_merged:
        show(chance_pathes_merged)
