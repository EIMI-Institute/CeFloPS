import math
import itertools
import pickle
import scipy
from scipy.spatial.distance import cdist, pdist
import trimesh
import os

from CeFloPS.data_processing.voxel_fun import get_adjacent_indices
from skimage.morphology import skeletonize
import networkx as nx

# import networkx as nx
import numpy as np
from .functions import (
    calculate_distance_euclidian,
    moving_average,
    condensed_to_square,
    mag,
    point_in,
    norm_vector,
    dot_product,
    count,
)
from .vessel2 import Vessel
from .vessel2 import Link
import hashlib

from joblib import Parallel, delayed
import CeFloPS.simulation.settings as settings
import sympy
from CeFloPS.data_processing.geometric_fun import (
    get_area_from_ring,
    tilt_vector,
    PI,
    get_center_ellipsis,
)
import logging
from CeFloPS.logger_config import setup_logger

logger = setup_logger(__name__, level=logging.INFO)


class Arteriol:
    def __init__(self, avg_diameter):
        self.avg_diameter = avg_diameter


def rround(x, unit):
    if x % unit > unit / 2:
        return x - (x % unit) + unit
    else:
        return x - (x % unit)


def is_in(x, arr):
    for p in arr:
        if p[0] == x[0] and p[1] == x[1] and p[2] == x[2]:
            return True
    return False


def filled(array, points, pitch):
    # array = set(array)

    centers_from_points = np.asarray(
        [
            [rround(point[0], pitch), rround(point[1], pitch), rround(point[2], pitch)]
            for point in points
        ]
    )
    a = []
    for i in range(len(array)):
        a.append(str(array[i]))
    b = []
    for i in range(len(centers_from_points)):
        b.append(str(centers_from_points[i]))
    return [point in a for point in b]


def get_points_in_roi(roipoints, vesselpoints, pitch):
    # returns indices of points that lie in roipoint array
    indices = []
    lays_in = filled(roipoints, vesselpoints, pitch)
    for i, b in enumerate(lays_in):
        if b:
            indices.append(i)
            logger.print("block_found", b)
    return indices


# vesselcreation functions
def set_rois_for_vessel(rois):
    """set_rois_for_vessel Connects rois and vessels, uses last points of vessels and checks if inside of mesh, if not take closest one.
    Connects all inner vessels to roi or at least the nearest vessel

    Args:
        rois (_type_): _description_
    """
    logger.print("Connecting ROIs...")

    VEINDISTANCE = 200
    blood_roi = [roi for roi in rois if "blood" in roi.name][0]
    vessels = blood_roi.geometry
    other_rois = [roi for roi in rois if "blood" not in roi.name]
    vesselends_artery = [
        vessel
        for vessel in vessels
        if vessel.type == "artery"
        and len(
            [
                link
                for link in vessel.links_to_path
                if link.source_index == len(vessel.path) - 1
            ]
        )
        == 0
    ]
    vesselendpoints_artery = [
        vessel.path[len(vessel.path) - 1]
        for vessel in vessels
        if vessel.type == "artery"
        and len(
            [
                link
                for link in vessel.links_to_path
                if link.source_index == len(vessel.path) - 1
            ]
        )
        == 0
    ]
    vesselends_vein = [
        vessel
        for vessel in vessels
        if vessel.type == "vein"
        and len([link for link in vessel.links_to_path if link.source_index == 0]) == 0
    ]
    vesselendpoints_vein = [
        vessel.path[len(vessel.path) - 1]
        for vessel in vessels
        if vessel.type == "vein"
        and len([link for link in vessel.links_to_path if link.source_index == 0]) == 0
    ]
    logger.print("vesselends #artery", len(vesselends_artery))
    logger.print("vesselends #vein", len(vesselends_vein))
    logger.print("vesselendpoints #artery", len(vesselendpoints_artery))
    logger.print("vesselendpoints #vein", len(vesselendpoints_vein))
    # logger.print("vesselnumber",len(vessels))
    # logger.print(blood_roi.geometry)
    with Parallel(n_jobs=-2) as para:
        vessel_register_roi_pairs = para(
            delayed(roiconnectionindices)(
                roiindex,
                roi,
                vesselendpoints_artery,
                vesselendpoints_vein,
                VEINDISTANCE,
            )
            for roiindex, roi in enumerate(other_rois)
        )
    for pairs in vessel_register_roi_pairs:
        for pair in pairs:
            if len(pair) == 2:
                v_indices = pair[0]
                roiindex = pair[1]
                for vesselindex in v_indices:
                    vesselends_artery[vesselindex].reachable_rois.append(
                        (
                            len(vesselends_artery[vesselindex].path) - 1,
                            other_rois[roiindex],
                        )
                    )
            if len(pair) == 3:
                roiindex = pair[1]
                v_indices = pair[0]
                roi = other_rois[roiindex]
                for index in v_indices:
                    # calculate closest entry index for roi and vein
                    closest_index = closest_point_in_path_index(
                        vesselends_vein[index], roi.center
                    )
                    roi.veins.append(
                        (
                            vesselendpoints_vein[index],
                            vesselends_vein[index],
                            closest_index,
                            vesselends_vein[index].path[closest_index],
                        )
                    )

    # ADD arteryends if they lie within the voxels of a roi. #3dlikes

    # ADD arteryends if they are a certain range to a artery that doesnt connect elsewhere? -> tubelikes

    changed = True
    while changed:
        changed = False
        for j, vessel in enumerate(vessels):
            before_len = len(vessel.reachable_rois)
            # assert that every vessel gets checked
            # logger.print("old", all([any([tupl == otupl for otupl in oldstate]) for tupl in oldstate]))
            for link in get_traversable_links(vessel):
                other_vessel = link.target_vessel
                for roi in other_vessel.get_rois(link.target_index):
                    if (link.source_index, roi) not in vessel.reachable_rois:
                        vessel.reachable_rois.append((link.source_index, roi))
                        vessel.reachable_rois = sorted(
                            list(set(vessel.reachable_rois)), key=lambda x: x[0]
                        )
                        changed = True
    if False:  # logger.print veinconnections
        for vessel in vessels:
            if len(vessel.reachable_rois) > 0:
                logger.print(
                    vessel.type,
                    vessel.associated_vesselname,
                    [roi.name for roi in vessel.get_rois(0)],
                )
                if "aorta" in vessel.associated_vesselname:
                    logger.print("---------------------------")
                if "jump_vessel" in vessel.associated_vesselname:
                    logger.print("ooooooooooooooooooooooooooooooooooooooooo")

    for roi in other_rois:
        roi.veins = sorted(
            roi.veins, key=lambda x: calculate_distance_euclidian(roi.center, x[3])
        )
    logger.print("Done!")


def consistent_hash(value):
    # convert in put to string
    value_str = str(value)
    # generate SHA-256 Hash
    hash_object = hashlib.sha256(value_str.encode("utf-8"))
    # convert hash to number
    hash_int = int(hash_object.hexdigest(), 16)

    return hash_int


def set_vessels_for_roi_single(roi, blood_roi):
    """set_rois_for_vessel Connects rois and vessels, uses last points of vessels and checks if inside of mesh, if not take closest one.
    Connects all inner vessels to roi or at least the nearest vessel

    Args:
        rois (_type_): _description_
    """
    logger.print("Connecting ROIs...")
    # check if roi connections are stored:
    vessel_hash = consistent_hash(settings.PATH_TO_VESSELS)
    voxeldistance = settings.ROI_VOXEL_PITCH
    roi_ident = consistent_hash(
        str(vessel_hash)
        + "_"
        + str(voxeldistance)
        + "_"
        + roi.name
        + "_"
        + str(settings.NEGATIVE_SPACES)
    )
    lookupname = "connections" + str(roi_ident) + ".cached_connections"
    # load connections if found
    found = True
    try:
        with open(
            settings.cache_dir + "/" + lookupname,
            "rb",
        ) as input_file:
            connections = pickle.load(input_file)
    except:
        logger.print("cached not found")
        found = False
    vessels = blood_roi.geometry
    other_rois = [roi]
    vesselends_artery = [
        vessel
        for vessel in vessels
        if vessel.type == "artery"
        and len(
            [
                link
                for link in vessel.links_to_path
                if link.source_index == len(vessel.path) - 1
            ]
        )
        == 0
    ]
    vesselendpoints_artery = [
        vessel.path[len(vessel.path) - 1]
        for vessel in vessels
        if vessel.type == "artery"
        and len(
            [
                link
                for link in vessel.links_to_path
                if link.source_index == len(vessel.path) - 1
            ]
        )
        == 0
    ]
    vesselends_vein = [
        vessel
        for vessel in vessels
        if vessel.type == "vein"
        and len([link for link in vessel.links_to_path if link.source_index == 0]) == 0
    ]
    vesselendpoints_vein = [
        vessel.path[len(vessel.path) - 1]
        for vessel in vessels
        if vessel.type == "vein"
        and len([link for link in vessel.links_to_path if link.source_index == 0]) == 0
    ]
    logger.print("ROI  ", roi.name)
    logger.print("vesselends #artery", len(vesselends_artery))
    logger.print("vesselends #vein", len(vesselends_vein))
    logger.print("vesselendpoints #artery", len(vesselendpoints_artery))
    logger.print("vesselendpoints #vein", len(vesselendpoints_vein))

    if not found:
        connections = [[], [], []]
        VEINDISTANCE = 200

        # logger.print("vesselnumber",len(vessels))
        # logger.print(blood_roi.geometry)
        with Parallel(n_jobs=-2) as para:
            vessel_register_roi_pairs = para(
                delayed(roiconnectionindices)(
                    roiindex,
                    roi,
                    vesselendpoints_artery,
                    vesselendpoints_vein,
                    VEINDISTANCE,
                )
                for roiindex, roi in enumerate(other_rois)
            )
        for pairs in vessel_register_roi_pairs:
            for pair in pairs:
                if len(pair) == 2:
                    v_indices = pair[0]
                    roiindex = pair[1]
                    for vesselindex in v_indices:
                        connections[0].append((vesselindex, roiindex))
                if len(pair) == 3:
                    roiindex = pair[1]
                    v_indices = pair[0]
                    roi = other_rois[roiindex]
                    for index in v_indices:
                        # calculate closest entry index for roi and vein
                        closest_index = closest_point_in_path_index(
                            vesselends_vein[index], roi.center
                        )

                        connections[1].append((closest_index, index, roiindex))

        # get entries
        for roi_index in enumerate(other_rois):
            break
            (
                vein_entries,
                artery_entries,
            ) = []  # either closest point in mesh or point in mesh
            connections[2].append((roi.vein_entries, roi.artery_entries, roi_index))

    # load from connections:
    roi.vein_entries = []  # voxel for flowwalk

    roi.artery_entries = []
    # store values in objects
    for i, data in enumerate(connections):
        # 0: vesselindex,roiindex
        # 1:closest_index,index,roi_index
        # 3: (artery_entries,vein_entries)
        if i == 0:
            for entry in data:
                vesselindex, roiindex = entry
                vesselends_artery[vesselindex].reachable_rois.append(
                    (
                        len(vesselends_artery[vesselindex].path) - 1,
                        other_rois[roiindex],
                    )
                )
        if i == 1:
            for entry in data:
                closest_index, index, roi_index = entry
                roi = other_rois[roiindex]
                roi.veins.append(
                    (
                        vesselendpoints_vein[index],
                        vesselends_vein[index],
                        closest_index,
                        vesselends_vein[index].path[closest_index],
                    )
                )

        if i == 2:
            for entry in data:
                vein_entries, artery_entries, roi_index = entry
                roi = other_rois[roi_index]
                roi.artery_entries = []
                roi.artery_entries = []
    # ADD arteryends if they lie within the voxels of a roi. #3dlikes

    # ADD arteryends if they are a certain range to a artery that doesnt connect elsewhere? -> tubelikes

    changed = True
    while changed:
        changed = False
        for j, vessel in enumerate(vessels):
            before_len = len(vessel.reachable_rois)
            # assert that every vessel gets checked
            # logger.print("old", all([any([tupl == otupl for otupl in oldstate]) for tupl in oldstate]))
            for link in get_traversable_links(vessel):
                other_vessel = link.target_vessel
                for roi in other_vessel.get_rois(link.target_index):
                    if (link.source_index, roi) not in vessel.reachable_rois:
                        vessel.reachable_rois.append((link.source_index, roi))
                        vessel.reachable_rois = sorted(
                            list(set(vessel.reachable_rois)), key=lambda x: x[0]
                        )
                        changed = True
    if False:  # logger.print veinconnections
        for vessel in vessels:
            if len(vessel.reachable_rois) > 0:
                logger.print(
                    vessel.type,
                    vessel.associated_vesselname,
                    [roi.name for roi in vessel.get_rois(0)],
                )
                if "aorta" in vessel.associated_vesselname:
                    logger.print("---------------------------")
                if "jump_vessel" in vessel.associated_vesselname:
                    logger.print("ooooooooooooooooooooooooooooooooooooooooo")

    for roi in other_rois:
        roi.veins = sorted(
            roi.veins, key=lambda x: calculate_distance_euclidian(roi.center, x[3])
        )

    if not found:
        with open(settings.cache_dir + "/" + lookupname, "wb") as handle:
            pickle.dump(connections, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.print("Done!")


def roiconnectionindices(
    roiindex, roi, vesselendpoints_artery, vesselendpoints_vein, VEINDISTANCE
):
    ret = []
    marked = False
    indices = get_points_in_roi(
        roi.geometry.get_points(), vesselendpoints_artery, settings.ROI_VOXEL_PITCH
    )
    if len(indices) > 0:
        marked = True
    if marked == False:
        logger.print([roi.center])
        logger.print()
        logger.print(vesselendpoints_artery)
        d = scipy.spatial.distance.cdist(
            [roi.center], vesselendpoints_artery, "euclidean"
        )
        indices.append(np.where(d == d.min())[1][0])  # closest one
    ret.append((indices, roiindex))
    # register veins
    """ logger.print(
        np.where(d <= VEINDISTANCE)[1],
        len(vesselends_artery),
        len(vesselendpoints_artery),
    ) """
    # get vesselends in roi
    indices = get_points_in_roi(
        roi.geometry.get_points(), vesselendpoints_vein, settings.ROI_VOXEL_PITCH
    )

    if len(indices) == 0:
        d = scipy.spatial.distance.cdist(
            [roi.center], vesselendpoints_vein, "euclidean"
        )
        [indices.append(i) for i in np.where(d <= VEINDISTANCE)[1]]

    ret.append((indices, roiindex, "vein"))

    return ret


def load_submeshes(meshname, save_folder=None):
    mesh = trimesh.load_mesh(meshname)
    submeshes = mesh.split()

    # check for watertightness
    for submesh in submeshes:
        assert submesh.is_watertight, "One Submesh is not watertight, aborting"
    ret = submeshes, [meshname + f"_{i}" for i in range(len(submeshes))]
    if save_folder != None:
        # save
        try:
            os.mkdir(save_folder)
        except OSError as error:
            logger.print(error)
        m = meshname.replace("\\", "/").split("/")[-1]
        logger.print(meshname, m)
        with open(f"{save_folder}/{m}_submeshes.pickle", "wb") as handle:
            pickle.dump(ret, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return ret


def calculate_min_area_cut(c, plane, area, submesh):
    # obtain minimal area ellipsis:
    min_area = area
    prior_normal = plane[1].copy()
    plane_position = plane[0]
    increment = 2
    changed = True
    blocked = set()
    while changed:
        changed = False
        pitched_normals = [
            tilt_vector(prior_normal, [increment, 0, 0]),
            tilt_vector(prior_normal, [-increment, 0, 0]),
            tilt_vector(prior_normal, [0, increment, 0]),
            tilt_vector(prior_normal, [0, -increment, 0]),
            tilt_vector(prior_normal, [0, 0, increment]),
            tilt_vector(prior_normal, [0, 0, -increment]),
        ]
        skip_next = False
        for i in range(len(pitched_normals)):
            if skip_next or i in blocked:
                # skip if this direction is tagged or the previous iterations reduced the area
                continue
            pitched_normal = pitched_normals[i]
            # get new cut
            cut = (
                trimesh.intersections.mesh_plane(
                    submesh,
                    pitched_normal,
                    plane_position,
                    return_faces=False,
                    local_faces=None,
                    cached_dots=None,
                ),
                plane_position,
            )
            cut = ([x for sub in cut[0] for x in sub], cut[1])
            ring = points_from_ring(cut[0], cut[1], False)
            # obtain ring and ellipsis
            if len(ring) != 0:
                try:
                    # logger.print("ringeln",len(ring))
                    new_area = get_area_from_ring(ring, plane_position, pitched_normal)
                    if new_area < min_area:
                        min_area = new_area
                        prior_normal = pitched_normal
                        changed = True
                        skip_next = True
                    else:
                        blocked.add(i)  # to reduce computations
                except Exception as e:
                    logger.print(e, ring)
                    # assert False
    # return min area & cut
    return cut, min_area


def calculate_ellipsis_diameter_cuts(submesh, cuts, plane_definitions, re_max=False):
    areas_from_ellipsis = []  # first iteration values for areas from ellipsis
    areas_from_max_diameter = []
    min_cuts = []
    for c, plane_def in zip(cuts, plane_definitions):
        # logger.print("plane_def", plane_def)
        # logger.print(len(c))
        ring = points_from_ring(c[0], c[1], False)
        areas_from_ellipsis.append(get_area_from_ring(ring, plane_def[0], plane_def[1]))
        if re_max:
            areas_from_max_diameter.append(
                (cdist(ring, ring, metric="euclidean").max() / 2) ** 2 * PI
            )

        minimal_a_cut, areas_from_ellipsis[-1] = calculate_min_area_cut(
            c, plane_def, areas_from_ellipsis[-1], submesh
        )
        min_cuts.append(minimal_a_cut)

    return min_cuts, areas_from_ellipsis


def p_submesh(submesh, path, normals, meshname, distance_between_pathpoints):
    cuts, plane_def = create_cuts(
        submesh, path, normals, meshname, distance_between_pathpoints
    )
    # logger.print(len(cuts[0]))
    # logger.print(len(cuts),len(plane_definitions))
    return calculate_ellipsis_diameter_cuts(submesh, cuts, plane_def), plane_def


from tqdm.auto import tqdm


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_info_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def process_submeshes(
    submeshes,
    meshnames,
    min_dinstance_search,
    distance_between_pathpoints,
    parallel=True,
    del_submeshes=False,
):
    """"""
    min_dinstance_search = min_dinstance_search
    distance_between_pathpoints = distance_between_pathpoints
    logger.print("Calculating Vessels for " + str(len(submeshes)) + " (sub)meshes.")
    with ProgressParallel(n_jobs=-2) as parallel:
        paths = parallel(
            delayed(stable_guidepoints_single)(
                submesh, 0, meshname, min_dinstance_search
            )
            for submesh, meshname in zip(submeshes, meshnames)  # if "heart" in meshname
        )
        # paths=center_paths(opaths,submeshes)
        logger.print("calculating normals")
        normals = parallel(
            delayed(get_normals_single)(path, distance_between_pathpoints)
            for path in paths  # if "heart" in meshname
        )
        logger.print("cutting submeshes")
        logger.print(type(paths), len(submeshes), len(paths))
        res = parallel(
            delayed(p_submesh)(
                submesh_name[0],
                paths[si],
                normals[si],
                submesh_name[1],
                distance_between_pathpoints,
            )
            for si, submesh_name in enumerate(
                zip(submeshes, meshnames)
            )  # if "heart" in meshname
        )
        cuts_in_mesh, areas_from_wriggled_ellipsis, plane_defs = [], [], []
        for r in res:
            cuts_in_mesh.append(r[0][0])
            areas_from_wriggled_ellipsis.append(r[0][1])
            plane_defs.append(r[1])

        diameter_ellipsis_as_circle = [
            [np.sqrt(np.asarray(areas_from_wriggled_ellipsis[i]) / PI) * 2][0]
            for i in range(len(areas_from_wriggled_ellipsis))
        ]
        logger.print("creating vessels")
        # TODO see why pathpoints are not always cnetered
        return parallel(
            delayed(create_vessel_single)(
                submesh_name[0],
                diameter_ellipsis_as_circle[si],
                cuts_in_mesh[si],
                submesh_name[1],
                del_submeshes,
                plane_defs[si][0],
                plane_defs[si][1],
            )
            for si, submesh_name in enumerate(
                zip(submeshes, meshnames)
            )  # if "heart" in meshname
        )


def process_mesh(meshname, min_dinstance_search=4.5, distance_between_pathpoints=1):
    """
    Function to load a single stl mesh and create vessels from it
    returns created vessel without links or direction
    """
    min_dinstance_search = min_dinstance_search
    distance_between_pathpoints = distance_between_pathpoints
    mesh = trimesh.load_mesh(meshname)
    submeshes = mesh.split()

    # check for watertightness
    for submesh in submeshes:
        assert submesh.is_watertight, "One Submesh is not watertight, aborting"
    logger.print("calculating paths", " for ", meshname)
    paths = create_guide_points(submeshes, 0, meshname, min_dinstance_search)
    # paths=center_paths(opaths,submeshes)
    logger.print("calculating normals" " for ", meshname)
    normals = get_normals(paths, submeshes, distance_between_pathpoints)
    logger.print("cutting submeshes" " for ", meshname)
    cuts_in_mesh = []
    plane_definitions = []
    areas_from_wriggled_ellipsis = []
    for si, submesh in enumerate(submeshes):
        cuts_in_mesh.append([])
        areas_from_wriggled_ellipsis.append([])
        cuts, plane_def = create_cuts(
            submesh, paths[si], normals[si], meshname, distance_between_pathpoints
        )
        logger.print(len(cuts[0]))
        plane_definitions.append(plane_def)
        areas_from_max_diameter = []
        cuts_from_wriggled_ellipsis = []
        min_areas = []
        areas_from_ellipsis = []
        areas_from_max_diameter = []
        for c, plane_def in zip(cuts, plane_definitions[si]):
            ring = points_from_ring(c[0], c[1], False)
            if ring == []:
                continue
            plane = plane_def
            plane_position = plane[0]
            areas_from_max_diameter.append(
                (cdist(ring, ring, metric="euclidean").max() / 2) ** 2 * PI
            )
            logger.print("ring", ring)
            areas_from_ellipsis.append(get_area_from_ring(ring, plane[0], plane[1]))

            # obtain minimal area ellipsis:
            min_area = areas_from_ellipsis[-1]
            prior_normal = plane[1].copy()
            increment = 2
            changed = True
            blocked = set()
            while changed:
                changed = False
                pitched_normals = [
                    tilt_vector(prior_normal, [increment, 0, 0]),
                    tilt_vector(prior_normal, [-increment, 0, 0]),
                    tilt_vector(prior_normal, [0, increment, 0]),
                    tilt_vector(prior_normal, [0, -increment, 0]),
                    tilt_vector(prior_normal, [0, 0, increment]),
                    tilt_vector(prior_normal, [0, 0, -increment]),
                ]
                skip_next = False
                for i in range(len(pitched_normals)):
                    if skip_next or i in blocked:
                        continue
                    pitched_normal = pitched_normals[i]
                    # get new cut
                    cut = (
                        trimesh.intersections.mesh_plane(
                            mesh,
                            pitched_normal,
                            plane_position,
                            return_faces=False,
                            local_faces=None,
                            cached_dots=None,
                        ),
                        plane_position,
                    )
                    cut = ([x for sub in cut[0] for x in sub], cut[1])
                    ring = points_from_ring(cut[0], cut[1], False)
                    # obtain ring and ellipsis
                    if len(ring) != 0:
                        new_area = get_area_from_ring(
                            ring, plane_position, pitched_normal
                        )
                        if new_area < min_area:
                            min_area = new_area
                            prior_normal = pitched_normal
                            changed = True
                            skip_next = True
                        else:
                            blocked.add(i)  # to reduce computations
            # save min area cut
            cuts_from_wriggled_ellipsis.append(cut)
            min_areas.append(min_area)
        cuts_in_mesh[si] = cuts_from_wriggled_ellipsis
        areas_from_wriggled_ellipsis[si] = min_areas
    diameter_ellipsis_as_circle = [
        [np.sqrt(np.asarray(areas_from_wriggled_ellipsis[i]) / PI) * 2][0]
        for i in range(len(areas_from_wriggled_ellipsis))
    ]

    logger.print("creating vessels" " for ", meshname)
    return create_vessels(
        cuts_in_mesh,
        meshname,
        submeshes,
        paths,
        diameter_ellipsis_as_circle,
        standard_speed_function,
        plane_definitions=plane_definitions,
    )


def compute_cuts_and_areas(
    submeshes, paths, normals, meshname, distance_between_pathpoints
):
    cuts_in_mesh = []
    plane_definitions = []
    areas_from_wriggled_ellipsis = []

    for si, submesh in enumerate(submeshes):
        cuts_in_mesh.append([])
        areas_from_wriggled_ellipsis.append([])

        cuts, plane_def = create_cuts(
            submesh, paths[si], normals[si], meshname, distance_between_pathpoints
        )
        plane_definitions.append(plane_def)

        cuts_from_wriggled_ellipsis, min_areas = compute_min_areas(
            submesh, cuts, plane_definitions[si]
        )

        cuts_in_mesh[si] = cuts_from_wriggled_ellipsis
        areas_from_wriggled_ellipsis[si] = min_areas

    return cuts_in_mesh, plane_definitions, areas_from_wriggled_ellipsis


def adaptive_plane_intersection(
    submesh, initial_plane_position, initial_plane_normal, min_points=5, step_size=0.1
):
    plane_position = initial_plane_position
    plane_normal = initial_plane_normal
    points = []

    while len(points) < min_points:
        # Intersect the mesh with the plane to get the intersection points
        cut = trimesh.intersections.mesh_plane(
            submesh,
            plane_normal,
            plane_position,
            return_faces=False,
        )

        # Flatten the result list and transform into a point ring
        points = [x for sub in cut for x in sub]

        if len(points) < min_points:
            # Move the plane slightly and try again
            plane_position += step_size * plane_normal

    return points


# Usage within your process
def compute_min_areas(submesh, cuts, plane_definitions, min_points=10):
    cuts_from_wriggled_ellipsis = []
    min_areas = []

    for c, plane_def in zip(cuts, plane_definitions):
        ring = points_from_ring(c[0], c[1], False)
        if not ring:
            # If ring is insufficient, use adaptive intersection
            initial_plane_position = plane_def[0]
            initial_plane_normal = plane_def[1]
            ring = adaptive_plane_intersection(
                submesh, initial_plane_position, initial_plane_normal, min_points
            )

        if not ring:
            continue  # Skip if even the adaptive method fails

        plane_position = plane_def[0]
        areas_from_ellipsis = [get_area_from_ring(ring, plane_position, plane_def[1])]
        min_area = areas_from_ellipsis[-1]

        cuts_from_wriggled_ellipsis.append((ring, plane_position))
        min_areas.append(min_area)

    return cuts_from_wriggled_ellipsis, min_areas


def process_individual_submesh(
    submesh,
    meshname,
    min_dinstance_search=4.5,
    distance_between_pathpoints=1,
    del_submesh=True,
):
    """
    Function to load a single stl mesh and create vessels from it
    returns created vessel without links or direction
    """
    submeshes = [submesh]

    # check for watertightness
    for submesh in submeshes:
        assert submesh.is_watertight, "One Submesh is not watertight, aborting"

    logger.print("calculating paths", " for ", meshname)
    paths = create_guide_points(submeshes, 0, meshname, min_dinstance_search)
    logger.print("calculating normals" " for ", meshname)
    normals = get_normals(paths, submeshes, distance_between_pathpoints)
    logger.print("cutting submeshes" " for ", meshname)

    cuts_in_mesh, plane_definitions, areas_from_wriggled_ellipsis = (
        compute_cuts_and_areas(
            submeshes, paths, normals, meshname, distance_between_pathpoints
        )
    )

    diameter_ellipsis_as_circle = [
        np.sqrt(np.asarray(areas) / np.pi) * 2 for areas in areas_from_wriggled_ellipsis
    ]

    logger.print("creating vessels" " for ", meshname)
    return create_vessels(
        cuts_in_mesh,
        meshname,
        submeshes,
        paths,
        diameter_ellipsis_as_circle,
        standard_speed_function,
        plane_definitions=plane_definitions,
        del_submesh=del_submesh,
    )[0]


def create_vessel_single(
    submesh,
    diameter_ellipsis_as_circle,
    cuts,
    name,
    del_submeshes=False,
    planes=None,
    demonstration=False,
):
    # main function to create a vessel

    diameters = list()
    if demonstration:
        selected_cuts = []
    # generate vesselpath (optionally diameters) using the cuts and centroids
    vessel_path = []
    for i, cut in enumerate(cuts):
        if len(cut[0]) == 0:
            logger.warning("Empty cut in vesselcreation passed")
            continue
        points = points_from_ring(cut[0], cut[1], False)
        if demonstration:
            selected_cuts.append(points)
        # use crude diameter estimation as backup
        diameters.append(calculate_diameter(points))
        vessel_path.append(
            calculate_centered_point(points, 2, planes[i][0], planes[i][1])[0]
        )
    if len(diameter_ellipsis_as_circle) != 0 and len(
        diameter_ellipsis_as_circle
    ) == len(vessel_path):
        diameters = list(diameter_ellipsis_as_circle)
    else:
        diameters = diameters

    logger.print("----")
    logger.print(vessel_path[0:100])
    assert len(vessel_path) == len(diameters)
    # clear up unfortunate cuts in path and guarantee, that from one point to the next is shorter than to the point after the next point
    changed = True
    lower_checked = 0
    while changed:
        changed = False
        for i, el in enumerate(vessel_path):
            ci = lower_checked + i
            if (ci < len(vessel_path) - 2) and (
                calculate_distance_euclidian(vessel_path[ci + 1], vessel_path[ci])
                > calculate_distance_euclidian(vessel_path[ci + 2], vessel_path[ci])
            ):
                vessel_path.pop(ci + 1)
                diameters.pop(ci + 1)
                changed = True
                lower_checked = max(0, ci - 1)
    # at the end at an intersection point with the mesh to account for empty space due to missing guidepoints. Copy NB diameter

    n_first = min(4, len(vessel_path))  # Number of points to consider at the start
    n_last = min(4, len(vessel_path))  # Number of points to consider at the end

    # Assuming the normals are provided or can be computed for each point, e.g., from a point cloud

    # calculate combined normals
    directions_to_first = np.asarray(vessel_path[0]) - np.asarray(
        vessel_path[1:n_first]
    )
    normals_to_first = directions_to_first / np.linalg.norm(
        directions_to_first, axis=1
    ).reshape(-1, 1)
    first_avg_normal = np.mean(normals_to_first, axis=0)
    first_avg_normal /= np.linalg.norm(first_avg_normal)

    directions_to_last = np.asarray(vessel_path[-1]) - np.asarray(
        vessel_path[-n_last:-1]
    )
    normals_to_last = directions_to_last / np.linalg.norm(
        directions_to_last, axis=1
    ).reshape(-1, 1)
    last_avg_normal = np.mean(normals_to_last, axis=0)
    last_avg_normal /= np.linalg.norm(last_avg_normal)

    # direction of the ray
    ray_origin_first = vessel_path[0]
    ray_direction_first = first_avg_normal

    ray_origin_last = vessel_path[-1]
    ray_direction_last = last_avg_normal

    # Ray-mesh intersection for the points
    locations_first, index_ray_first, index_tri_first = submesh.ray.intersects_location(
        ray_origins=[ray_origin_first], ray_directions=[ray_direction_first]
    )

    locations_last, index_ray_last, index_tri_last = submesh.ray.intersects_location(
        ray_origins=[ray_origin_last], ray_directions=[ray_direction_last]
    )

    if len(locations_first) > 0:
        logger.print("First Intersection Point:", locations_first[0])
    else:
        logger.print("No intersection found for the first ray.")
    if len(locations_last) > 0:
        logger.print("Last Intersection Point:", locations_last[0])
    else:
        logger.print("No intersection found for the last ray.")

    vessel_path.append(locations_last[0])
    diameters.append(diameters[-1] * 0.8)
    vessel_path.insert(0, locations_first[0])
    diameters.insert(0, diameters[0] * 0.8)

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
    if len(insertions) < 2:
        if 0 in insertions:
            vessel.path_insertion = "a"
        else:
            vessel.path_insertion = "b"
    else:
        vessel.path_insertion = "both"

    if demonstration:
        vessel.cuts = cuts
    if demonstration:
        vessel.selected_cuts = selected_cuts
    vessel.volume = submesh.volume
    vessel.submesh = None
    if not del_submeshes:
        vessel.submesh = submesh
    smooth_fill_end_diameter_single(vessel)
    return vessel


def create_vessels(
    cuts_in_mesh,
    name,
    submeshes,
    speed_function,
    diameter_ellipsis_as_circle,
    demonstration=False,
    plane_definitions=None,
    del_submesh=False,
):
    """
    Function to calculate a vesselobject from a given split stl mesh
    returns created vessel without links or direction
    """
    vessels = []
    for si, cuts in enumerate(cuts_in_mesh):
        if plane_definitions is None:
            vessel = create_vessel_single(
                submeshes[si], diameter_ellipsis_as_circle, cuts, name
            )
        else:
            vessel = create_vessel_single(
                submeshes[si],
                diameter_ellipsis_as_circle,
                cuts,
                name,
                planes=plane_definitions[si],
                del_submeshes=del_submesh,
            )
        vessels.append(vessel)
    smooth_fill_end_diameters(vessels)
    return vessels


def smooth_fill_end_diameter_single(vessel):
    if not hasattr(vessel, "diameters"):
        if "jump" in vessel.associated_vesselname:
            vessel.avg_diameter = 15
        vessel.diameters = [vessel.avg_diameter for _ in vessel.path]

    # more points than diameters:
    if len(vessel.diameters) < len(vessel.path):
        # fill in avg diameter if none were set (only "heart")
        if len(vessel.diameters) == 0:
            for p in vessel.path:
                vessel.diameters.append(vessel.avg_diameter)
        # if one misses
        if len(vessel.diameters) == len(vessel.path) - 1:
            vessel.diameters = list(vessel.diameters)
            # find out on which side the point distance is irregular
            if vessel.path_insertion == "a":
                vessel.diameters.insert(0, vessel.diameters[0])
            elif vessel.path_insertion == "b":
                vessel.diameters.append(vessel.diameters[len(vessel.diameters) - 1])
        # if 2 miss, add to both sides a copy of the relative side
        if len(vessel.diameters) == len(vessel.path) - 2:
            vessel.diameters = list(vessel.diameters)
            vessel.diameters.append(vessel.diameters[len(vessel.diameters) - 1])
            vessel.diameters.insert(0, vessel.diameters[0])
    ten_percent_odd = len(vessel.path) / 10
    if ten_percent_odd % 2 == 0:
        ten_percent_odd += 1
    smoothing_window = min(5, ten_percent_odd)
    # use min(5,10 percent of diameters)
    vessel.diameters = moving_average(vessel.diameters, smoothing_window)


def smooth_fill_end_diameters(vessels):
    # fill ends with duplicate, if empty fill with avg to make len path equal len dia, afterwards use moving average to smooth out error and keep trends
    for vessel in vessels:
        smooth_fill_end_diameter_single(vessel)


# def center_paths(paths, submeshes):  # TODO used?
#     centered_paths = []
#     for m, mesh in enumerate(meshes):
#         submeshesm = submeshes[m]
#         centered_paths.append([])
#         for p, path in paths[m]:
#             submesh = submeshesm[p]
#             centered_paths[m].append([])
#             for q, point in path:
#                 if q < len(path) + 1:
#                     cut = trimesh.intersections.mesh_plane(
#                         submesh,
#                         path[q + 1] - path[q],
#                         point,
#                         return_faces=False,
#                         local_faces=None,
#                         cached_dots=None,
#                     )

#                 else:
#                     cut = (
#                         trimesh.intersections.mesh_plane(
#                             submesh,
#                             path[q] - path[q - 1],
#                             point,
#                             return_faces=False,
#                             local_faces=None,
#                             cached_dots=None,
#                         ),
#                         plane_position,
#                     )
#                 cut = ([x for sub in cut[0] for x in sub], cut[1])
#                 centered_paths[m][p].append(
#                     calculate_centered_point(points_from_ring(cut[0], cut[1], False), 0)
#                 )
#     return centered_paths


def get_normals(paths, submeshes, distance_between_pathpoints):
    """
    get normals between pathpoints
    """
    normals = []
    for p, path in enumerate(paths):
        normals.append([])
        guide_point = path[0]
        for next_point in path[1::]:
            normal = [
                next_point[0] - guide_point[0],
                next_point[1] - guide_point[1],
                next_point[2] - guide_point[2],
            ]
            # normiere vektor und multipliziere mit Schnittabstand
            normal_len = mag(normal)
            normal = np.asarray(normal)
            normal = normal / normal_len
            normal = normal * distance_between_pathpoints
            normals[p].append(normal)
            guide_point = next_point
    # zu starke winkel abschwaechen TODO
    """ for p,path in enumerate(paths):
        for n, normal_x in enumerate(normals[p]):
                if(n+1<len(normals[p]) and n-1>0):
                    if(calculate_angle_difference(normal_x, normals[p][n+1])>35 ):
                        normals[p][n]=(normals[p][n+1]+normals[p][n-1])/2
                        normal_len=mag(normals[p][n])
                        normals[p][n]=normals[p][n]/normal_len"""
    return normals


def get_normals_single(path, distance_between_pathpoints):
    """
    get normals between pathpoints
    """
    normals = []
    guide_point = path[0]
    for next_point in path[1::]:
        normal = [
            next_point[0] - guide_point[0],
            next_point[1] - guide_point[1],
            next_point[2] - guide_point[2],
        ]
        # normiere vektor und multipliziere mit Schnittabstand
        normal_len = mag(normal)
        normal = np.asarray(normal)
        normal = normal / normal_len
        normal = normal * distance_between_pathpoints
        normals.append(normal)
        guide_point = next_point
    return normals


def create_cuts(submesh, path, plane_normals, name, distance_between_pathpoints):
    """
    create cuts in a mesh with planes with given normals at pathpoint positions to get the centered pathpoints and diameter data
    """
    """ if("heart" in name):
        return path """
    # normals and path are both nparrays!
    # assert that the path is straight inside the submesh
    plane_definitions = []
    for i in range(len(path) - 2):
        if calculate_distance_euclidian(
            path[i], path[i + 1]
        ) >= calculate_distance_euclidian(path[i], path[i + 2]):
            path[i + 1] = [
                math.sqrt((path[i][0] - path[i + 2][0]) ** 2),
                math.sqrt((path[i][1] - path[i + 2][1]) ** 2),
                math.sqrt((path[i][2] - path[i + 2][2]) ** 2),
            ]
            logger.print("CHANGED PATH")
    pathcuts = []
    index = 0
    plane_position = path[index].copy()
    plane_normal = plane_normals[index].copy()
    plane_position += (
        plane_normal / 20
    )  # set first position to be a bit further from the end
    cut = (
        trimesh.intersections.mesh_plane(
            submesh,
            plane_normal,
            plane_position,
            return_faces=False,
            local_faces=None,
            cached_dots=None,
        ),
        plane_position,
    )
    cut = ([x for sub in cut[0] for x in sub], cut[1])
    # logger.print("first cut", len(cut[0]),index, "of ", len(path))
    use_next_normal = False
    diff_to_next_point = calculate_distance_euclidian(plane_position, path[index + 1])
    while len(cut[0]) > 0:  # and index < len(path):
        last_diff = diff_to_next_point
        diff_to_next_point = calculate_distance_euclidian(
            plane_position, path[index + 1]
        )
        pathcuts.append(cut)

        plane_definitions.append((plane_position, plane_normal))
        # pick normal for nearest pathpoint
        # d=cdist([plane_position],path,"euclidean")
        # index=np.where(d==d.min())[1][0]
        if diff_to_next_point <= 1.6 or len(cut[0]) == 0:
            use_next_normal = True

        plane_position = plane_position.copy() + plane_normal  # step forward
        cut = (
            trimesh.intersections.mesh_plane(
                submesh,
                plane_normal,
                plane_position,
                return_faces=False,
                local_faces=None,
                cached_dots=None,
            ),
            plane_position,
        )
        cut = ([x for sub in cut[0] for x in sub], cut[1])
        if use_next_normal and (
            index < len(plane_normals) - 1
        ):  # use next normal if available
            index += 1
            plane_normal = plane_normals[index].copy()
            use_next_normal = False
        # logger.print("cut", len(cut[0]), index, "of ", len(path))
        if len(cut[0]) == 0:  # terminate on empty cut
            # empty cut
            break

        if (
            index == len(plane_normals) - 1
        ):  # and not submesh.contains([plane_position])[0]:  # check if there needs to be new cuts, without changing to another part of the vessel(U shaped ones!), as the meshes ar easserrted to be watertight, mesh.contains suffices
            # logger.print("BREAK")
            pathcuts.append(cut)
            plane_definitions.append((plane_position, plane_normal))
            break
    # as this only cuts between first guidingpoint and last one, but the mesh may continue beyond those for a bit, we need to add a few pre- and post intersections with newly extracted normals along our plane definitions
    """ plane_positions=(plane_position, plane_normal)
    base_point_pre=plane_definitions[0][0]
    base_point_post=plane_definitions[-1][0]
    normals=[[
                -next_point[0] + base_point_post[0],
                -next_point[1] + base_point_post[1],
                -next_point[2] + base_point_post[2],
            ]#points from -5,-4,-3,-2 to -1
             for next_point,nn in plane_definitions[-4:-1]]

    post_normal=np.sum(normals, axis=0)/mag(np.sum(normals, axis=0))
    normals=[[
                - next_point[0] + base_point_pre[0],
                - next_point[1] + base_point_pre[1],
                - next_point[2] + base_point_pre[2],
            ]#points from 1,2,3,4 to 0
             for next_point,nn in plane_definitions[1:4]]

    pre_normal=np.sum(normals, axis=0)/mag(np.sum(normals, axis=0))

    plane_position=base_point_pre+pre_normal
    cut = (
            trimesh.intersections.mesh_plane(
                submesh,
                pre_normal,
                plane_position,
                return_faces=False,
                local_faces=None,
                cached_dots=None,
            ),
            plane_position,
        )
    cut = ([x for sub in cut[0] for x in sub], cut[1])
    #cut towards pre end and insert
    while(len(cut)>3 and submesh.contains([plane_position])[0]):
        plane_position=plane_position.copy() + pre_normal
        plane_definitions.insert(0, (plane_position, pre_normal))
        pathcuts.insert(0,cut)
        #create next cut
        cut = (
            trimesh.intersections.mesh_plane(
                submesh,
                pre_normal,
                plane_position,
                return_faces=False,
                local_faces=None,
                cached_dots=None,
            ),
            plane_position,
        )
        cut = ([x for sub in cut[0] for x in sub], cut[1])

    plane_position=base_point_post+post_normal
    cut = (
            trimesh.intersections.mesh_plane(
                submesh,
                post_normal,
                plane_position,
                return_faces=False,
                local_faces=None,
                cached_dots=None,
            ),
            plane_position,
        )
    cut = ([x for sub in cut[0] for x in sub], cut[1])
    #cut towards post end and append
    while(len(cut)>0 and submesh.contains([plane_position])[0]):
        plane_position=plane_position.copy() + post_normal
        plane_definitions.append((plane_position, post_normal))
        pathcuts.append(cut)
        cut = (
            trimesh.intersections.mesh_plane(
                submesh,
                post_normal,
                plane_position,
                return_faces=False,
                local_faces=None,
                cached_dots=None,
            ),
            plane_position,
        )
        cut = ([x for sub in cut[0] for x in sub], cut[1])  """

    return pathcuts, plane_definitions


class Graph:
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


def stable_guidepoints_single(
    submesh, m, name, min_dinstance_search, demonstration=False
):
    # voxelizes submesh and extracts skeleton coordinates. Does not utilise min distance search
    print(submesh)
    voxelgrid = submesh.voxelized(pitch=1)
    print("Voxel grid created")
    voxelgrid = voxelgrid.fill()
    print("Voxel grid filled")
    if np.all(voxelgrid.matrix == 0):
        print("Voxel grid matrix is all zeros after fill()")
    voxelgrid_skeleton = skeletonize(voxelgrid.matrix)
    still_set_voxels = set(tuple(entry) for entry in np.argwhere(voxelgrid_skeleton))
    print(f"Still set voxels: {len(still_set_voxels)}")
    graph = Graph()
    skeleton = voxelgrid_skeleton
    for x, y, z in still_set_voxels:
        adjacent_coordinates = get_adjacent_indices((x, y, z), skeleton.shape)
        for coord in adjacent_coordinates:
            if coord in still_set_voxels:
                graph.add_edge(((x, y, z), coord))
    edges_p_node = set()
    for node in graph.node_to_edges:
        edges_p_node.add(len(graph.node_to_edges[node]))

    G = nx.Graph()
    for node in graph.node_to_edges:
        G.add_node(node)
        for edge in graph.node_to_edges[node]:
            G.add_edge(node, edge)

    longest_path_nodes = []
    for node in graph.node_to_edges:
        if len(graph.node_to_edges[node]) == 1:
            longest_path_nodes.append((node, longest_path_from_node(node, graph)))

    longest_path_nodes = sorted(longest_path_nodes, key=lambda x: x[1])
    print(longest_path_nodes)
    assert longest_path_nodes[-1][1] == longest_path_nodes[-2][1]
    l1, l2 = longest_path_nodes[-2][0], longest_path_nodes[-1][0]
    pitch = voxelgrid.pitch

    output = []
    for coords in nx.shortest_path(G, l1, l2):
        voxel_center_homogeneous = np.append(coords, 1)  # Make it homogeneous
        voxel_center = voxelgrid.transform.dot(voxel_center_homogeneous)[:3]
        # logger.print(coords,voxelgrid.matrix.shape)
        output.append(voxel_center)
    return [output[0]] + output[1:-1:10] + [output[-1]]


def create_guide_points(submeshes, m, name, min_dinstance_search, demonstration=False):
    """
    Take all mesh vertices and approximate pathpoints that lay inside a given watertight mesh and go from one end to another given a shape as <=========>
    """
    guide_points = []
    for s, submesh in enumerate(submeshes):
        # PREV create_guide_points_single  stable_guidepoints_single
        guide_points.append(
            stable_guidepoints_single(
                submesh, m, name, min_dinstance_search, demonstration
            )
        )
    return guide_points


def create_guide_points_single(
    submesh, m, name, min_dinstance_search, demonstration=False
):
    """
    Take all mesh vertices and approximate pathpoints that lay inside a given watertight mesh and go from one end to another given a shape as <=========>
    """
    if demonstration:
        selections = []
    if "heart" in name:
        min_dinstance_search = 8
    elif "vl.blood_vessels.veins.stl" in name:
        min_dinstance_search = 8  # hat ein sehr kugeliges gefaess
    else:
        min_dinstance_search = 4.5
    # logger.print("SEARCHDISTANCE",min_dinstance_search)
    guide_points = []
    shape = {i for sub in submesh.faces for i in sub}

    meshneighbours = read_neighbours(submesh.faces, len(submesh.vertices))
    startindex = highestNeighbourCount(shape, meshneighbours)[0]
    guide_point = submesh.vertices[startindex]
    guide_points.append(guide_point)
    unused_points = shape
    while unused_points:
        temp_distances = [
            [
                calculate_distance_euclidian(
                    guide_point, submesh.vertices[point_index]
                ),
                point_index,
            ]
            for point_index in unused_points
        ]

        temp_distances.sort()
        selection = [
            x
            for x in temp_distances
            if x[0] <= min_dinstance_search + temp_distances[0][0]
        ]
        points_from_selection = [submesh.vertices[x[1]] for x in selection]
        if demonstration:
            selections.append(points_from_selection)
        # logger.print("in selection:", len(selection), "--", min_dinstance_search+temp_distances[0][0])
        guide_point = np.asarray(calculate_centered_point(points_from_selection, 0))
        guide_points.append(guide_point)
        unused_points = [
            x[1]
            for x in temp_distances
            if x[0] > min_dinstance_search + temp_distances[0][0]
        ]
    if demonstration:
        return guide_points, selections
    return guide_points


def points_from_ring(cut, origin, all_rings, debug=False):
    """
    get points from a cut when a mesh got cut multiple times
    Approximate result, based on first derivative of point distance with a cutoff at 2mm TODO parameterize?
    returns ring thats closest to the given origin
    """
    # detect pointclouds that are grouped together and only return the closest one to originpoint
    assert len(origin) == 3, "Origin should be 3d coordinate"
    # take all points and put into list, may include duplicates
    if len(cut) < 1:
        return []
    cutpoints = []
    for point in cut:
        if not point_in(point, cutpoints):
            cutpoints.append(point)

    # get points that are farthest apart
    distances = pdist(cutpoints, metric="euclidean")
    max_dist_pair_indices = condensed_to_square(
        np.where(distances == distances.max())[0][0], len(cutpoints)
    )
    outerpoint = cutpoints[max_dist_pair_indices[0]]

    # sort list by distance to farthest point
    distances_for_list = []  # [distance, index in cutpoints]
    for pointindex, point in enumerate(cutpoints):
        distances_for_list.append(
            (calculate_distance_euclidian(outerpoint, point), pointindex)
        )
    distances_for_list.sort()
    if debug:
        logger.print("distancelist_sorted_from_farthest", distances_for_list)
    sorted_cutpoints = []
    for i, point_index_pair in enumerate(distances_for_list):
        sorted_cutpoints.append(cutpoints[point_index_pair[1]])

    # detect bigger jumps in distance to the outest point indicating a new cut thats not connected to the previous
    distanceincrease = [distances_for_list[x][0] for x in range(len(sorted_cutpoints))]
    if debug:
        logger.print("distanceincrease_sorted_from_farthest", distanceincrease)
    dist_der = [
        -distanceincrease[x] + distanceincrease[x + 1]
        for x in range(len(distanceincrease) - 1)
    ]
    if debug:
        logger.print("distancechange_sorted_from_farthest", dist_der)

    # teste alle splits mit maxima und waehle den split mit der groessten anzahl, der gleichmaessig splittet und alle extreama sind groesser als die standardabweichung
    ring_collection = [[]]
    i = 0
    # logger.print(distanceincrease)
    # logger.print(dist_der)
    trenndistanz = max(
        2.5, max(dist_der) / 3
    )  # calculate_diameter(sorted_cutpoints)//2
    # logger.print("trenndistanz:",trenndistanz)
    # logger.print(max(dist_der), np.mean(dist_der))
    for p, point in enumerate(sorted_cutpoints):
        if (p > 0 and p < len(sorted_cutpoints) - 1) and (
            dist_der[p - 1] > trenndistanz
        ):
            ring_collection.append([])
            i += 1
        ring_collection[i].append(point)
    # return only closest ring to origin
    if not all_rings:
        if len(ring_collection) <= 1:
            return ring_collection[0]
        origin_dist_to_rings = [
            calculate_distance_euclidian(origin, ringcenter)
            for ringcenter in [
                calculate_centered_point(ring, 0) for ring in ring_collection
            ]
        ]
        # logger.print(origin_dist_to_rings)
        # logger.print("returning points from ring",origin_dist_to_rings.index(min(origin_dist_to_rings)),"of", len(ring_collection)-1 )
        return ring_collection[
            origin_dist_to_rings.index(min(origin_dist_to_rings))
        ]  # nearest ring to origin

    return ring_collection


# ------------------ sequ link gathering methods --------------------------------


def create_connections(vessels):
    """
    creates connections between vessels according to their endpoint relations (which endpoint lies in which mesh)
    """
    vessel_dict = {}
    for vessel in vessels:
        vessel_dict[vessel.id] = vessel
    """{1919205010992: <vessel_functions.Vessel at 0x28a6068c4c0>,
    1919213011104: <vessel_functions.Vessel at 0x28a2a4fc640>,
    1919213011296: <vessel_functions.Vessel at 0x28a2ad7fd60>,"""

    mesh_color = [100, 100, 100, 50]
    path_point_color = [100, 100, 100, 50]
    highlight_color = [255, 0, 0, 250]

    linkoperations = []  # [operationtarget, link]

    vessel_endpoints = []
    vessel_boundaries = []
    insert_index = -1
    for vessel in vessels:
        vessel_endpoints.extend((vessel.path[0], vessel.path[len(vessel.path) - 1]))
        vessel_boundaries.append([insert_index + 1, vessel])
        insert_index = insert_index + 2

    vessel_endpoints = np.asarray(vessel_endpoints)

    link_collect = [
        get_vessel_endpoints_per_mesh(
            vessels[i], i, vessel_endpoints, vessel_boundaries
        )
        for i in range(len(vessels))
    ]
    endpoint_dict = {}
    for endp_submesh_collection in link_collect:
        if len(endp_submesh_collection) > 1:
            for endpoints in endp_submesh_collection[1::]:
                if endpoints not in endpoint_dict:
                    endpoint_dict[endpoints] = []
                endpoint_dict[endpoints].append(endp_submesh_collection[0])
    """{(1919213011824, 42): [1919213011680],
    (1919213012400, 28): [1919213011680, 1919213015424],
    (1919213013888, 58): [1919213011680],...}"""

    connections = []
    count = 0
    for endpointkey in endpoint_dict:
        best_vessel_to_connect = None
        if len(endpoint_dict[endpointkey]) > 1:
            if len(endpoint_dict[endpointkey]) == 2:
                # in 2 anderen meshes -> 64 moegliche in B,C relationen (a in b, a in c, b in c, b in a, c in a, c in b)
                # wenn a in b und b in a , auch fuer c-> dann seien diese verbunden und alle dazwichen werden geloescht
                a = vessel_dict[endpointkey[0]]
                b = vessel_dict[endpoint_dict[endpointkey][0]]
                c = vessel_dict[endpoint_dict[endpointkey][1]]
                if endpoint_pair_relation(endpointkey, b, vessel_dict)[0]:
                    count += 1
                    if endpoint_pair_relation(endpointkey, c, vessel_dict)[0]:
                        logger.print("a-b-c")
                        # create_connection(c, endpointkey,vessel_dict,connections,"abc - triple")
                        logger.print("---------------", a.id, b.id, c.id)
                        logger.print(
                            " a point",
                        )
                        # dreieck, wenn man alle verbindet, dann kann es zu leichten ungenauigkeiten kommen
                        # connect a and b to c
                        create_connection(
                            b, endpointkey, vessel_dict, connections, "ab connect to c"
                        )
                        create_connection(
                            c, endpointkey, vessel_dict, connections, "ab connect to c"
                        )
                    # assert 1==2, "das wird nicht verwendet"
                    else:
                        logger.print("ab connect to c")
                        create_connection(
                            c, endpointkey, vessel_dict, connections, "ab connect to c"
                        )
                        continue

                elif endpoint_pair_relation(endpointkey, c, vessel_dict)[0]:  # nur a-c
                    count += 1
                    logger.print("ac connect to b")
                    create_connection(
                        b, endpointkey, vessel_dict, connections, "ac connect to b"
                    )
                    continue
                else:
                    count += 1
                    logger.print("nur a zu iwem connect to closest biggest")
                    highest_volume_vessel = b
                    if c.volume > highest_volume_vessel.volume:
                        highest_volume_vessel = c
                    create_connection(
                        highest_volume_vessel,
                        endpointkey,
                        vessel_dict,
                        connections,
                        "nur a zu iwem connect to closest biggest",
                    )
                    continue

            if len(endpoint_dict[endpointkey]) == 3:
                # in 2 anderen meshes -> 128 moegliche relationen von endpunkt in mesh
                a = vessel_dict[endpointkey[0]]
                b = vessel_dict[endpoint_dict[endpointkey][0]]
                c = vessel_dict[endpoint_dict[endpointkey][1]]
                d = vessel_dict[endpoint_dict[endpointkey][2]]
                if endpoint_pair_relation(endpointkey, b, vessel_dict)[0]:
                    if endpoint_pair_relation(endpointkey, b, vessel_dict)[0]:
                        if endpoint_pair_relation(endpointkey, c, vessel_dict)[0]:
                            if endpoint_pair_relation(endpointkey, d, vessel_dict)[0]:
                                logger.print("abcd 2")
                                assert 1 == 2, "das hier wird nicht gebraucht"
                                continue
                        logger.print("abc 2 connect to d")  # TODO 2
                        count += 1
                        create_connection(
                            d,
                            endpointkey,
                            vessel_dict,
                            connections,
                            "abc 2 connect to d",
                        )
                        continue
                    logger.print("ab 2")
                    assert 1 == 2, "das hier wird nicht gebraucht"
                    continue
                if endpoint_pair_relation(endpointkey, c, vessel_dict)[0]:
                    if endpoint_pair_relation(endpointkey, d, vessel_dict)[0]:
                        logger.print("acd 2")
                        assert 1 == 2, "das hier wird nicht gebraucht"
                        continue
                    logger.print("ac 2 connect to biggetr between b and d")  # TODO 1
                    highest_volume_vessel = b
                    if d.volume > highest_volume_vessel.volume:
                        highest_volume_vessel = d
                    create_connection(
                        highest_volume_vessel,
                        endpointkey,
                        vessel_dict,
                        connections,
                        "ac 2 connect to biggetr between b and d",
                    )
                    continue
                if endpoint_pair_relation(endpointkey, d, vessel_dict)[0]:
                    logger.print("ad 2")
                    assert 1 == 2, "das hier wird nicht gebraucht"
                    continue

                else:
                    count += 1
                    logger.print("nur a zu iwem connect to closest biggest 2")
                    highest_volume_vessel = b
                    if c.volume > highest_volume_vessel.volume:
                        highest_volume_vessel = c
                    create_connection(
                        highest_volume_vessel,
                        endpointkey,
                        vessel_dict,
                        connections,
                        "nur a zu iwem connect to closest biggest 2",
                    )
                    continue

        else:
            count += 1
            best_vessel_to_connect = vessel_dict[endpoint_dict[endpointkey][0]]
            nearest_connection_point_index = closest_point_in_path_index(
                best_vessel_to_connect, vessel_dict[endpointkey[0]].path[endpointkey[1]]
            )
            connections.append(
                (
                    vessel_dict[endpointkey[0]],
                    Link(
                        vessel_dict[endpointkey[0]],
                        endpointkey[1],
                        best_vessel_to_connect,
                        nearest_connection_point_index,
                        "normalh",
                    ),
                    "normal h",
                )
            )  # von ende zu naechster vessel
            connections.append(
                (
                    best_vessel_to_connect,
                    Link(
                        best_vessel_to_connect,
                        nearest_connection_point_index,
                        vessel_dict[endpointkey[0]],
                        endpointkey[1],
                        "normalz",
                    ),
                    "normal z",
                )
            )  # zurueck

    # logger.print("connected",count,"of",len(endpoint_dict))
    return connections


def manual_link_insertion_heart(vessels):
    center_left_chamber = [179.24761044, 203.81855294, 248.20464232]
    center_right_chamber = [148.71463263, 181.67490973, 244.12171739]

    # vessel fuer aorta, pulmonargefaesse und hohlvene feststellen
    vesselcollection_pulmonary_right = []
    vesselcollection_pulmonary_left = []
    vesselcollection_pulmonar_artery_left = []
    vesselcollection_pulmonar_artery_right = []
    for vessel in vessels:
        if "aorta" in vessel.associated_vesselname:
            aorta_vessel = vessel
        if "bot" in vessel.associated_vesselname:
            bot_vein = vessel
        if "top" in vessel.associated_vesselname:
            top_vein = vessel
        if (
            "al.blood_vessels.pulmonary_arts.left_pulmonary_arts"
            in vessel.associated_vesselname
        ):
            vesselcollection_pulmonar_artery_left.append(vessel)
        if (
            "al.blood_vessels.pulmonary_arts.right_pulmonary_arts"
            in vessel.associated_vesselname
        ):
            vesselcollection_pulmonar_artery_right.append(vessel)
        if (
            "vl.blood_vessels.pulmonary_veins.left_pulmonary_veins"
            in vessel.associated_vesselname
        ):
            vesselcollection_pulmonary_left.append(vessel)
        if (
            "vl.blood_vessels.pulmonary_veins.right_pulmonary_veins"
            in vessel.associated_vesselname
        ):
            vesselcollection_pulmonary_right.append(vessel)

    pulmonar_vein_left = find_vessel_longest(vesselcollection_pulmonary_left)

    # logger.print(vesselcollection_pulmonar_artery_left)
    pulmonar_artery_left = find_vessel_longest(vesselcollection_pulmonar_artery_left)

    con_pul_v_r = find_connected_structures(vesselcollection_pulmonary_right)
    vesselcollection_pulmonary_right = [
        part for part in con_pul_v_r if len(part) > 3
    ]  # obsolet?
    logger.print([len(part) for part in vesselcollection_pulmonary_right])
    assert len(vesselcollection_pulmonary_right) == 2, len(
        vesselcollection_pulmonary_right
    )
    pulmonar_vein_right_1 = find_vessel_longest(vesselcollection_pulmonary_right[0])
    pulmonar_vein_right_2 = find_vessel_longest(vesselcollection_pulmonary_right[1])

    pulmonar_artery_right = find_vessel_longest(vesselcollection_pulmonar_artery_right)

    assert pulmonar_artery_left is not None
    assert pulmonar_artery_right is not None
    assert pulmonar_vein_right_1 is not None
    assert bot_vein is not None
    assert top_vein is not None
    assert pulmonar_vein_right_2 is not None
    assert pulmonar_vein_left is not None
    assert aorta_vessel is not None

    jump_vessel_left_chamber = Vessel(
        [center_left_chamber, [179, 204, 248]],
        0,
        0,
        0,
        [0, 0],
        "vl.jump_vessel_left_chamber",
        standard_speed_function,
    )

    if calculate_distance_euclidian(
        aorta_vessel.path[0], center_left_chamber
    ) < calculate_distance_euclidian(
        aorta_vessel.path[len(aorta_vessel.path) - 1], center_left_chamber
    ):
        aorta_link_index = 0
    else:
        aorta_link_index = len(aorta_vessel.path) - 1
    # von pulmonarvenen
    if calculate_distance_euclidian(
        pulmonar_vein_left.path[0], center_left_chamber
    ) < calculate_distance_euclidian(
        pulmonar_vein_left.path[len(pulmonar_vein_left.path) - 1], center_left_chamber
    ):
        # if(pulmonar_vein_left.diameter_a>pulmonar_vein_left.diameter_b):
        link_pul_left_index = 0
    else:
        link_pul_left_index = len(pulmonar_vein_left.path) - 1

    if calculate_distance_euclidian(
        pulmonar_vein_right_1.path[0], center_left_chamber
    ) < calculate_distance_euclidian(
        pulmonar_vein_right_1.path[len(pulmonar_vein_right_1.path) - 1],
        center_left_chamber,
    ):
        # if(pulmonar_vein_right_1.diameter_a>pulmonar_vein_right_1.diameter_b):
        pulmonar_vein_right_1_index = 0
    else:
        pulmonar_vein_right_1_index = len(pulmonar_vein_right_1.path) - 1

    jump_vessel_right_chamber = Vessel(
        [center_right_chamber, [149, 182, 244]],
        0,
        0,
        0,
        [0, 0],
        "al.jump_vessel_right_chamber",
        standard_speed_function,
    )

    # holhlvene (2Teile)
    if calculate_distance_euclidian(
        bot_vein.path[0], center_right_chamber
    ) < calculate_distance_euclidian(
        bot_vein.path[len(bot_vein.path) - 1], center_right_chamber
    ):
        link_bot_vein_index = 0
    else:
        link_bot_vein_index = len(bot_vein.path) - 1
    if calculate_distance_euclidian(
        top_vein.path[0], center_right_chamber
    ) < calculate_distance_euclidian(
        top_vein.path[len(top_vein.path) - 1], center_right_chamber
    ):
        link_top_vein_index = 0
    else:
        link_top_vein_index = len(top_vein.path) - 1

    # pulmonararterien
    if calculate_distance_euclidian(
        pulmonar_artery_left.path[0], center_right_chamber
    ) < calculate_distance_euclidian(
        pulmonar_artery_left.path[len(pulmonar_artery_left.path) - 1],
        center_right_chamber,
    ):
        # if(pulmonar_artery_left.diameter_a>pulmonar_artery_left.diameter_b):
        link_pulmonar_artery_left_index = 0
    else:
        link_pulmonar_artery_left_index = len(pulmonar_artery_left.path) - 1

    if calculate_distance_euclidian(
        pulmonar_artery_right.path[0], center_right_chamber
    ) < calculate_distance_euclidian(
        pulmonar_artery_right.path[len(pulmonar_artery_right.path) - 1],
        center_right_chamber,
    ):
        # if(pulmonar_artery_right.diameter_a>pulmonar_artery_right.diameter_b):
        link_pulmonar_artery_right_index = 0
    else:
        link_pulmonar_artery_right_index = len(pulmonar_artery_right.path) - 1

    if calculate_distance_euclidian(
        pulmonar_vein_right_2.path[0], center_right_chamber
    ) < calculate_distance_euclidian(
        pulmonar_vein_right_2.path[len(pulmonar_vein_right_2.path) - 1],
        center_right_chamber,
    ):
        # if(pulmonar_vein_right_2.diameter_a>pulmonar_vein_right_2.diameter_b):
        pulmonar_vein_right_2_index = 0
    else:
        pulmonar_vein_right_2_index = len(pulmonar_vein_right_2.path) - 1

    if jump_vessel_left_chamber not in vessels:
        vessels.insert(0, jump_vessel_left_chamber)
        logger.print("added Vessel left chamber")
    if jump_vessel_right_chamber not in vessels:
        vessels.insert(0, jump_vessel_right_chamber)
        logger.print("added Vessel right chamber")
    # apply links
    # links:
    """
        top ------\
        hohlvene    right_chamber0 rc1 ASC----- lungenarterien
        bot ------/

        lungenvenen -----\
                            lc 1 left_chamber0 DESC ------ aorta
        lungtenvenen -----/

    """
    top_vein.tags.append("at_heart")
    bot_vein.tags.append("at_heart")
    pulmonar_artery_left.tags.append("at_heart")
    pulmonar_artery_right.tags.append("at_heart")
    pulmonar_vein_right_2.tags.append("at_heart")
    aorta_vessel.tags.append("at_heart")
    pulmonar_vein_right_1.tags.append("at_heart")
    pulmonar_vein_left.tags.append("at_heart")
    if link_top_vein_index == 0:
        top_vein.reverse(static_links=False)
        link_top_vein_index = len(top_vein.path) - 1
        top_vein.tags.append("safe_dir")
    else:
        top_vein.tags.append("safe_dir")

    if link_bot_vein_index == 0:
        bot_vein.reverse(static_links=False)
        link_bot_vein_index = len(bot_vein.path) - 1

        bot_vein.tags.append("safe_dir")
    else:
        bot_vein.tags.append("safe_dir")

    if link_pulmonar_artery_left_index == 0:
        pulmonar_artery_left.tags.append("safe_dir")
    else:
        pulmonar_artery_left.reverse(static_links=False)
        link_pulmonar_artery_left_index = len(pulmonar_artery_left.path) - 1
        pulmonar_artery_left.tags.append("safe_dir")

    if link_pulmonar_artery_right_index == 0:
        pulmonar_artery_right.tags.append("safe_dir")
    else:
        pulmonar_artery_right.reverse(static_links=False)
        link_pulmonar_artery_right_index = len(pulmonar_artery_right.path) - 1
        pulmonar_artery_right.tags.append("safe_dir")

    if pulmonar_vein_right_2_index == 0:
        pulmonar_vein_right_2.reverse(static_links=False)
        pulmonar_vein_right_2_index = len(pulmonar_vein_right_2.path) - 1
        pulmonar_vein_right_2.tags.append("safe_dir")
    else:
        pulmonar_vein_right_2.tags.append("safe_dir")

    if aorta_link_index == 0:
        aorta_vessel.tags.append("safe_dir")
    else:
        aorta_vessel.reverse(static_links=False)
        aorta_link_index = len(aorta_vessel.path) - 1
        aorta_vessel.tags.append("safe_dir")

    if pulmonar_vein_right_1_index == 0:
        pulmonar_vein_right_1.reverse(static_links=False)
        pulmonar_vein_right_1_index = len(pulmonar_vein_right_1.path) - 1
        pulmonar_vein_right_1.tags.append("safe_dir")
    else:
        pulmonar_vein_right_1.tags.append("safe_dir")

    if link_pul_left_index == 0:
        pulmonar_vein_left.reverse(static_links=False)
        link_pul_left_index = len(pulmonar_vein_left.path) - 1
        pulmonar_vein_left.tags.append("safe_dir")
    else:
        pulmonar_vein_left.tags.append("safe_dir")

    top_vein.add_link(
        Link(top_vein, link_top_vein_index, jump_vessel_right_chamber, 0, "normalh")
    )

    bot_vein.add_link(
        Link(bot_vein, link_bot_vein_index, jump_vessel_right_chamber, 0, "normalh")
    )
    # relink?

    ##########RELINK
    """ jump_vessel_right_chamber.add_link(
        Link(0, bot_vein,link_bot_vein_index,"normalh"))


    jump_vessel_right_chamber.add_link(
        Link(0, top_vein, link_top_vein_index,"normalh"))  """
    ##############
    # to lung arteries
    jump_vessel_right_chamber.add_link(
        Link(
            jump_vessel_right_chamber,
            1,
            pulmonar_artery_left,
            link_pulmonar_artery_left_index,
            "normalh",
        )
    )
    jump_vessel_right_chamber.add_link(
        Link(
            jump_vessel_right_chamber,
            1,
            pulmonar_artery_right,
            link_pulmonar_artery_right_index,
            "normalh",
        )
    )

    ######
    """ pulmonar_artery_left.add_link(
       Link(link_pulmonar_artery_left_index , jump_vessel_right_chamber, 1,"normalh"))
    pulmonar_artery_right.add_link(
        Link(link_pulmonar_artery_right_index, jump_vessel_right_chamber, 1 ,"normalh"))  """
    #########

    # left chamber side
    pulmonar_vein_right_2.add_link(
        Link(
            pulmonar_vein_right_2,
            pulmonar_vein_right_2_index,
            jump_vessel_left_chamber,
            0,
            "normalh",
        )
    )
    jump_vessel_left_chamber.add_link(
        Link(jump_vessel_left_chamber, 1, aorta_vessel, aorta_link_index, "normalh")
    )
    #####
    """ jump_vessel_left_chamber.add_link(
        Link(1, pulmonar_vein_right_2, pulmonar_vein_right_2_index ,"normalh"))
    aorta_vessel.add_link(
        Link(aorta_link_index, jump_vessel_left_chamber,0 ,"normalh"))  """
    ########

    pulmonar_vein_right_1.add_link(
        Link(
            pulmonar_vein_right_1,
            pulmonar_vein_right_1_index,
            jump_vessel_left_chamber,
            0,
            "normalh",
        )
    )
    to_rem = []
    for link in pulmonar_vein_left.links_to_path:
        if link.source_index == len(pulmonar_vein_left.path) - 1:
            logger.print("REM", str(link))
            to_rem.append(link)
    for link in to_rem:
        if link.target_index == 0:
            link.target_vessel.reverse()
            link.target_vessel.tags.append("safe_dir")
        pulmonar_vein_left.links_to_path.remove(link)
        logger.print("PULL ", pulmonar_vein_left.path[len(pulmonar_vein_left.path) - 1])
    pulmonar_vein_left.add_link(
        Link(
            pulmonar_vein_left,
            link_pul_left_index,
            jump_vessel_left_chamber,
            0,
            "normalh",
        )
    )

    #####
    """ jump_vessel_left_chamber.add_link(
        Link(1, pulmonar_vein_right_1, pulmonar_vein_right_1_index,"normalh"))
    jump_vessel_left_chamber.add_link(
        Link(1, pulmonar_vein_right_1, link_pul_left_index,"normalh"))  """
    ########

    logger.print(
        "pulmonar_vein_right_1",
        [str(link) for link in pulmonar_vein_right_1.links_to_path],
    )
    logger.print(
        "pulmonar_vein_right_2",
        [str(link) for link in pulmonar_vein_right_2.links_to_path],
    )
    logger.print(
        "pulmonar_vein_left", [str(link) for link in pulmonar_vein_left.links_to_path]
    )
    logger.print("--------------------------------")
    logger.print(
        "pulmonar_artery_left",
        [str(link) for link in pulmonar_artery_left.links_to_path],
    )
    logger.print(
        "pulmonar_artery_right",
        [str(link) for link in pulmonar_artery_right.links_to_path],
    )
    logger.print("--------------------------------")
    logger.print("aorta_vessel", [str(link) for link in aorta_vessel.links_to_path])
    logger.print("--------------------------------")
    logger.print("top_vein", [str(link) for link in top_vein.links_to_path])
    logger.print("bot_vein", [str(link) for link in bot_vein.links_to_path])
    logger.print("--------------------------------")
    logger.print(
        "jump_vessel_left_chamber",
        [str(link) for link in jump_vessel_left_chamber.links_to_path],
    )
    logger.print(
        "jump_vessel_right_chamber",
        [str(link) for link in jump_vessel_right_chamber.links_to_path],
    )
    jump_vessel_left_chamber.tags.append("safe_dir")
    jump_vessel_right_chamber.tags.append("safe_dir")

    """  #set direcctions:
    for vessel in adjacent_veins:
        for link in vessel.links_to_path:
            #set incoming vessels to end with end of path

    for vessel in adjacend_arteries: """


# ---------------------------   general operations on vessels:


def find_vessel_longest(vessels):
    """finds longest vessel"""
    max_length = 0
    for vessel in vessels:
        if vessel.length > max_length:
            selected_vessel = vessel
            max_length = vessel.length
    return selected_vessel


def flip_vessels(vessels):
    """
    Flips the direction of vessels if their direction is not guaranteed as correct.
    Does so until the amount of traversable vessels got maximised
    """
    reversed_vein_method = False
    changed_directions = True
    prev_3 = []
    while changed_directions:
        t1 = get_traversion_regions(vessels)

        change_directions(vessels)

        t2 = get_traversion_regions(vessels)

        t1_c = [len(part) for part in t1]
        t2_c = [len(part) for part in t2]
        t1_c.sort()
        t2_c.sort()
        if t1_c == t2_c:
            logger.print("==", t1_c, "==", t2_c)
            changed_directions = False
        # check ob die vessel in den teilen die gleichen sind
        """ part1
        part2
        if(all([vessel in part2 for vessel in part1])):
            changed_directions=False """
        if count(t1_c, prev_3) == 2:
            changed_directions = False
        # logger.print("pre", prev_3, count(t1_c, prev_3))

        prev_3.append(t2_c)  # aenderungen
        if len(prev_3) == 5:
            prev_3 = prev_3[1::]

    if reversed_vein_method:
        # alle arterien umdrehen
        for vessel in vessels:
            if "al" in vessel.associated_vesselname:
                flip_richtung(vessel)


def links_at(vessel, index):
    out = []
    for link in vessel.links_to_path:
        if link.source_index == index:
            out.append(link)
    return out


def no_links_to(vessel, vessels):
    for v in vessels:
        for link in get_traversable_links(v):
            if link.target_vessel == vessel:
                return False
    return True


def get_traversable_links(new_vessel, entry=0, all_endpoints=False):
    """Get traversable links for a vessel, based on if its classified as artery or vein

    Args:
        new_vessel (Vessel): Vesselobject which has an amount of associated links

    Returns:
        list of Link: traversable links of vessel
    """
    if new_vessel.type == "vein":
        try:
            if all_endpoints:
                return links_at(new_vessel, len(new_vessel.path) - 1)
            else:
                if new_vessel.highest_link() != None:
                    return [new_vessel.highest_link()]

        except:
            # logger.print("vein has no links or wrong direction")
            return []
        return []
    else:
        return new_vessel.next_links(entry)


def get_traversion_regions(vessels, all_endpoints=False):
    """
    returns collection of traversable vessels according to used traversal method TODO parameterize traversal method
    """
    traversables = [
        [link.target_vessel for link in get_traversable_links(svessel, all_endpoints)]
        + [svessel]
        for svessel in vessels
    ]
    # logger.print("trav",traversables)
    part_in = lambda array, inarray: any([part in array for part in inarray])
    add_unadded = lambda inp, array: [
        array.append(data) for data in inp if (data not in array)
    ]
    result = []
    while len(traversables) > 0:
        while len(traversables[0]) == 0:
            traversables.remove(traversables[0])
            # logger.print("rem")
            if len(traversables) == 0:
                break
        if len(traversables) == 0:
            break
        collect = traversables[0]
        for i, item in enumerate(traversables):
            if part_in(collect, item):
                add_unadded(item, collect)
                traversables[i] = []
        result.append(collect)

    change = True
    while change:
        change = False
        for part in result:
            t = part
            for i, partb in enumerate(result):
                if partb is not t:
                    if part_in(partb, part):
                        add_unadded(partb, part)
                        result.remove(partb)
                        change = True
    return result


def change_directions(vessels):
    """
    changes the direction of vessels if they are not alone in one traversal region and another region gets bigger if flipped TODO vergleich mit anderer methode, testschreiben
    """
    trav = get_traversion_regions(vessels)
    # find single outer vessel
    not_only_singles = any([len(part) > 1 for part in trav])
    # logger.print("NOT ONLY SINGLES", not_only_singles)
    for part in trav:
        if (
            len(part) == 1 and not_only_singles
        ):  # ist nur ein gross und es gibt groessere
            flip_richtung(part[0])
            # hat 2 links und nur einer ist mehrfach genommen
        # richtungen traversierbarere vesseln mit den anderen vergleichen und jene vesseln mit 2 links umkehren
    for part in trav:
        for vessel in part:
            if "safe_dir" not in vessel.tags:
                trav = get_traversion_regions(vessels)
                vessel_has_both_ends_linked = all(
                    [
                        any([link.source_index == 0 for link in vessel.links_to_path]),
                        any(
                            link.source_index == len(vessel.path) - 1
                            for link in vessel.links_to_path
                        ),
                    ]
                )
                # logger.print("vessel has both ends", vessel_has_both_ends_linked)
                if vessel_has_both_ends_linked:
                    flip_richtung(vessel)
                    # logger.print("check",len(part))
                    # wenn das eine neue region erschaffen hat: reverse
                    t2 = get_traversion_regions(vessels)
                    if len(t2) > len(trav):
                        flip_richtung(vessel)


def flip_richtung(vessel):
    """
    flips the direction of a vessel, throws error if no direction was set prior to this operation
    """
    if "safe_dir" not in vessel.tags:
        vessel.reverse(static_links=False)


def richtungsvektor(vessel):
    """
    returns a vector that points from the beginning of the vessels path to its end
    """
    rv = norm_vector(
        np.asarray(vessel.path[len(vessel.path) - 1]) - np.asarray(vessel.path[0])
    )
    return rv


def closest_point_in_path_index(
    vessel, point
):  # TODO rename get_closest_index_too_point
    """
    returns the closest pathpoint to the given point for a given vessel
    """
    d = scipy.spatial.distance.cdist([point], vessel.path, "euclidean")
    return np.where(d == d.min())[1][0]


def lookup_vessel_from_index(boundaries, index):
    """finds vessel from array that holds which values in athother data structure belong to which object and return sindex and object"""
    # get vessel from index k:
    result_vessel_info = None
    for vessel_info in boundaries:
        if vessel_info[0] <= index:  # letzte boundary, die kleiner oder gleich ist
            result_vessel_info = vessel_info
        else:
            break
    return [index - result_vessel_info[0], result_vessel_info[1]]


# def find_points_near_mesh(vessel, points):
#     """
#     returns one if a point is in 1.2 times the radius of the given vessel
#     """
#     d = scipy.spatial.distance.cdist(vessel.path, points, "euclidean")
#     out = set(np.where(d <= vessel.avg_diameter * 1.2)[1])
#     return [1 if i in out else -1 for i, k[i] in enumerate(k)]


# alternative Methode zum Verbinden, setzt voraus, dass das ende der vessel einen punkt besitzt.


def get_vessel_endpoints_per_mesh(vessel, i, vessel_endpoints, vessel_boundaries):
    """
    returns endpoints that are inside of a vessels mesh
    """
    midpoint = calculate_centered_point(vessel.path, 1)
    distance_to_midpoint = np.asarray(
        [
            calculate_distance_euclidian(endpoint, midpoint)
            for endpoint in vessel_endpoints
        ]
    )
    selection = vessel_endpoints[
        [x <= vessel.length * 1.2 for x in distance_to_midpoint]
    ]

    if len(selection) > 0:
        select_indices = [
            i
            for i, x in enumerate(
                x <= vessel.length * 1.2 for x in distance_to_midpoint
            )
            if x == True
        ]
        # logger.print(selection, "<- select")
        vessel_end_hits = np.asarray(
            trimesh.proximity.signed_distance(vessel.submesh, selection)
        )  # trimesh.proximity.signed_distance(vessel.submesh,selection )
        # vessel_end_hits=[select_indices[i] for i, x in enumerate(x >= 0 for x in vessel_end_hits) if x == True]
        vessel_end_hits = [select_indices[i] for i in np.where(vessel_end_hits >= 0)[0]]
        # von selection wieder zu allen kommen:

        vessels_to_link = [
            lookup_vessel_from_index(vessel_boundaries, hit)
            for hit in vessel_end_hits
            if hit not in [i * 2, i * 2 + 1]
        ]

        vessel_endpoints_in_vessel = [vessel.id]
        for vessel_info in vessels_to_link:
            vessel_b = vessel_info[1]
            b_index = 0
            if vessel_info[0]:
                b_index = len(vessel_b.path) - 1
            vessel_endpoints_in_vessel.append((vessel_b.id, b_index))
        return vessel_endpoints_in_vessel


def closest_point_in_path_index(
    vessel, point, index=False
):  # TODO compare to closest point path
    d = scipy.spatial.distance.cdist([point], vessel.path, "euclidean")
    # if index:return
    return np.where(d == d.min())[1][0]


def create_connection(vesselc, endpointkey, vessel_dict, connections, note="-"):
    """
    Creates the connection between two vessels
    """
    nearest_connection_point_index = closest_point_in_path_index(
        vesselc, vessel_dict[endpointkey[0]].path[endpointkey[1]]
    )
    connections.append(
        (
            vessel_dict[endpointkey[0]],
            Link(
                vessel_dict[endpointkey[0]],
                endpointkey[1],
                vesselc,
                nearest_connection_point_index,
                note,
            ),
            note,
        )
    )  # von ende zu naechster vessel
    connections.append(
        (
            vesselc,
            Link(
                vesselc,
                nearest_connection_point_index,
                vessel_dict[endpointkey[0]],
                endpointkey[1],
                note,
            ),
            note + "_back_link",
        )
    )  # zurueck


def endpoint_pair_relation(vesselendpunktkey, vessel, vessel_dict):
    """
    returns if a given vesselendpoint would connect to another vessels endpoint
    """
    cp = closest_point_in_path_index(
        vessel, vessel_dict[vesselendpunktkey[0]].path[vesselendpunktkey[1]]
    )
    # logger.print(cp)
    if cp == len(vessel.path) - 1 or cp == 0:
        return True, cp
    else:
        return False, -1


def clear_links(vessels):
    """
    delete all links from all given vessels
    """
    for vessel in vessels:
        vessel.links_to_path = []


def apply_links(connections):
    """
    apply all links from givven connections
    """
    for link in connections:
        app = link[
            1
        ]  # [int(link.target[0]),link.target[1],int(link.target[2]),link.target[3]]
        if (
            isinstance(app.target_index, np.generic)
            and not type(app.target_index) == int
        ):
            app.target_index = app.target_index.item()
        if (
            isinstance(app.source_index, np.generic)
            and not type(app.source_index) == int
        ):
            app.source_index = app.source_index.item()
        link[0].add_link(app)


def count_problematic_vessels(vessels):
    """
    counts 'problematic' vessels, that are not connected at end but between their ends (no regular entry) TODO used?
    """
    count_all = len(vessels)
    count_compl = 0
    count_any = 0
    count_nan = 0
    for i, vessel in enumerate(vessels):
        found = False
        # for endpointkey in endpoint_dict:
        if any(
            [
                link.source_index == 0 or link.source_index == len(vessel.path)
                for link in vessel.links_to_path
            ]
        ):  # vessel hat link an anfang oder ende
            found = True
            # if(len(endpoint_dict[endpointkey])>1):
            # logger.print( endpoint_dict[endpointkey])
            # if(vessel.id in endpoint_dict[endpointkey]):
            #  found=True
        if found == False:
            if len(vessel.links_to_path) > 0:
                count_any += 1
                logger.print(i)
            # logger.print("No endpoint found for vessel:", vessel.id)
            logger.print(vessel.links_to_path)
            count_nan += 1
        else:
            # logger.print("found")
            count_compl += 1
    logger.print(
        count_compl,
        "/",
        count_all,
        " with ",
        count_nan,
        " not connected at end: but in between ends:",
        count_any,
        "leaving: ",
        count_nan - count_any,
        " problems",
    )


def find_connected_structures(vessels):
    """
    returns connected parts of given vessel-collection
    """
    traversables = [
        [link.target_vessel for link in svessel.links_to_path] + [svessel]
        for svessel in vessels
    ]
    # logger.print("trav", traversables)
    part_in = lambda array, inarray: any([part in array for part in inarray])
    add_unadded = lambda inp, array: [
        array.append(data) for data in inp if (data not in array)
    ]
    result = []
    while len(traversables) > 0:
        while len(traversables[0]) == 0:
            traversables.remove(traversables[0])
            # logger.print("rem")
            if len(traversables) == 0:
                break
        if len(traversables) == 0:
            break
        collect = traversables[0]
        for i, item in enumerate(traversables):
            if part_in(collect, item):
                add_unadded(item, collect)
                traversables[i] = []
        result.append(collect)

    change = True
    while change:
        change = False
        for part in result:
            t = part
            for i, partb in enumerate(result):
                if partb is not t:
                    if part_in(partb, part):
                        add_unadded(partb, part)
                        result.remove(partb)
                        change = True
    return result


"""
    connected_parts=[[]]

    rest=vessels.copy()
    todo=[rest[0]]
    parts=0
    while(len(rest)>0):

        if(len(todo)==0):#neuer part
            todo=[rest[0]]
            parts=parts+1
            connected_parts.append([])
        if(todo[0] not in rest):
            todo.remove(todo[0])
            continue
        selected_vessel=todo[0]
        for link in selected_vessel.links_to_path:
            if(link.target_vessel in rest):
                todo.append(link.target)
        #logger.print(selected_vessel)
        rest.remove(selected_vessel)
        todo.remove(selected_vessel)
        connected_parts[parts].append(selected_vessel)
    #check ob parts zusammengelegt werden sollten
    finalparts=[]
    includes= lambda part, other_part: any([element in other_part for element in part])
    ind_insert=0
    while(len(connected_parts)>0):
        dele=[]
        finalparts.append([])
        #logger.print("finalparts", finalparts , finalparts[ind_insert])

        finalparts[ind_insert].extend(connected_parts[0])
        for part in connected_parts:
            if(part != connected_parts[0]):
                other_part=part
                if(includes(connected_parts[0],other_part)):
                    finalparts[ind_insert].extend(other_part)
                    dele.append(other_part)
                    logger.print("includes")

        dele.append(connected_parts[0])
        for part in dele:
            connected_parts.remove(part)
        ind_insert+=1
    return finalparts """


def get_stl_for_vesselcollection(vessels):
    """
    returns the associated meshnames for given vessels
    """
    names = []
    for vessel in vessels:
        if vessel.associated_vesselname not in names:
            names.append(vessel.associated_vesselname)
    return names


def find_vessel_biggest_diameter(vesselcollection):
    """
    returns vessel with biggest diameter from given collection
    """
    # logger.print(f"searching in {len(vesselcollection)} vessels")
    selected_vessel = None
    max_diameter = 0
    selected_vessel = None
    for vessel in vesselcollection:
        # logger.print("vessel",vessel.diameter_a, vessel.diameter_b )
        if vessel.diameter_a > max_diameter:
            selected_diameter = vessel.diameter_a
            selected_vessel = vessel
        elif vessel.diameter_b > max_diameter:
            selected_diameter = vessel.diameter_b
            selected_vessel = vessel
    # logger.print("returning", selectedvessel)
    # die ohne schnitte sind nan
    if selected_vessel is None:
        selected_vessel = vesselcollection[0]
    return selected_vessel


def generate_directions(
    connected_vessels, startvessels=None, reversed_vein_method=True
):
    """
    TODO use
    generates first 'safe' directions (one end free vessels have flow toward their connected end if vein, other way round else)
    """
    # veins
    # start has link but end not
    votees = []
    exists_link_at_beginning = lambda vessel: any(
        [link.source_index == 0 for link in vessel.links_to_path]
    )
    exists_link_at_end = lambda vessel: any(
        [link.source_index == len(vessel.path) - 1 for link in vessel.links_to_path]
    )
    list_without = lambda a, b: [x for x in a if x not in b]

    # finde erste schicht mit einem leeren ende
    vein_like = [
        vessel for vessel in connected_vessels if "vl" in vessel.associated_vesselname
    ]
    artery_like = [
        vessel for vessel in connected_vessels if "al" in vessel.associated_vesselname
    ]

    logger.print(get_stl_for_vesselcollection(vein_like))

    # flussrichtung gemaess "venenfluss"
    set_vessels = []
    logger.print(len(connected_vessels), len(vein_like))
    iteration_vessels = []
    i = 0
    for vessel in vein_like:
        if exists_link_at_beginning(vessel) and not exists_link_at_end(vessel):
            vessel.reverse(static_links=False)  # zum link hin
            vessel.tags.append("safe_dir")
            iteration_vessels.append(vessel)
            set_vessels.append(vessel)
            i += 1
            logger.print(i)
        if not exists_link_at_beginning(vessel) and exists_link_at_end(vessel):
            # zum link hin
            vessel.tags.append("safe_dir")
            iteration_vessels.append(vessel)
            set_vessels.append(vessel)
            i += 1
            # logger.print(i)
    remaining_vessels = list_without(vein_like, set_vessels)

    previouslen = len(remaining_vessels) + 1
    iteration = -1
    later_iterations = []
    link_at_beginning_directed = lambda vessel: any(
        [
            link.source_index == 0 and link.target_vessel.richtung != "nan"
            for link in vessel.links_to_path
        ]
    )
    link_at_end_directed = lambda vessel: any(
        [
            link.source_index == len(vessel.path) - 1
            and link.target_vessel.richtung != "nan"
            for link in vessel.links_to_path
        ]
    )
    while len(remaining_vessels) > 0 and len(remaining_vessels) < previouslen:
        set_vessels = []
        later_iterations.append([])
        iteration += 1
        previouslen = len(remaining_vessels)
        for vessel in remaining_vessels:
            # TODO check ob beide seiten einen link haben!
            if link_at_beginning_directed(vessel) and not link_at_end_directed(vessel):
                # vom link aus weiter
                set_vessels.append(vessel)
                later_iterations[iteration].append(vessel)
            elif link_at_end_directed(vessel) and not link_at_beginning_directed(
                vessel
            ):  # eine linkseite hat bereits eine richtung, TODO evtl check ob die andere nicht
                vessel.reverse(static_links=False)
                set_vessels.append(vessel)
                later_iterations[iteration].append(vessel)
            else:
                # logger.print(vessel, iteration, "has alreadey link at 0!")
                ends_indices = [
                    i
                    for i, link in enumerate(vessel.links_to_path)
                    if (
                        (link.source_index == 0)
                        and link.target_vessel.richtung != "nan"
                    )
                    or (
                        link.source_index == len(vessel.path) - 1
                        and link.target_vessel.richtung != "nan"
                    )
                ]
                # logger.print("ends", ends_indices, vessel.links_to_path, len(vessel.path))

                # logger.print(vessel.links_to_path[ends_indices[0]][1].richtung, vessel.links_to_path[ends_indices[1]][1].richtung)
        remaining_vessels = list_without(remaining_vessels, set_vessels)
    # all reamaining ramaining vains dont build an alternative way and their fluid direction will ge gained by comparing which general vector is most similiar with the incoming vectors
    ll = []
    for vein in remaining_vessels:
        vote_for_direction_vein(vein, connected_vessels)
        ll.append(vein)

    if not reversed_vein_method:
        if len(artery_like) > 0:
            if startvessels == None:
                logger.print(
                    "search for starvessel", find_vessel_biggest_diameter(artery_like)
                )
                start_vessels = [find_vessel_biggest_diameter(artery_like)]
                # set initial direction for startvessel (biggest)
                start_vessels[0]
                # finde richtung raus:
                if start_vessels[0].diameter_a < start_vessels[0].diameter_b:
                    # gehe in richtung kleinerer gefaesse
                    # iteriere nach unten wenn unten kleiner
                    start_vessels[0].reverse(static_links=False)
            # arterylike

            logger.print(start_vessels)
            directed_arteries = {}
            for art_vessel in start_vessels:
                directed_arteries[art_vessel.id] = []
            logger.print(start_vessels)
            for art_vessel in start_vessels:
                # von startvesseln aus in alle richtungen stroemen
                for link in art_vessel.get_traversable_links():
                    if link.target_index == 0:
                        ...
                    else:
                        link.target_vessel.reverse(static_links=False)
                    directed_arteries[art_vessel.id].append(link.target)
    else:
        # wie venen nur andersrum
        set_vessels = []
        logger.print(len(connected_vessels), len(artery_like))
        i = 0
        for vessel in artery_like:
            if exists_link_at_beginning(vessel) and not exists_link_at_end(vessel):
                vessel.reverse(static_links=False)  # zum link hin
                vessel.tags.append("safe_dir")
                iteration_vessels.append(vessel)
                set_vessels.append(vessel)
                i += 1
                logger.print(i)
            if not exists_link_at_beginning(vessel) and exists_link_at_end(vessel):
                # zum link hin
                vessel.tags.append("safe_dir")
                iteration_vessels.append(vessel)
                set_vessels.append(vessel)
                i += 1
                # logger.print(i)
        remaining_vessels = list_without(artery_like, set_vessels)

        previouslen = len(remaining_vessels) + 1
        iteration = -1
        link_at_beginning_directed = lambda vessel: any(
            [
                link.source_index == 0 and link.target_vessel.richtung != "nan"
                for link in vessel.links_to_path
            ]
        )
        link_at_end_directed = lambda vessel: any(
            [
                link.source_index == len(vessel.path) - 1
                and link.target_vessel.richtung != "nan"
                for link in vessel.links_to_path
            ]
        )
        while len(remaining_vessels) > 0 and len(remaining_vessels) < previouslen:
            set_vessels = []
            later_iterations.append([])
            iteration += 1
            previouslen = len(remaining_vessels)
            for vessel in remaining_vessels:
                # TODO check ob beide seiten einen link haben!
                if link_at_beginning_directed(vessel) and not link_at_end_directed(
                    vessel
                ):
                    # vom link aus weiter
                    set_vessels.append(vessel)
                    later_iterations[iteration].append(vessel)
                elif link_at_end_directed(vessel) and not link_at_beginning_directed(
                    vessel
                ):  # eine linkseite hat bereits eine richtung, TODO evtl check ob die andere nicht
                    vessel.reverse(static_links=False)
                    set_vessels.append(vessel)
                    later_iterations[iteration].append(vessel)
                else:
                    # logger.print(vessel, iteration, "has alreadey link at 0!")
                    ends_indices = [
                        i
                        for i, link in enumerate(vessel.links_to_path)
                        if (
                            (link.source_index == 0)
                            and link.target_vessel.richtung != "nan"
                        )
                        or (
                            link.source_index == len(vessel.path) - 1
                            and link.target_vessel.richtung != "nan"
                        )
                    ]
                    # logger.print("ends", ends_indices, vessel.links_to_path, len(vessel.path))

                    # logger.print(vessel.links_to_path[ends_indices[0]][1].richtung, vessel.links_to_path[ends_indices[1]][1].richtung)
            remaining_vessels = list_without(remaining_vessels, set_vessels)
        # all reamaining ramaining vains dont build an alternative way and their fluid direction will ge gained by comparing which general vector is most similiar with the incoming vectors
        for art in remaining_vessels:
            vote_for_direction_vein(art, connected_vessels)
            ll.append(art)
        """ for art in artery_like:
            if(art.richtung=="ASC"):
                art.richtung="DESC"
            else:
                art.richtung="ASC" """

    logger.print("OUT", len(vein_like))
    return (
        iteration_vessels,
        votees,
        later_iterations,
        ll,
    )  # iteration vessels are safe vessels from first one, later iterations are not safe from link directions


def vote_for_direction_vein(vessel, allvessels, method=0):
    """
    votes for the flowdirection of a vessel based on the direction surrounding cells flow points toward the vessel
    """
    if method == 0:
        possible_direction_vectors = dict()
        possible_direction_vectors["keep"] = norm_vector(
            -np.asarray(vessel.path[0]) + np.asarray(vessel.path[len(vessel.path) - 1])
        )
        possible_direction_vectors["change"] = norm_vector(
            np.asarray(vessel.path[0]) - np.asarray(vessel.path[len(vessel.path) - 1])
        )

        problematics = []
        incoming_vectors = []
        for link in vessel.links_to_path:
            if link.source_index != 0 and link.source_index != len(vessel.path) - 1:
                linkvektor = norm_vector(
                    np.asarray(vessel.path[link.source_index])
                    - np.asarray(link.target_vessel.path[link.target_index])
                )
                incoming_vectors.append(linkvektor)

        # dot product to all vectors
        asc_dot = [
            dot_product(vector, possible_direction_vectors["keep"])
            for vector in incoming_vectors
        ]
        desc_dot = [
            dot_product(vector, possible_direction_vectors["change"])
            for vector in incoming_vectors
        ]
        # logger.print("adot", asc_dot, len(incoming_vectors))
        # logger.print("ddot",desc_dot, len(incoming_vectors))
        logger.print("problematics", problematics)
        c_a_d = [value for value in asc_dot if value > 0]
        c_d_d = [value for value in desc_dot if value > 0]
        # direction of the one that shared most similarities with the other vectors
        if len(c_a_d) < len(c_d_d):
            vessel.reverse(static_links=False)
            logger.print("reversed", c_a_d, len(c_a_d), c_d_d, len(c_d_d))
        elif len(c_a_d) == len(c_d_d):
            if sum(c_a_d) < sum(c_d_d):
                vessel.reverse(static_links=False)
                logger.print("reversed on ==", c_a_d, len(c_a_d), c_d_d, len(c_d_d))

    else:
        # alternative: minimiere anzahl transversiebarer regioenen!
        trav = get_traversion_regions(allvessels)
        lenasc = len(trav)
        logger.print([len(p) for p in trav])
        vessel.reverse(static_links=False)
        trav = get_traversion_regions(allvessels)
        logger.print("trav:", [len(p) for p in trav])
        lendesc = len(trav)
        logger.print(lendesc, lenasc)
        if lenasc > lendesc:
            return
        if lenasc < lendesc:
            vessel.reverse(static_links=False)
            return


def all_link_points(vessel):
    """
    return outgoing points for all links in vessel
    """
    points = []
    for link in vessel.links_to_path:
        points.append(vessel.path[link.source_index])
    return points


# def closest_vein(point, organ_name):
#     """
#     returns the closes vein for a given organ mesh TODO test, used
#     """
#     organ_mesh = trimesh.load(organ_name)
#     one_side_free_vessels = []
#     for vessel in vessels:
#         # vessels with one unconnected end:
#         outer_value = 0
#         inner_value = len(vessel.path) - 1
#         if vessel.get_direction_by_assoc_name() == "ASC":
#             outer_value = len(vessel.path) - 1
#             inner_value = 0
#         unlinked = True
#         for link in vessel.links_to_path:
#             if link.source_index == outer_value:
#                 unlinked = False
#                 break
#         if unlinked:
#             one_side_free_vessels.append([vessel, outer_value])

#     # find out in which organs those vessels end
#     vessel_end_hits = np.asarray(
#         trimesh.proximity.signed_distance(
#             organ,
#             [
#                 vessel_plus_index[0].path[vessel_plus_index[1]]
#                 for vessel_plus_index in one_side_free_vessels
#             ],
#         )
#     )  # trimesh.proximity.signed_distance(vessel.submesh,selection )
#     # vessel_end_hits=[select_indices[i] for i, x in enumerate(x >= 0 for x in vessel_end_hits) if x == True]
#     # vessel_end_hits=[select_indices[i] for i in np.where(vessel_end_hits>=0)[0]]
#     vessels_organ_dist[i] = vessel_end_hits
#     vessels_in_organ[i] = [
#         one_side_free_vessels[i] for i in np.where(vessel_end_hits >= 0)[0]
#     ]


def distance_to_nearest_vein(point, veinentries):
    """
    retirms the minimum distance between point ans points in veinentries
    """
    return min([calculate_distance_euclidian(point, vpoint) for vpoint in veinentries])


"""
def calculate_linkchances(linked_vessels):

    #calculates initial linkchances based on diameter

    for vessel in linked_vessels:
        if "jump_vessel" in vessel.associated_vesselname:
            vessel.avg_diameter = 200
            vessel.diameter_a = 200
            vessel.diameter_b = 200

    for vessel in linked_vessels:
        # create linkchances: all chances shall add up to 100%
        if len(vessel.links_at_index(vessel.last_index())) == 0:
            # vessel ends without connecting to another vessel at its end in traversaldirection
            # create internal link after the last link in traversaldirection
            if vessel.richtung == "ASC":
                if vessel.get_highest_link() != None:
                    self_index = (
                        vessel.get_highest_link().source_index
                    )  # 0 llink, 1 index
                    vessel.add_link(Link(self_index, vessel, self_index + 1))
            elif vessel.richtung == "DESC":
                if vessel.get_lowest_link() != None:
                    self_index = (
                        vessel.get_lowest_link().source_index
                    )  # 0 llink, 1 index
                    vessel.add_link(Link(self_index, vessel, self_index - 1))

        all_outgoing_diameters = 0
        link_chance_diameters = []
        for link in vessel.links_to_path:
            # add diameters, smalles between linkedvessel and linkingvessel at linkposition (approx. with avg, a and b)
            if vessel.lookup_index(link) in range(
                0, len(vessel.path) // 10
            ):  # avg dia a or b
                dia_internal = vessel.diameter_a
            elif vessel.lookup_index(link) in range(
                len(vessel.path) - len(vessel.path) // 10, len(vessel.path)
            ):  # avg dia a or b
                dia_internal = vessel.diameter_a
            else:
                dia_internal = vessel.avg_diameter
            dia = dia_internal
            if link.target_vessel.avg_diameter < dia:
                dia = link.target_vessel.avg_diameter
            all_outgoing_diameters += dia
            link_chance_diameters.append(dia)
        logger.print(
            "vessel", vessel.associated_vesselname, "diameters", all_outgoing_diameters
        )
        # set likchances according to ratio of total outgoing diameters
        for i, dia in enumerate(link_chance_diameters):
            vessel.links_to_path[i].chance = dia / all_outgoing_diameters
 """

# ----------------------------- travel functions --------------------------------


def get_speed(list_of_vesselobjects):
    """speed_function Calculates speed of a cell in a vessel

    Args:
        distance (float): distance the cell travelled in mm
        list_of_vesselobjects (list): list of vesselobjects that the cell travelled in

    Returns:
        float: number of seconds it took to travel
    """
    x_speed = [0.01, 8, 25]
    f_speed = [0.5, 100, 2500]
    # interpolieren
    diameter = sum(vessel.avg_diameter for vessel in list_of_vesselobjects)
    diameter = diameter / len(list_of_vesselobjects)
    return np.interp(diameter, x_speed, f_speed)


def standard_speed_function(
    distance, list_of_vesselobjects=None, volume=None, multiplier=1
):
    """speed_function Calculates speed mm/s of a cell in a vessel

    Args:
        distance (float): distance the cell travelled in mm
        list_of_vesselobjects (list): list of vesselobjects that the cell travelled in

    Returns:
        float: number of seconds it took to travel
    """
    if list_of_vesselobjects == None or len(list_of_vesselobjects) == 0:
        speed = settings.CAPILLARY_SPEED * multiplier
        # if in_lung:
        #   speed = 0.5 * settings.CAPILLARY_SPEEDUP
        if volume is not None:
            if type(volume.get_symval(volume.Q_1)) == sympy.Symbol:
                assert False
            speed_m = volume.get_symval(volume.Q_1) / volume.A  # in m/s
            speed = speed_m * 1000  # mm/s
    else:
        vessel = list_of_vesselobjects[0]

        if vessel.type == "vein":
            x_speed = [0.008, 0.02, 5, 30]
            f_speed = [1, 2, 100, 380]
        else:
            x_speed = [0.008, 0.05, 4, 25]  # durchmesser 25 eig aorta
            f_speed = [1, 50, 450, 480]
        # interpolieren
        diameter = sum(vessel.avg_diameter for vessel in list_of_vesselobjects)
        diameter = diameter / len(list_of_vesselobjects)
        speed = np.interp(diameter, x_speed, f_speed)
    return distance / speed


def travel_distance(pointarray):
    return [
        calculate_distance_euclidian(point, pointarray[i + 1])
        for i, point in enumerate(pointarray)
        if (i < len(pointarray) - 1)
    ]


def create__time_coord_output(pointarray, vessels, previous_timevalue):
    if len(pointarray) == 0 or type(pointarray) is None:
        return []
    logger.print("-----", len(pointarray), vessels, previous_timevalue)
    dist = travel_distance(pointarray)
    speed = get_speed(vessels)
    times = [dist / speed for dist in dist]
    # accumulated_times
    prev_val = previous_timevalue
    for i in range(len(times)):
        times[i] = prev_val + times[i]
        prev_val = times[i]
    return times


def travel_from(vessel, index):
    """Function to generate travel route for a cell dropped in vessel at index, returns [cell_travel_route, cell_travel_distance, cell_travel_time]"""
    cell_current_vessel = vessel
    cell_travel_distance = 0
    cell_travel_time = 0
    cell_travel_index = index
    cell_travel_route = []  # np.asarray(vessel.path[cell_travel_index])]
    cell_travel_times = []
    vesselcount = 0
    while moved_cell := True:
        logger.print(cell_current_vessel.associated_vesselname)
        travel_result_for_vessel = cell_current_vessel.travel(
            cell_travel_index, None, standard_speed_function
        )
        # logger.print(travel_result_for_vessel)
        vesselcount += 1
        cell_travel_route.extend(travel_result_for_vessel[0])
        cell_travel_distance += travel_result_for_vessel[1]
        cell_travel_time += travel_result_for_vessel[2]
        prev_travel_time = cell_travel_times[-1] if cell_travel_times else 0
        cell_travel_times += create__time_coord_output(
            travel_result_for_vessel[0], [cell_current_vessel], prev_travel_time
        )

        logger.print(travel_result_for_vessel[4])

        if (
            travel_result_for_vessel[4] == False or vesselcount >= 1000
        ):  # keine bewegung und es ist nicht die erste iteration
            moved_cell = False
            break

        next_index = travel_result_for_vessel[5]
        # add linkdistance and time
        link_distance = calculate_distance_euclidian(
            cell_current_vessel.path[cell_travel_index],
            travel_result_for_vessel[3].path[next_index],
        )
        cell_travel_distance += link_distance

        prev_travel_time = cell_travel_times[-1] if cell_travel_times else 0
        # logger.print("lower prev_travel_time: ",prev_travel_time)
        # logger.print("ctttt",cell_travel_times)
        cell_travel_times += create__time_coord_output(
            [
                cell_current_vessel.path[cell_travel_index],
                travel_result_for_vessel[3].path[next_index],
            ],
            [cell_current_vessel, travel_result_for_vessel[3]],
            prev_travel_time,
        )
        cell_travel_time += link_distance / get_speed(
            [cell_current_vessel, travel_result_for_vessel[3]]
        )

        cell_travel_index = next_index
        cell_current_vessel = travel_result_for_vessel[3]
    return [
        cell_travel_route,
        cell_travel_distance,
        cell_travel_time,
        cell_travel_times,
    ]


def insert_unique_values_in_list(triple, output):
    """Inserts a triplets index values into an output list of indices at every mention of the value and doesnt insert a value multiple times"""
    for i, j in itertools.product(range(3), range(3)):
        if triple[j] not in output[triple[i]]:
            output[triple[i]].append(triple[j])


def read_neighbours(array_of_subarrays, length):
    """Reads the neighbours (values that are grouped together in subarrays) from an array/list that contains multiple values per cell"""
    output = [[] for _ in range(length)]
    for triple in array_of_subarrays:
        # alle nb die noch nicht an dem index der vorkommt stehen eintragen
        insert_unique_values_in_list(triple, output)
    return output


# Stelle verbundene Strukturen im Mesh fest, Parameter: Punktindex
def trace_shape_for_vertex_index(index, neighbourarray):
    """Iterate over all Neighbours from one index of a vertex and output the detected connected structure"""
    shape = {index}
    search = [index]
    searched = []
    while search:
        for connectedPoint in neighbourarray[search[0]]:
            shape.add(connectedPoint)
            if connectedPoint not in search and connectedPoint not in searched:
                search.append(connectedPoint)
        searched.append(search[0])
        search.remove(search[0])
    return list(shape)


def count_empty_cells(arr):
    """counts all non-empty cells in an array or list and outputs the count and index of first non-empty entry in array"""
    count = 0
    firsthit = False
    fIndex = 0
    for i in range(len(arr)):
        if len(arr[i]) > 0:
            count += 1
            if firsthit == False:
                firsthit = True
                fIndex = i
    # logger.print(str(count) + "  index: " + str(fIndex))
    return [count, fIndex]


def get_mesh_vertices_by_indices(indices, mesh):
    """returns a list of points from a given list of indices corresponding to points mesh vertices"""
    return [mesh.vertices[index] for index in indices]


def max_distance(pointA, pointlist):
    """calculates the maximum distance between one point and a list of points and returns the
    distance plus the original point and the furthest point from it"""
    q = cdist([pointA], pointlist, metric="euclidean")
    b = q.max()
    return [b, pointlist[np.where(q == b)[1][0]], pointA]


def highestNeighbourCount(arrayOfIndices, neighbourArray):
    """calculates which point has the highest count of neighbours"""
    # Test auf nArray len is lower than highest index
    if max(arrayOfIndices) > len(neighbourArray):
        logger.print("more indices than nArrayLength")
        return []
    # no negative indices
    if min(arrayOfIndices) < 0:
        logger.print("negative indices in indexarray")
        return []

    maxlength = 0
    # laenge rausfinden
    for index in arrayOfIndices:
        if len(neighbourArray[index]) > maxlength:
            maxlength = len(neighbourArray[index])

    return [
        index for index in arrayOfIndices if len(neighbourArray[index]) == maxlength
    ]


def updateDistancesToStartPoint(pointsToDo, endpunkt, mesh):
    """ """
    return [
        [
            calcDistance(
                get_mesh_vertices_by_indices([punktinfo[1]], mesh)[0], endpunkt
            ),
            punktinfo[1],
        ]
        for punktinfo in pointsToDo
    ]


def getEndpunkteByID(a, shapeEndPunkte):
    """ """
    for element in shapeEndPunkte:
        if element[0] == id(a):
            return element


def calcDistance(pointA, pointB):
    """calculates the distance between two triplets that represent 3D points"""
    return math.sqrt(
        (pointA[0] - pointB[0]) * (pointA[0] - pointB[0])
        + (pointA[1] - pointB[1]) * (pointA[1] - pointB[1])
        + (pointA[2] - pointB[2]) * (pointA[2] - pointB[2])
    )


def calculate_centered_point(points, method, plane_position=None, plane_normal=None):
    """ """
    if method == 0:  # centroid
        total = 0
        x = [0, 0, 0]
        for point in points:
            x[0] = x[0] + point[0]
            x[1] = x[1] + point[1]
            x[2] = x[2] + point[2]
            total = total + 1
        return [x[0] / total, x[1] / total, x[2] / total]
    elif method == 1:  # bounding box center
        min_x = points[0][0]
        max_x = points[0][0]
        min_y = points[0][1]
        max_y = points[0][1]
        min_z = points[0][2]
        max_z = points[0][2]
        for point in points:
            if point[0] < min_x:
                min_x = point[0]
            if point[0] > max_x:
                max_x = point[0]
            if point[1] < min_y:
                min_y = point[1]
            if point[1] > max_y:
                max_y = point[1]
            if point[2] < min_z:
                min_z = point[2]
            if point[2] > max_z:
                max_z = point[2]
        return [
            min_x + ((max_x - min_x) / 2),
            min_y + ((max_y - min_y) / 2),
            min_z + ((max_z - min_z) / 2),
        ]
    elif method == 2:  # ellipsis centre
        if len(points) < 6:
            # no fitting possible
            return calculate_centered_point(points, 0)
        assert (
            plane_position is not None and plane_normal is not None
        ), "Need to pass plane data for ellipsis fitting and centre calculation"
        logger.print("pointlen for centre", len(points))
        return get_center_ellipsis(points, plane_position, plane_normal)
    else:
        logger.print("unknown method to calculate points!")


# Methode, um Kreise zu finden
def get_nearest_cirlce(point_on_shape, neighbourarray, mesh):
    # gehe durch nachbarn und finde diejenigen, die am
    # wenigsten weit weg sind
    #  (reicht schon bei langen blutgefaessen)
    # fuer kurze gefaesse: beziehung der vertexnormals?
    # finde kuerzeste strecke wobei priorisiert wird,
    # dass der abstand zum
    # referenzpunkt gleich bleibt (setzt vcoraus, dass
    # dieser relativ mittig
    #  zu dem punkt auf der shape ist)
    # hier auch andere nicht genommene pfade rein -> nur eine richtung
    # erster punkt darf nicht checkedNB markieren, da sonst keine
    # vollendung des kreises erfolgen kann

    suchpunkt = point_on_shape
    circle = [suchpunkt]
    checkedNB = []
    tempNBDist = []
    while circle.count(suchpunkt) == 1:  # bis anfang ist ende
        tempNBDist = []
        for nachbar in neighbourarray[suchpunkt]:
            # (nachbar != point_on_shape)):#and (nachbar not in checkedNB)
            if nachbar not in [circle[-1], circle[len(circle) - 2]]:
                checkedNB.append(nachbar)
                # logger.print("appended ", nachbar, " for ", suchpunkt)
                tempNBDist.append(
                    [
                        calcDistance(
                            get_mesh_vertices_by_indices([nachbar], mesh)[0],
                            get_mesh_vertices_by_indices([suchpunkt], mesh)[0],
                        ),
                        nachbar,
                    ]
                )
        # if(point_on_shape in neighbourarray[suchpunkt]):
        #    logger.print("HEOY", suchpunkt)
        # logger.print("sp:", suchpunkt, neighbourarray[suchpunkt], min(tempNBDist))
        circle.append(min(tempNBDist)[1])
        suchpunkt = min(tempNBDist)[1]
        # logger.print(tempNBDist)
        # logger.print("checked: ", checkedNB)
        # logger.print(circle)

    return circle


def same_counts(a1, a2):
    a1.sort()
    a2.sort()
    return a1 == a2


def closest_point_on_mesh(coordinates, mesh):
    """find the closest point index on the mesh from given coordinates"""
    return mesh.nearest.vertex(coordinates)


def closest_point_on_mesh_manual(coordinates, shape, mesh):
    """find the closest point index on the mesh from given coordinates"""
    min_distance = calculate_distance_euclidian(coordinates, mesh.vertices[shape[0]])
    result_point = shape[0]
    for point in shape:
        pointcoordinates = mesh.vertices[point]
        calculated_distance = calculate_distance_euclidian(
            pointcoordinates, coordinates
        )
        if calculated_distance < min_distance:
            logger.print("changed point")
            min_distance = calculated_distance
            result_point = point
    return result_point


def calculate_diameter(set_of_points):
    """calculate_diameter calculates the diameter precisely by comparing all point distances to each other
    Args:
        set_of_points (list): list of vectors that represent points of the same dimension

    Returns:
        float: biggest distance between the given points"""
    return cdist(set_of_points, set_of_points, metric="euclidean").max()


def old_calculate_diameter(set_of_points):
    diameter = -1
    for pointA in set_of_points:
        for pointB in set_of_points:
            temp = calcDistance(pointA, pointB)
            if temp > diameter:
                diameter = temp
    return diameter


def drop_cell_in_vessels(list_of_vesselobjects, drop_coordinates, speed_function):
    """Function to simulate the travel of a cell in a collection of interconnected Vessels"""
    # parametercheck
    assert type(list_of_vesselobjects) == list
    assert [type(x) == Vessel for x in list_of_vesselobjects]
    assert len(drop_coordinates) == 3
    assert [type(x) in [float, int] for x in drop_coordinates]

    # find nearest vessel and nearest point on its path
    vessel_mid_points = [
        [
            (vessel.path[len(vessel.path) - 1][0] + vessel.path[0][0]) / 2,
            (vessel.path[len(vessel.path) - 1][1] + vessel.path[0][1]) / 2,
            (vessel.path[len(vessel.path) - 1][2] + vessel.path[0][2]) / 2,
        ]
        for vessel in list_of_vesselobjects
    ]

    d = scipy.spatial.distance.cdist([drop_coordinates], vessel_mid_points, "euclidean")
    closest_vessel = list_of_vesselobjects[np.where(d == d.min())[1][0]]
    logger.print(closest_vessel)
    # finde naechsten punkt im pfad:
    dp = scipy.spatial.distance.cdist(
        [drop_coordinates], closest_vessel.path, "euclidean"
    )
    closest_point_index = np.where(dp == dp.min())[1][0]

    cell_current_vessel = closest_vessel
    cell_travel_distance = 0
    cell_travel_time = 0
    cell_travel_index = closest_point_index
    cell_travel_route = [closest_vessel.path[cell_travel_index]]
    while moved_cell := True:
        logger.print(cell_travel_index, cell_current_vessel)
        travel_result_for_vessel = cell_current_vessel.travel(cell_travel_index)
        logger.print(travel_result_for_vessel)

        cell_travel_route += travel_result_for_vessel[0]
        cell_travel_distance += travel_result_for_vessel[1]
        cell_travel_time += travel_result_for_vessel[2]

        # keine bewegung und es ist nicht die erste iteration
        if travel_result_for_vessel[4]:
            moved_cell = False
            continue

        next_index = travel_result_for_vessel[5]
        # add linkdistance and time
        link_distance = calculate_distance_euclidian(
            cell_current_vessel.path[cell_travel_index],
            travel_result_for_vessel[3].path[next_index],
        )
        cell_travel_distance += link_distance
        cell_travel_time += speed_function(
            link_distance, [cell_current_vessel, travel_result_for_vessel[3]]
        )

        cell_travel_index = next_index
        cell_current_vessel = travel_result_for_vessel[3]
    return [cell_travel_route, cell_travel_distance, cell_travel_time]


# this class moves the operation of finding outif a cell was in  part of the path of a vessel from caclulating on the output to using if statements and registering times in the logger this class represents
def no_takeable_link_between(part_entry, part_exit, vessel):
    outgoing_indices = [link.source_index for link in get_traversable_links(vessel)]
    no_between_links = True
    for index in outgoing_indices:
        if index < part_exit and index > part_entry:
            no_between_links = False
    return no_between_links


def all_possible_paths_marked(start_vessel, start_index, marks_vessel_index):
    """For a given start_vessel check if all traversable vessels are in makrs vessels and have an index that would get traversed

    Args:
        start_vessel (_type_): Vessel to search from
        start_index (_type_): index to search from
        marks_vessel_index (_type_): vessels and their indices that are searched for

    Returns:
        _type_: True, wenn jeder moegliche Weg mit einem Mark beschränkt ist, false sonst
    """
    if start_vessel == marks_vessel_index[0] and marks_vessel_index[1] >= start_index:
        return True
    tlinks = [
        link
        for link in get_traversable_links(start_vessel)
        if link.source_index >= start_index
    ]
    t_vessel_index = [(link.target_vessel, link.target_index) for link in tlinks]
    if len(t_vessel_index) == 0:
        return False
    return all(
        [
            all_possible_paths_marked(start_vessel, start_index, marks_vessel_index)
            for start_vessel, start_index in t_vessel_index
        ]
    )
