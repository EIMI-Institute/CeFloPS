import trimesh
import pickle
import os
import CeFloPS.simulation.settings as settings
from CeFloPS.simulation.common.blood_roi import Blood_roi
from CeFloPS.simulation.common.tissue_roi_parallel import (
    Tissue_roi,
    fit_capillary_speed,
)
import glob 
from joblib import Parallel, delayed
import CeFloPS.simulation.common.vessel_functions as f
import numpy as np
from CeFloPS.simulation.common.vessel2 import TissueLink
import scipy
from .common.shared_geometry import SharedGeometry
import sys
import yaml

# TODO permanent fix!
sys.modules["common"] = sys.modules["CeFloPS.simulation.common"]

from common.functions import *
from common.vessel_functions import *


def load_vessels(custompath=None):
    """load_vessels Load vessels from file specified in settings

    Returns:
        _type_: _description_
    """
    path = settings.PATH_TO_VESSELS
    if custompath is not None:
        path = custompath
    print("Loading vessels from", path)

    with open(
        path,
        "rb",
    ) as input_file:
        vessels, vois_preset = pickle.load(input_file)
    try:
        for vessel in vessels:
            vessel.register_volumes()
    except:
        ...
    return vessels, vois_preset


def load_concentration(location):
    """load_concentration Load concentration from txt file

    Args:
        location (_type_): _description_

    Returns:
        _type_: _description_
    """
    out = []
    with open(location) as f:
        for line in f:
            out.append(line.split("\n")[0])
    out = out[1::]
    for i in range(len(out)):
        out[i] = float(out[i])
    return out


def subtract_voxels(voxelized_mesh_a, voxelized_mesh_b):
    """subtract_voxels Given 2 voxelGrids subtract all voxels set in b that are also set in a from a

    Args:
        voxelized_mesh_a (_type_): _description_
        voxelized_mesh_b (_type_): _description_

    Returns:
        _type_: _description_
    """
    subtracted_count = 0
    # voxelized_mesh_a gets all overlapping voxels from mesh b removed
    assert type(voxelized_mesh_a) == trimesh.voxel.base.VoxelGrid
    assert type(voxelized_mesh_b) == trimesh.voxel.base.VoxelGrid

    # filled voxels from grids
    filled_a = np.argwhere(voxelized_mesh_a.matrix)
    filled_b = np.argwhere(voxelized_mesh_b.matrix)

    # quick lookup set
    filled_b_set = set(map(tuple, filled_b))

    # Iterate over filled voxels in mesh a and remove if they exist in mesh b
    for voxel in filled_a:
        if tuple(voxel) in filled_b_set:
            # Set the voxel to False (unfilled) in the matrix
            voxelized_mesh_a.matrix[tuple(voxel)] = False
            subtracted_count += 1
    voxel_volume = settings.ROI_VOXEL_PITCH**3
    return subtracted_count * voxel_volume


def create_mesh(name, negative_voxelgrids=None):
    """create_mesh Load mesh and according to rules (hollow) create voxelgrids and subtract negative grids

    Args:
        name (_type_): _description_
        negative_voxelgrids (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    print("create Mesh for", name)
    mesh = trimesh.load(name)
    meshes = mesh.split()
    output = []
    for mesh in meshes:
        centre = mesh.centroid
        volume = mesh.volume  # in mm^3
        geometry = mesh.voxelized(settings.ROI_VOXEL_PITCH).fill()

        if any([identifier in name for identifier in settings.ROI_2_5D]):
            geometry = geometry.hollow()

        # remove negative spaces
        if negative_voxelgrids != None:
            sub_volume = 0
            print(name + " loaded, checking for subtraction...")
            for negative_grid in negative_voxelgrids:
                sub_volume = subtract_voxels(geometry, negative_grid)
            print("volume, subtraction", volume, sub_volume)
            volume = volume - sub_volume  # also in mm^3
        print(name + " done!")
        output.append((name, np.asarray(geometry.points), volume, centre, mesh))
        if len(meshes) == 1:
            return name, np.asarray(geometry.points), volume, centre, mesh
    return output


def create_roi(name, meshdata, blood_roi, store_loc=False):
    """create_roi For meshdata generated wutg create´_mesh create a parallelTissue Object that holds k_values and points of the geometry

    Args:
        name (_type_): _description_
        meshdata (_type_): _description_
        blood_roi (_type_): _description_

    Returns:
        _type_: _description_
    """
    mesh = meshdata[0]
    volume = meshdata[1]
    centre = meshdata[2]
    print("create Roi for", name)
    roitype = 3
    string = "All"
    if any([ident in name for ident in settings.ROI_2_5D]):
        roitype = 2
        print("USE 2 for ", name)

    rate_constants = settings.f_settings.get_rate_constant(name)  # transition rates
    name = name  # used to identify a loaded voi
    k_name = settings.f_settings.get_identifier(name)  # used for transition chance
    roi_name = settings.f_settings.get_concentration_identifier(
        k_name
    )  # used for conecntration plots

    return (
        string,
        Tissue_roi(
            name[len(settings.PATH_TO_STLS) + 1 : :],
            name,
            roitype,
            rate_constants,
            blood_roi,
            mesh,
            volume,
            centre,
            k_name,  # =get_k_name(name),
            roi_name,
            store_loc=store_loc,  # =get_roi_name(name),
        ),
    )


def select_rois(all_rois, keywords):
    """select_rois Method to get rois by keyword filter

    Args:
        all_rois (_type_): _description_
        keywords (_type_): _description_

    Returns:
        _type_: _description_
    """
    selection = set()
    for roi in all_rois:
        if any([keyword in roi.name for keyword in keywords]):
            selection.add(roi)
            # all_rois.remove(roi)
    return list(selection)


import sympy


def create_vectorfield(roi):
    if roi.name != "blood":
        print("createing vectorfield for", roi.name)
        roi.create_flow_vectorfield(save=True)


def load_vois(
    blood_only=False,
    names_only=False,
    vessel_not_allowed_in_path=True,
    exclude_strings=[],
):
    """load_vois Load all meshfiles for simulation

    Args:
        blood_only (bool, optional): _description_. Defaults to False.
        names_only (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
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
        ):
            extravessels.append(name)
            to_remove.append(name)

    [all_files.remove(name) for name in to_remove]
    print(
        "remaining after excluding negspaces and vesselidents unspecified:",
        len(all_files),
    )

    vessels += extravessels
    if names_only:
        return all_files

    # fill the roi dictionary

    rois["Muscle"] = []
    rois["Lung"] = []
    rois["Liver"] = []
    rois["Grey_matter"] = []
    rois["Myocardium"] = []
    rois["Spleen"] = []
    rois["GUC_lesions"] = []
    rois["All"] = []
    # sort rois
    # define blood roi and subtract negative stls as well as vesselstls
    blood_roi = Blood_roi("blood", settings.PATH_TO_VESSELS, 1)
    voi_addition = blood_roi.additional_vois
    #blood_roi.additional_vois = None
    try:
        subs = settings.NEGATIVE_SPACES
        prefix = settings.PATH_TO_STLS
        subs_meshes = [trimesh.load(prefix + "/" + name) for name in subs]
        print(subs_meshes)
        if settings.SUBTRACT_VESSELMESHES:
            subs_meshes += [trimesh.load(path) for path in settings.VESSELPATHS]
    except Exception as e:
        print(f"Couldn't load negative Spaces, Error: {e}")
    negative_voxelgrids = [
        sub.voxelized(settings.ROI_VOXEL_PITCH).fill() for sub in subs_meshes
    ]
    print("negative spaces", negative_voxelgrids)

    # load meshes and map them to a region, load the rate constants and create rois, also filter that only specified geometries within all_files are loaded

    with Parallel(n_jobs=5) as para:
        identifiers_meshes = para(
            delayed(create_mesh)(name, negative_voxelgrids)
            for name in all_files
            if any([identifier in name for identifier in settings.identifiers])
        )
    print("loaded meshes!")
    # name, np.asarray(geometry.points), volume, centre
    identifiers_rois = [
        create_roi(identifiers_meshes[i][0], identifiers_meshes[i][1::], blood_roi)
        for i, name in enumerate(
            identifiers_meshes
        )  # if any([identifier in name for identifier in  ["adrenal", "intest", "throat", "stomach", "kidney", "Kidney", "pancreas", "Pancreas", settings.MUSCLE_IDENTIFIER , settings.LUNG_IDENTIFIER, settings.LIVER_IDENTIFIER, settings.GREY_MATTER_IDENTIFIER, settings.MYOCARDIUM_IDENTIFIER, settings.SPLEEN_IDENTIFIER]])
    ]

    for ident, roi in identifiers_rois:
        roi.blood_roi = blood_roi
        rois[ident].append(roi)
        print(ident, roi, id(roi.blood_roi))
        roi.recreate_compartments()

    roi_mapping = {
        "adrenal gland": select_rois(rois["All"], ["adrenal"]),  # adrenal
        "gastrointestinal tract": select_rois(
            rois["All"], ["intest", "throat", "stomach"]
        ),  # intest, throat, stomach
        "kidneys": select_rois(rois["All"], ["kidney", "Kidney"]),  # kidney
        "liver": rois["Liver"],  # liver
        "lungs": rois["Lung"],  # lung
        "muscle": rois["Muscle"],
        "pancreas": select_rois(rois["All"], ["pancreas", "Pancreas"]),  # pancreas
        "spleen": rois["Spleen"],
        "brain": rois[
            "Grey_matter"
        ],  # [rois["GUC_lesions"] + rois["Myocardium"] + rois["Grey_matter"]],#TODO dont load?
    }
    roi_mapping_str = {
        "adrenal gland": [
            roi.name for roi in select_rois(rois["All"], ["adrenal"])
        ],  # adrenal
        "gastrointestinal tract": [
            roi.name
            for roi in select_rois(rois["All"], ["intest", "throat", "stomach"])
        ],  # intest, throat, stomach
        "kidneys": [
            roi.name for roi in select_rois(rois["All"], ["kidney", "Kidney"])
        ],  # kidney
        "liver": [roi.name for roi in rois["Liver"]],  # liver
        "lungs": [roi.name for roi in rois["Lung"]],  # lung
        "muscle": [roi.name for roi in rois["Muscle"]],
        "pancreas": [
            roi.name for roi in select_rois(rois["All"], ["pancreas", "Pancreas"])
        ],  # pancreas
        "spleen": [roi.name for roi in rois["Spleen"]],
        "brain": [roi.name for roi in rois["Grey_matter"]],
        "blood": [blood_roi.name],
    }
    # ident roi.name

    final_rois = (
        roi_mapping["adrenal gland"]
        + roi_mapping["gastrointestinal tract"]
        + roi_mapping["kidneys"]
        + roi_mapping["liver"]
        + roi_mapping["lungs"]
        + roi_mapping["muscle"]
        + roi_mapping["pancreas"]
        + roi_mapping["spleen"]
        + roi_mapping["brain"]
    )

    print("simulate rois:", [roi.name for roi in final_rois])
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
    for v in problems:
        # removes.append(v)
        # print("removed not connected from: ", v.associated_vesselname)
        blood_roi.geometry.remove(v)
    # remove profiles as they cant be pickled
    for v in blood_roi.geometry:
        v.unregister_functions()  # as lambda functions cant be pickled
    print("blood_roi_vessels", len(blood_roi.geometry))

    # for every roi, check and set rois for vessel
    connect_vessels_to_vois(final_rois, blood_roi.geometry)
    if False:
        with Parallel(n_jobs=-2) as parallel:
            parallel(delayed(create_vectorfield)(roi) for roi in final_rois)
    else:
        if settings.USE_FLOW_WALK:
            for roi in final_rois:
                if "blood" not in roi.name:
                    # check if saved, else create
                    try:
                        with open(
                            settings.cached_flowfield_dir + "/" + roi.cache_name,
                            "rb",
                        ) as input_file:
                            vectorfield = pickle.load(input_file)
                        print("Successfully used cached field for", roi.name)
                        roi.vectorfield = SharedGeometry(vectorfield)
                        fit_capillary_speed([roi], save=True)
                        print(roi.vectorfield.get_points())
                    except:
                        print("Couldnt load vectorfield for", roi.name)
                        roi.create_flow_vectorfield(save=True)
                        fit_capillary_speed([roi], save=True)

    for roi in final_rois:
        if any(
            [ident in roi.name for ident in settings.PLOT_COMPARTMENTS_WITH_NAMEPART]
        ):
            roi.add_compartment_plotter()
    final_rois += voi_addition
    return final_rois, blood_roi, roi_mapping, roi_mapping_str
    # vf.set_rois_for_vessel(final_rois)


def get_closest_index_to_point(point, pointlist, return_distance=False):
    """get_closest_index_to_point Get closest point in pointlist and index

    Args:
        point (_type_): _description_
        pointlist (_type_): _description_
        return_distance (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    d = scipy.spatial.distance.cdist([point], pointlist, "euclidean")
    if return_distance:
        closest_index = np.where(d == d.min())[1][0]
        return closest_index, pointlist[closest_index]

    return np.where(d == d.min())[1][0]


def ends_in_mesh(vessel, mesh_tuple_set):
    """ends_in_mesh Return True if vesselend (artery ending or vein beginning) point lays in mesh tuple set-> vessel ends in mesh

    Args:
        vessel (_type_): _description_
        mesh_tuple_set (_type_): _description_

    Returns:
        _type_:
    """

    if vessel.type == "vein":
        point = vessel.path[0]
    else:
        point = vessel.path[-1]
    voxel_point = (
        rround(point[0], settings.ROI_VOXEL_PITCH),
        rround(point[1], settings.ROI_VOXEL_PITCH),
        rround(point[2], settings.ROI_VOXEL_PITCH),
    )
    return voxel_point in mesh_tuple_set


def store_vein_info(vein_index, vein_vessel, voi_object, voi_index):
    """store_vein_info store veinindex and next vein in voi object"""
    voi_object.outlets.append((vein_vessel, vein_index, voi_index))
    voi_object.veins.append(
        (
            vein_vessel.path[vein_index],
            vein_vessel,
            vein_index,
            voi_object.geometry.get_points()[voi_index],
        )
    )


def store_artery_info(artery_index, artery_vessel, voi_object, voi_index):
    """store_vein_info crate tlink at arteryindex to voiindex"""
    link_to_voi = TissueLink(artery_vessel, artery_index, voi_object, voi_index)
    voi_object.inlets.append((artery_vessel, artery_index, voi_index))
    artery_vessel.links_to_vois.append(link_to_voi)
    artery_vessel.reachable_rois.append((artery_index, voi_object))


def store_info(
    vessel_index, vessel, voi_object, voi_index, connected_vessels, roi_data_cache=None
):
    """store_info Automatically store artery or vein info

    Args:
        vessel_index (_type_): _description_
        vessel (_type_): _description_
        voi_object (_type_): _description_
        voi_index (_type_): _description_
    """
    if roi_data_cache != None:
        roi_data_cache[voi_object.name].append(
            (vessel_index, vessel.id, voi_object.name, voi_index)
        )
    connected_vessels.add(vessel)
    if vessel.type == "vein":
        store_vein_info(vessel_index, vessel, voi_object, voi_index)
    elif vessel.type == "artery":
        store_artery_info(vessel_index, vessel, voi_object, voi_index)
    else:
        assert False, "Vessel has to be artery or vein"


def get_vessel_by_id(id_searched, vessels):
    """get_vessel_by_id get vessel by id in list of vessels

    Args:
        id_searched (_type_): _description_
        vessels (_type_): _description_

    Returns:
        _type_: _description_
    """
    for vessel in vessels:
        if vessel.id == id_searched:
            return vessel


def get_voi_by_name(name_searched, final_rois):
    """get_voi_by_name Get voi by its name in list of vois

    Args:
        name_searched (_type_): _description_
        final_rois (_type_): _description_

    Returns:
        _type_: _description_
    """
    for voi in final_rois:
        if voi.name == name_searched:
            return voi


def restore_connections(
    list_of_4_tuples, vessels, final_rois, connected_vessels=set()
):  # this can be done as vessels is guarenteed to be the same if not renamed TODO make not changeable?
    """restore_connections Restore Connections between VOIs and vessels based on cached listdata

    Args:
        list_of_4_tuples (_type_): _description_
        vessels (_type_): _description_
        final_rois (_type_): _description_
    """
    for v_index, vessel, voi_object, voi_index in list_of_4_tuples:
        store_info(
            v_index,
            get_vessel_by_id(vessel, vessels),
            get_voi_by_name(voi_object, final_rois),
            voi_index,
            connected_vessels,
        )


def connect_vessels_to_vois(final_rois, vessels):
    """connect_vessels_to_vois Connect all innerlaying vessels, rest of vessels to fullfill bidirectional rule and rest with threshold

    Args:
        final_rois (_type_): _description_
        vessels (_type_): _description_
    """
    for vessel in vessels:
        vessel.links_to_vois = []
    roi_data_cache = dict(
        (roi.name, []) for roi in final_rois if "blood" not in roi.name
    )

    available_vessels = vessels.copy()
    connected_vessels = set()
    vesselends_in = dict()
    # sort rois by size:
    roi_arrays = [
        (roi, roi.geometry.get_points())
        for roi in final_rois
        if "blood" not in roi.name
    ]  # geometry is sorted tuplebased
    roi_arrays = sorted(roi_arrays, key=lambda l: len(l[1]))
    meshes = [set([tuple(e) for e in roi_array[1]]) for roi_array in roi_arrays]
    # meshes is set of voxelcoordinates with tuples as sets for fast lookups

    # check if all rois already have saved data caches:
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
    if use_cached:
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
        # create everything and save in files

        for m_index, mesh in enumerate(meshes):
            # len(l) is proportional to volume of mesh thus lower meshes get iterated over first
            to_add_to_mesh = []  # collect vessels that lay in mesh
            for vessel in available_vessels:
                if ends_in_mesh(vessel, mesh):  # end is in meshvoxels
                    # find which indices are to be connected
                    v = vessel
                    index_v = 0
                    if v.type != "vein":
                        index_v = len(v.path) - 1
                    index_m = get_closest_index_to_point(
                        v.path[index_v], roi_arrays[m_index][1]
                    )
                    to_add_to_mesh.append((v, index_v, index_m))
            for vessel, index_v, index_m in to_add_to_mesh:
                available_vessels.remove(vessel)  # remove from  available vessels
                if m_index not in vesselends_in:
                    vesselends_in[m_index] = []
                vesselends_in[m_index].append(
                    (vessel, index_v, index_m)
                )  # only end in one, priority of smallest volume

        for m_index, values in vesselends_in.items():
            voi = roi_arrays[m_index][0]

            for vessel, index_v, index_m in values:
                store_info(
                    index_v, vessel, voi, index_m, connected_vessels, roi_data_cache
                )  # voi gets vein and index, vessel gets voi and voiindex and startindex

        # rest of vessels is available for connections
        available_artpoints = []
        available_veinpoints = []
        vessel_for_index_art = dict()
        vessel_for_index_vein = dict()
        min_index_art = 0
        min_index_vein = 0
        for (
            vessel
        ) in (
            available_vessels
        ):  # TODO only remove those vesssels that have both ends in mesh?
            if vessel.type == "vein":
                available_veinpoints += vessel.path
                vessel_for_index_vein[
                    (min_index_vein, min_index_vein + len(vessel.path))
                ] = vessel  # 10 and 5 -> 0,10; 10,15   ;; 0-9 10-15
                min_index_vein += len(vessel.path)
            else:
                available_artpoints += vessel.path

                vessel_for_index_art[
                    (min_index_art, min_index_art + len(vessel.path))
                ] = vessel  # 10 and 5 -> 0,10; 10,15   ;; 0-9 10-15
                min_index_art += len(vessel.path)

        for m_index, mesh in enumerate(meshes):
            if m_index not in vesselends_in:
                search_vein = True
                search_artery = True
                print("Need vein, art")
            else:
                search_vein = True
                search_artery = True
                for tup in vesselends_in[m_index]:
                    vessel, index_v, index_m = tup
                    if vessel.type == "vein":
                        search_vein = False
                    else:
                        search_artery = False

            if search_artery:
                print("need art")
                # find closest pathpoint to meshpoints
                closest_vessel = None
                dist = scipy.spatial.distance.cdist(
                    roi_arrays[m_index][1], available_artpoints, "euclidean"
                )
                closest_index_mesh = np.where(dist == dist.min())[0][0]
                closest_index_to_point = np.where(dist == dist.min())[1][
                    0
                ]  # closest vesselpoint
                for ranges in vessel_for_index_art:
                    if (
                        ranges[0] <= closest_index_to_point
                        and ranges[1] > closest_index_to_point
                    ):
                        # ranges is key for vessel
                        closest_vessel = vessel_for_index_art[ranges]
                        closest_vessel_index = closest_index_to_point - ranges[0]
                        break
                # create tlink to closest meshpoint per vessel that ends in mesh or gets connected to mesh
                print("Art found")
                store_info(
                    closest_vessel_index,
                    closest_vessel,
                    roi_arrays[m_index][0],
                    closest_index_mesh,
                    connected_vessels,
                    roi_data_cache,
                )
            if search_vein:
                print("need vein")
                # find closest pathpoint to meshpoints
                closest_vessel = None
                dist = scipy.spatial.distance.cdist(
                    roi_arrays[m_index][1], available_veinpoints, "euclidean"
                )
                closest_index_mesh = np.where(dist == dist.min())[0][0]
                closest_index_to_point = np.where(dist == dist.min())[1][
                    0
                ]  # closest vesselpoint
                for ranges in vessel_for_index_vein:
                    if (
                        ranges[0] <= closest_index_to_point
                        and ranges[1] > closest_index_to_point
                    ):
                        # ranges is key for vessel
                        closest_vessel = vessel_for_index_vein[ranges]
                        closest_vessel_index = closest_index_to_point - ranges[0]
                        break
                # create tlink to closest meshpoint per vessel that ends in mesh or gets connected to mesh
                print("Vein found")
                store_info(
                    closest_vessel_index,
                    closest_vessel,
                    roi_arrays[m_index][0],
                    closest_index_mesh,
                    connected_vessels,
                    roi_data_cache,
                )

        def vessel_in_bbox(vessel):
            # hardcode bounding box around heart and aorta
            p2 = vessel.path[-1]  # Last point in the path

            bounding_box = [
                (57.0831, 221.179, 41.5196),  # Lower left front corner
                (204.463, 272.721, 369.734),  # Upper right back corner
            ]

            if (
                bounding_box[0][0] <= p2[0] <= bounding_box[1][0]
                and bounding_box[0][1] <= p2[1] <= bounding_box[1][1]
                and bounding_box[0][2] <= p2[2] <= bounding_box[1][2]
            ):
                return True
            else:
                return False

        # for loose vesselends see, if any roi has min value closer than 5 mm and connect them
        loose_veinends = [
            vessel
            for vessel in vessels
            if vessel not in connected_vessels
            and vessel.type == "vein"
            and no_links_to(vessel, vessels)
        ]
        loose_arteryends = [
            vessel
            for vessel in vessels
            if vessel not in connected_vessels
            and not vessel_in_bbox(vessel)
            and vessel.type != "vein"
            and len(links_at(vessel, len(vessel.path) - 1)) == 0
        ]
        threshold = 25
        verbose = True
        # get distance against all vois and if distance lower threshold connect
        for vein in loose_veinends:
            point = vein.path[0]
            min_dist = None
            roi_to_connect = None
            for roi in final_rois:
                if (
                    "blood" not in roi.name
                ):  # limit amount of calculation by removing ones that are too far away
                    index, m_point = get_closest_index_to_point(
                        point, roi.geometry.get_points(), return_distance=True
                    )
                    distance_to_roi = calculate_distance_euclidian(point, m_point)
                    if verbose:
                        print(
                            vein.associated_vesselname[-20::], roi.name, distance_to_roi
                        )
                    if distance_to_roi < threshold and (
                        min_dist == None or min_dist > distance_to_roi
                    ):
                        min_dist = distance_to_roi
                        roi_to_connect = roi
                        roi_to_connect_index = index
            if roi_to_connect != None:
                store_info(
                    0,
                    vein,
                    roi_to_connect,
                    roi_to_connect_index,
                    connected_vessels,
                    roi_data_cache,
                )

        for art in loose_arteryends:
            point = art.path[-1]
            min_dist = None
            roi_to_connect = None
            for roi in final_rois:
                if (
                    "blood" not in roi.name
                ):  # limit amount of calculation by removing ones that are too far away
                    index, m_point = get_closest_index_to_point(
                        point, roi.geometry.get_points(), return_distance=True
                    )
                    distance_to_roi = calculate_distance_euclidian(point, m_point)
                    if verbose:
                        print(
                            art.associated_vesselname[-20::], roi.name, distance_to_roi
                        )
                    if distance_to_roi < threshold and (
                        min_dist == None or min_dist > distance_to_roi
                    ):
                        min_dist = distance_to_roi
                        roi_to_connect = roi
                        roi_to_connect_index = index
            if roi_to_connect != None:
                store_info(
                    len(art.path) - 1,
                    art,
                    roi_to_connect,
                    roi_to_connect_index,
                    connected_vessels,
                    roi_data_cache,
                )

        for roi in final_rois:
            if "blood" not in roi.name:
                with open(settings.cache_dir + "/" + roi.cache_name, "wb") as handle:
                    pickle.dump(
                        roi_data_cache[roi.name],
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

    changed = True
    while changed:
        changed = False
        for j, vessel in enumerate(vessels):
            before_len = len(vessel.reachable_rois)
            # assert that every vessel gets checked
            # print("old", all([any([tupl == otupl for otupl in oldstate]) for tupl in oldstate]))
            for link in get_traversable_links(vessel):
                other_vessel = link.target_vessel
                for roi in other_vessel.get_rois(link.target_index):
                    if (link.source_index, roi) not in vessel.reachable_rois:
                        vessel.reachable_rois.append((link.source_index, roi))
                        vessel.reachable_rois = sorted(
                            list(set(vessel.reachable_rois)), key=lambda x: x[0]
                        )
                        changed = True
    if False:  # print veinconnections
        for vessel in vessels:
            if len(vessel.reachable_rois) > 0:
                print(
                    vessel.type,
                    vessel.associated_vesselname,
                    [roi.name for roi in vessel.get_rois(0)],
                )
                if "aorta" in vessel.associated_vesselname:
                    print("---------------------------")
                if "jump_vessel" in vessel.associated_vesselname:
                    print("______________________________")


def get_voi_vessels(
    blood_roi,
    final_rois,
    use_exact_method=False,
    final_roi_meshes=[],
    coronary_only=True,
):
    if use_exact_method:
        assert len(final_rois) - 1 == len(final_roi_meshes)
    vesselendpoints_artery = [
        vessel.path[len(vessel.path) - 1]
        for vessel in blood_roi.geometry
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
    vesselendpoints_vein = [
        vessel.path[len(vessel.path) - 1]
        for vessel in blood_roi.geometry
        if vessel.type == "vein"
        and len([link for link in vessel.links_to_path if link.source_index == 0]) == 0
    ]
    vesselends_artery = [
        vessel
        for vessel in blood_roi.geometry
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
        for vessel in blood_roi.geometry
        if vessel.type == "vein"
        and len([link for link in vessel.links_to_path if link.source_index == 0]) == 0
    ]
    dict_arteries_in_rois = dict()
    dict_veins_in_rois = dict()
    if not use_exact_method:
        for roi in final_rois[0:-1]:
            dict_arteries_in_rois[roi.name] = f.get_points_in_roi(
                roi.geometry.get_points(), vesselendpoints_artery, 1.5
            )
        for roi in final_rois[0:-1]:
            dict_veins_in_rois[roi.name] = f.get_points_in_roi(
                roi.geometry.get_points(), vesselendpoints_vein, 1.5
            )
    else:
        for roi in final_roi_meshes:  # TODO order!
            dict_arteries_in_rois[roi.name] = get_points_in_roi_mesh(
                roi, vesselendpoints_artery
            )
        for roi in final_roi_meshes:
            dict_veins_in_rois[roi.name] = get_points_in_roi_mesh(
                roi, vesselendpoints_vein
            )

    for key, value in dict_arteries_in_rois.items():
        dict_arteries_in_rois[key] = [vesselends_artery[index] for index in value]
    for key, value in dict_veins_in_rois.items():
        dict_veins_in_rois[key] = [vesselends_vein[index] for index in value]

    if coronary_only:
        for name in dict_arteries_in_rois:
            if "heart" in name or "pericard" in name:
                to_rem = []
                for item in dict_arteries_in_rois[name]:
                    if "heart" not in item.associated_vesselname:
                        to_rem.append(item)
                for non_coronary_item in to_rem:
                    dict_arteries_in_rois[name].remove(non_coronary_item)
        for name in dict_veins_in_rois:
            if "heart" in name or "pericard" in name:
                to_rem = []
                for item in dict_veins_in_rois[name]:
                    if "heart" not in item.associated_vesselname:
                        to_rem.append(item)
                for non_coronary_item in to_rem:
                    dict_veins_in_rois[name].remove(non_coronary_item)

    return dict_arteries_in_rois, dict_veins_in_rois


def lies_within_mesh(points, mesh):
    hits = np.asarray(trimesh.proximity.signed_distance(mesh, points))
    return [True if i >= 0 else False for i in hits]


def get_points_in_roi_mesh(roimesh, vesselpoints):
    # returns indices of points that lie in roipoint array
    indices = []
    lays_in = lies_within_mesh(vesselpoints, roimesh)
    for i, b in enumerate(lays_in):
        if b:
            indices.append(i)
            print("block_found", b)
    return indices


def traverse_vessel(vessel, start_index, path_added, time_added, cell=None):
    """Traverse a vessel until a decision for further travel has to be made by a cell

    Args:
        start_index (_type_): Index at which traversal starts
        path_added (_type_): Previous path from traversal-chain
        time_added (_type_): Previous time from traversal-chain

    Returns:
        _type_: (pathpoints, times)
    """
    # returns pathpart and time added
    # or links and time added til links
    if vessel.type == "vein":
        if cell is not None:
            if cell.loc != "vein" and "jump" not in vessel.associated_vesselname:
                cell.update_cell_cvolume("vein", cell.time)
        # print("vein", path_added, time_added)
        # go in direction until entering artery
        if vessel.no_higher_links(start_index):
            print("Model is not connected?", vessel.associated_vesselname)

            path_added += vessel.path[start_index + 1 : :]
            # time_added += vessel.ti mes[start_index::]
            time_added += vessel.get_times(
                start_index,
                len(vessel.path) - 1,
                no_vol_speed=not (settings.USE_VOL_SPEED),
            )
            return (path_added, time_added)
        else:
            # recursively traverse next vessel
            hl = vessel.highest_link()

            path_added += vessel.path[start_index + 1 : hl.source_index + 1] + [
                hl.target_vessel.path[hl.target_index]
            ]
            # time_added += vessel.t imes[start_index : hl.source_index] + [hl.get_time()]
            time_added += vessel.get_times(
                start_index, hl.source_index, no_vol_speed=not (settings.USE_VOL_SPEED)
            ) + [hl.get_time()]
            return traverse_vessel(
                vessel.highest_link().target_vessel,
                hl.target_index,
                path_added,
                time_added,
                cell,
            )
    else:
        if cell is not None:
            if cell.loc != "artery" and "jump" not in vessel.associated_vesselname:
                cell.update_cell_cvolume("artery", cell.time)

        # go in direction until entering a linked position
        if vessel.no_higher_links(start_index, ignore_tlinks=settings.IGNORE_TLINKS):
            path_added += vessel.path[start_index + 1 : :]
            # time_added += vessel.ti mes[start_index::]
            time_added += vessel.get_times(
                start_index,
                len(vessel.path) - 1,
                no_vol_speed=not (settings.USE_VOL_SPEED),
            )
            return (path_added, time_added)
        else:
            path_added += vessel.path[start_index + 1 : :]
            # time_added += vessel.ti mes[start_index::]
            time_added += vessel.get_times(
                start_index,
                len(vessel.path) - 1,
                no_vol_speed=not (settings.USE_VOL_SPEED),
            )
            # return possible next links
            return (
                (
                    vessel,
                    vessel.next_links(
                        start_index, ignore_tlinks=settings.IGNORE_TLINKS
                    ),
                ),  # TODO in cell remove startlink
                path_added,
                time_added,
                start_index,
            )


# voi fitting
def flow_walk_step(roi, pos_index, speed=1):
    roi_array = roi.vectorfield.get_points()
    roi_points_sorted = roi.geometry.get_points()
    # print(roi_array[pos_index],pos_index)
    selection = [
        int(i)
        for k, i in enumerate(roi_array[pos_index][0])  # nbs
        if roi_array[pos_index][1][k] > 0 and i != -1
    ]  # possible indexes from current index
    chances = [
        roi_array[pos_index][1][k]
        for k, i in enumerate(roi_array[pos_index][0])
        if roi_array[pos_index][1][k] > 0 and i != -1
    ]
    # print(selection,roi_array[pos_index][0],roi_array[pos_index][1])

    if len(selection) == 0:
        return None

    chances = normalize(chances)
    new_index = random.choices(selection, weights=chances, k=1)[0]
    time_taken = (
        calculate_distance_euclidian(
            roi_points_sorted[pos_index], roi_points_sorted[new_index]
        )
        / speed
    )
    new_point = roi_points_sorted[new_index]
    return new_index, new_point, time_taken


def generate_times_per_inlet(amount_of_paths, roi, scale_value=100000000, speed=1):
    starts = [v[2] for v in roi.inlets]
    qs = [
        int(v[0].volumes[-1].get_symval(v[0].volumes[-1].Q_1) * scale_value)
        for v in roi.inlets
    ]
    paths = []

    for i, cindex in enumerate(starts):
        pathamount = amount_of_paths
        if pathamount == "q":
            pathamount = qs[i]
        for number in range(pathamount):
            current_index = cindex
            path = [roi.geometry.get_points()[current_index]]
            while True:
                x = flow_walk_step(roi, current_index, speed)
                if x == None:
                    break
                current_index, next_point, time = x
                path.append(time)
            paths.append(path)
    return paths
