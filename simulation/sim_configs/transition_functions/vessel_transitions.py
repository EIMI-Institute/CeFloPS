import random
import numpy as np
from CeFloPS.simulation.common.functions import (
    create_vector,
    random_step_3D,
)
from CeFloPS.simulation.common.vessel_functions import (
    standard_speed_function,
    calculate_distance_euclidian,
    Arteriol,
)
import CeFloPS.simulation.settings as settings
from scipy.spatial.distance import cdist
import itertools
from CeFloPS.simulation.common.vessel2 import Vessel
from .three_dim_connection import travel_to, travel_to_vein


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
            assert False
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


def t_position_time_vessel(cell, simulation, p_link_chances):
    """Traverse veins until highest links and arteries until decision events happen. Solve Decisions if simtime is recent

    Args:
        cell (_type_): cell to get cellstate

    Raises:
        Exception: _description_
    """
    trav_vessel, reference_index = cell.current_position
    # sampler
    if trav_vessel.type == "vein":
        if cell.loc != "vein":  # sample locations
            cell.update_cell_cvolume("vein", cell.time)
    else:
        if cell.loc != "artery":  # sample locations
            cell.update_cell_cvolume("artery", cell.time)
    if trav_vessel in simulation.logging_vessels:
        if (
            reference_index in simulation.entry_indices
            or reference_index in simulation.exit_indices
        ):
            cell.update_sampler(trav_vessel, reference_index)

    # if no r in this state, get one
    if cell.radint_in_cell[0] is None or cell.radint_in_cell[0] != trav_vessel:
        # get new radius upon entering new vessels, else keep radius index, thus the procentual position in all volumes
        cell.radint_in_cell = trav_vessel.get_cell_radius(reference_index)
        if (
            cell.radint_in_cell[0] is None
        ):  # generate first point in cell, instant displacement? what if start in voi
            # use TODO
            ...
    if True:  # cell.steptime == 0:
        # WAIT if needed
        if simulation.time < cell.time:
            return
        # else move and transition as given by next target index / vector
        reference_index += 1
        res = trav_vessel.determine_next_target_index(reference_index, p_link_chances)
        if type(res) == tuple:
            next_target_object, next_target_index = res
        else:
            if res == 0:  # go to closest vein
                _, next_target_object = get_nearest_veinend(
                    simulation.end_veins,
                    trav_vessel,
                    trav_vessel.path[len(trav_vessel.path) - 1],
                    simulation.artery_nearest_object_cache,
                )
                next_target_index = 0
            else:  # go to closest voi
                # assert trav_vessel.type == "artery"  # and at end assert?
                next_target_object = res
                next_target_index = get_nearest_voi_index(
                    trav_vessel,
                    simulation.artery_nearest_object_cache,
                    roi=next_target_object,
                )
        # print("target obnj, index: ", next_target_object, next_target_index)
        # get target_vector and steptime until a targetpoint is reached with this vector
        if type(next_target_object)!=Vessel:
            if "blood" not in next_target_object.name:
                assert next_target_object.geometry!=None


        (
            movement_vector,
            updated_steps,
            time_per_step,
        ) = trav_vessel.get_next_target_vector(
            start_point=np.asarray(cell.path[-1]),
            vol=cell.radint_in_cell[2],
            rindex=cell.radint_in_cell[1],
            next_object=next_target_object,
            next_index=next_target_index,
            start_refpoint=trav_vessel.path[reference_index - 1],
            logging=cell.logging,
        )
        #TODO change this to reflect arbitrary speedupdates e g all 100 ms celltime sample a new speed for cells


        cell.region_sampler.update(next_target_object,next_target_index,cell.time)

        cell.multimove(movement_vector, time_per_step, updated_steps)
        new_position_of_cell = (next_target_object, next_target_index)

    return new_position_of_cell  # ,location_change


def t_location_ART_VOI(cell, result):
    """state_artery_choice Make a choice for next link taken

    Args:
        result (_type_): (vessel, nextlinks), path, times, start_index
    """
    # make choice for artery
    cell.waiting_on_decision = result

    cell.append_path(result[1])

    kumulative_array = list(itertools.accumulate(result[2]))
    upd_time = np.asarray(kumulative_array) + cell.time  # TODO correct
    cell.append_times(result[2])

    cell.current_position = [result[0], result[3]]
    # print(result)

    # cell.update_sampler(cell.vessel_sampler, result[1], upd_time)
    assert len(result[1]) == len(result[2]), (
        len(result[1]),
        len(result[2]),
    )
    assert len(result[1][0]) == 3


def t_location_ARTEND_VEIN_VOI(cell, result, simulation):
    """state_artery_end Traversal ended with arteryend, choose to go to next VOI or vein

    Args:
        result (_type_): _description_
    """
    # add to path and times

    cell = cell
    cell.append_path(result[0])
    kumulative_array = list(itertools.accumulate(result[1]))

    upd_time = np.asarray(kumulative_array) + cell.time

    cell.append_times(result[1])
    # print(cell.current_position[0].associated_vesselname)
    # exited vessels
    assert len(result[0]) == len(result[1]), (
        len(result[0]),
        len(result[1]),
    )
    if len(result[0]) > 0:
        assert len(result[0][0]) == 3  # OUT OF RANGE 510 vol
    # cell.update_sampler(cell.vessel_sampler, result[0], upd_time)
    if len(cell.current_position[0].get_rois(cell.current_position[1])) > 0:
        state_switch_to_voi(cell, simulation)

    else:
        state_switch_to_vein(simulation)


def state_switch_to_voi(cell, simulation):
    """state_switch_to_voi Go to next voi"""
    from_artery = cell.current_position[0]
    # if there is a connected roi
    n_roi = random.choice(cell.current_position[0].get_rois(cell.current_position[1]))
    change_time = travel_to(
        cell,
        random.choice(n_roi.geometry.get_points()),
        standard_speed_function,
        simulation.artery_nearest_object_cache,
        from_artery,
        n_roi,
    )  # travel to random point there
    # print("trav to roi",n_roi.name)
    cell.staged_changes.append(
        (
            change_time,
            cell.change_roi,
            n_roi,
        )  # this one still has roi blood? see staged changes
    )  # entry in roi)
    cell.update_cell_cvolume("vois", change_time)


def state_switch_to_vein(cell, simulation, debug=False):
    """state_switch_to_vein Go to next vein"""
    if debug:
        print(
            "self.pos at nearest vein",
            cell.current_position[0].associated_vesselname,
            len(cell.current_position[0].path),
        )
    entry, vein = get_nearest_veinend(
        simulation.end_veins,
        cell.current_position[0],
        cell.current_position[0].path[len(cell.current_position[0].path) - 1],
        simulation.artery_nearest_object_cache,
    )
    if False:
        print("to nearest: vein", vein.associated_vesselname)
    # stay in blood
    assert "blood" in cell.roi.name
    travel_to_vein(cell, entry, vein, standard_speed_function, 0)


def get_nearest_voi_index(from_artery, artery_nearest_object_cache, roi=None):
    if from_artery != None and roi != None and artery_nearest_object_cache != None:
        # return the time at which a cell officially entered the capillary system e.g. is is mesh
        if (
            from_artery.id + roi.name + "enter_capillars_point"
            not in artery_nearest_object_cache
        ):
            # print(11111111111111111)
            selection_points = roi.geometry.get_points()
            out_point = from_artery.path[len(from_artery.path) - 1]
            #print(selection_points,out_point)
            d = cdist([out_point], selection_points, "euclidean")
            index = np.where(d == d.min())[1][0]
            closest_point_roi = selection_points[index]

            artery_nearest_object_cache[
                from_artery.id + roi.name + "enter_capillars_point"
            ] = index
        return artery_nearest_object_cache[
            from_artery.id + roi.name + "enter_capillars_point"
        ]


def get_nearest_veinend(
    end_veins,
    from_artery,
    out_point,
    artery_nearest_object_cache,
    debug=False,
):
    if from_artery.id + "vein" not in artery_nearest_object_cache:
        # print(11111111111111111111111)
        selection = end_veins
        if "pulmonar" in from_artery.associated_vesselname:
            selection = [
                vein for vein in end_veins if "pulmonar" in vein.associated_vesselname
            ]
            # print("no lungs?")
        selection_points = [vein.path[0] for vein in selection]

        d = cdist([out_point], selection_points, "euclidean")
        index = np.where(d == d.min())[1][0]
        if debug:
            print(
                "ARTERY ",
                from_artery.associated_vesselname,
                " TO ",
                selection[index].associated_vesselname,
            )
        artery_nearest_object_cache[from_artery.id + "vein"] = (
            selection_points[index],
            selection[index],
        )
    return artery_nearest_object_cache[from_artery.id + "vein"]
