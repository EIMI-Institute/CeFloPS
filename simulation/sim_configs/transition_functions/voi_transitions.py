import random
import numpy as np
from CeFloPS.simulation.common.functions import (
    create_vector,
    random_step_3D,
    normalize,
    mag,
)
from CeFloPS.simulation.common.vessel_functions import (
    standard_speed_function,
    calculate_distance_euclidian,
)
from scipy.spatial.distance import cdist
from CeFloPS.simulation.common.vessel2 import Vessel


def t_position_time_voi_r_walk(cell, bound, simulation, debug=False):
    if not bound:
        travel = d_walk(cell, simulation)
        if debug:
            print("TRAV", travel)
        cell.append_path([travel[0]])
        cell.append_times([travel[1]])
        cell.prev_sim_time = sum([travel[1]])
        new_pos = cell.path[-1]
        check_end_of_trav(cell, new_pos)
    else:
        # wait
        cell.time += 1
        cell.times.append(1)
        cell.path.append(cell.path[len(cell.path) - 1])


def t_position_time_voi_f_walk(cell, bound, simulation, debug=False):
    #print(f"[FW] in {cell.roi.name}") TODO add logger
    if not bound:
        r= flow_walk_step(
            cell.roi, cell.reference_index, speed=cell.roi.speed
        )
        if r==None:
            return -1
        new_index, new_point, time_taken =r
        cell.append_path([new_point])
        cell.append_times([time_taken])
        cell.prev_sim_time = sum([time_taken])
        cell.reference_index = new_index
        new_pos = cell.path[-1]
        check_end_of_trav(cell, new_pos)
    else:
        # wait
        cell.time += 1
        cell.times.append(1)
        cell.path.append(cell.path[len(cell.path) - 1])
        #print("Wait")


def t_location_VOI_VEIN_straight(cell):
    dists = [
        calculate_distance_euclidian(voipoint, cell.path[-1])
        for veinpoint, veinvessel, voiindex, voipoint in cell.roi.veins
    ]
    index = np.argmin(dists)
    chosen_vein = cell.roi.veins[index]
    targetpoint, targetvessel, voiindex, voipoint = chosen_vein
    travel_to_vein(
        cell, targetpoint, targetvessel, standard_speed_function, 0, rw=False
    )


def t_location_VOI_VEIN_biased_rw(cell):
    dists = [
        calculate_distance_euclidian(voipoint, cell.path[-1])
        for veinpoint, veinvessel, voiindex, voipoint in cell.roi.veins
    ]
    # print("veins to select from", cell.roi.veins, "roiname", cell.roi.name)
    index = np.argmin(dists)
    chosen_vein = cell.roi.veins[index]
    targetpoint, targetvessel, voiindex, voipoint = chosen_vein
    travel_to_vein(cell, targetpoint, targetvessel, standard_speed_function, 0, rw=True)


def check_end_of_trav(cell, new_pos):
    if any(
        [
            calculate_distance_euclidian(new_pos, vein_entry_point) < 1
            for vein_entry_point, vein_vessel, vein_index, roi_point in cell.roi.veins
        ]
    ):  # TODO CHECK AGAINST ALL POINTS
        """print(sorted([
            calculate_distance_euclidian(new_pos, entry)
            for entry, v in cell.roi.veins
        ]))"""
        cell.targeted_vessel = []
        dists = [
            calculate_distance_euclidian(voipoint, cell.path[-1])
            for veinpoint, veinvessel, voiindex, voipoint in cell.roi.veins
        ]
        # print("veins to select from", cell.roi.veins, "roiname", cell.roi.name)
        index = np.argmin(dists)
        chosen_vein = cell.roi.veins[index]
        targetpoint, targetvessel, voiindex, voipoint = chosen_vein
        # print("end of trav")
        cell.change_roi(cell.blood)
        cell.prev_sim_time = 0
        # cell.current_position = [chosen_vein, chosen_vein[2]]
        travel_to_vein(
            cell, targetpoint, targetvessel, standard_speed_function, 0, rw=True
        )


def d_walk(cell, simulation, limit_boundary=False, debug=False):
    # cell in blood in capillary system
    # in compartment (muskel organ etc)
    # 3d random walk
    # chances per directionvector to rois / blood vesssels (c_blood plus all destinations), length of vector is sum of all attractions
    vectors = []
    attraction = 0.5  #  cell.roi.get_attraction(simulation)
    multiplier = 1
    if attraction < 0:
        # multiplier += abs(attraction * 2)
        # print("multiplier", multiplier)
        attraction = 1
    spread = attraction
    # print("s",spread)
    vectors.append([spread, 0, 0])
    vectors.append([0, spread, 0])
    vectors.append([0, 0, spread])
    vectors.append([0, -spread, 0])
    vectors.append([-spread, 0, 0])
    vectors.append([0, 0, -spread])
    # add vectors in all directions for randomness

    # bloodvessel attraction:
    if True:  # len(cell.targeted_vessel) == 0:
        if debug:
            print("--- ", attraction)
            for roi in CeFloPS.simulation.rois:
                print(
                    "target and concentration, ",
                    roi.name,
                    roi.get_concentration_share(simulation),
                    roi.concentration_share_target,
                )
                print(roi.name, roi.get_attraction(simulation))
        closest = []
        for vein_entry_point, vein_vessel, vein_index, roi_point in cell.roi.veins:
            closest.append(vein_entry_point)
        closest = sorted(
            closest,
            key=lambda x: calculate_distance_euclidian(cell.current_position, x),
        )
        assert len(closest) > 0, cell.roi.name

        # cell.blood.get_attraction(simulation)
        # print(vesselvectors)
        # print([mag(v) for v in vesselvectors])
        cell.targeted_vessel = random.choices(
            closest[0:5],
            k=1,
            weights=[
                10 / calculate_distance_euclidian(cell.current_position, v)
                for v in closest[0:5]
            ],
        )[0]
    """ attraction = sum(
        [
            roi.get_attraction(simulation)
            for roi in [cell.blood]  # +in CeFloPS.simulation.rois +
            if roi.name
            != cell.roi.name  # entry_vessel.get_rois(len(entry_vessel.path)-1)
        ]
    )

    if attraction < 0:
        attraction = 1 """

    # TODO look at attraction again
    attraction = 2
    vectors.append(
        create_vector(
            source_to_target=[
                cell.current_position,
                cell.targeted_vessel,
            ],
            length=attraction,
            keep_distance_relation=False,
        )
    )
    if all([mag(vec) == 0 for vec in vectors]):
        spread = 1
        # print("s",spread)
        vectors.append([spread, 0, 0])
        vectors.append([0, spread, 0])
        vectors.append([0, 0, spread])
        vectors.append([0, -spread, 0])
        vectors.append([-spread, 0, 0])
        vectors.append([0, 0, -spread])

    multiplier = 1  # TODO mult
    return random_step_3D(
        cell.current_position,
        vectors,
        0.5,
        standard_speed_function,
        multiplier,
    )


def flow_walk_step(roi, pos_index, speed):
    #print("Walk in ",roi.name,"speed ", speed)
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


def travel_to_vein(
    cell, destination, vein, speed_fun, pathindex, rw=False, DBPATHTIMELEN=False
):
    """if settings.DIRECT_RETURN_PATH_1TOA==False:
    #random walk to nearest vein instead of going in a straight line
    path, times=rw_to(destination)


    cell.path +=path
    cell.times += times
    cell.time +=sum(times)
    #cell.path += [destination]
    cell.current_position = [vein, pathindex]
    return"""
    cell.update_cell_cvolume("vein", cell.time)
    if not rw:
        # called after c1, comprelease, arteryend
        assert len(destination) == 3, "destination is no point"
        assert hasattr(vein, "path"), "Vein has to be vessel"
        assert len(vein.path) > pathindex, (
            "index has to be integer lower pathlen",
            pathindex,
            len(vein.path),
            type(pathindex),
        )
        multiplier = 0.1
        """ if "lung" in self.roi.name:
                multiplier = 2  # TODO mult """
        # print("travel to vein")
        cell.times += [
            speed_fun(
                calculate_distance_euclidian(
                    cell.path[len(cell.path) - 1], destination
                ),
                multiplier=multiplier,
            )
        ]
        cell.time += speed_fun(
            calculate_distance_euclidian(cell.path[len(cell.path) - 1], destination),
            multiplier=multiplier,
        )
        if cell.logging and False:
            print(
                "Traveled to vein from",
                cell.path[len(cell.path) - 1],
                destination,
                cell.time,
            )
        if DBPATHTIMELEN:
            print("len after travel to vein", len(cell.times), len(cell.path))
        cell.path += [destination]
        cell.current_position = [vein, pathindex]
        return
    # in case rw is set:
    # rw with litle noise until exit is reached (dist of 1) then call self without rw set
    # print(self.path[-1])
    npoints, ntimes = rw_to_destination(
        position=cell.path[-1], destination=destination, cutoff=1
    )
    cell.times += ntimes
    cell.path += npoints
    cell.time += sum(ntimes)
    travel_to_vein(cell, destination, vein, speed_fun, pathindex, rw=False)


def rw_to_destination(position, destination, cutoff=1):
    times = []
    points = []
    position = position.copy()
    while calculate_distance_euclidian(position, destination) > cutoff:
        vectors = []
        spread = 0.15
        vectors.append([spread, 0, 0])
        vectors.append([0, spread, 0])
        vectors.append([0, 0, spread])
        vectors.append([0, -spread, 0])
        vectors.append([-spread, 0, 0])
        vectors.append([0, 0, -spread])
        vectors.append(
            create_vector(
                source_to_target=[
                    position,
                    destination,
                ],
                length=1,
                keep_distance_relation=False,
            )
        )
        new_pos, time = random_step_3D(
            position,
            vectors,
            0.5,
            standard_speed_function,
            4,
        )
        times.append(time)
        points.append(new_pos)
        position = new_pos
    # print("reached",calculate_distance_euclidian(new_pos, destination), sum(times))
    return points, times
