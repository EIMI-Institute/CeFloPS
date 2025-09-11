import numpy as np
from CeFloPS.simulation.common.functions import create_vector, random_step_3D
from CeFloPS.simulation.common.vessel_functions import (
    standard_speed_function,
    calculate_distance_euclidian,
    Arteriol,
)
from scipy.spatial.distance import cdist
from CeFloPS.simulation.common.vessel2 import Vessel


def travel_to(
    cell,
    destination,
    speed_fun,
    artery_nearest_object_cache=None,
    from_artery=None,
    roi=None,
    debug=False,
    DBPATHTIMELEN=False,
):
    multiplier = 1
    start_time = cell.time
    cell.current_position = destination
    if cell.logging and False:
        print(
            "Traveled to VOI from",
            cell.path[len(cell.path) - 1],
            destination,
            cell.time,
        )

    if DBPATHTIMELEN:
        print("len before TRAVELTO", len(cell.times), len(cell.path))
    if from_artery != None and roi != None and artery_nearest_object_cache != None:
        # return the time at which a cell officially entered the capillary system e.g. is is mesh
        if (
            from_artery.id + roi.name + "enter_capillars_point"
            not in artery_nearest_object_cache
        ):
            # print(11111111111111111)
            selection_points = roi.geometry.get_points()
            out_point = from_artery.path[len(from_artery.path) - 1]
            d = cdist([out_point], selection_points, "euclidean")
            index = np.where(d == d.min())[1][0]
            closest_point_roi = selection_points[index]

            artery_nearest_object_cache[
                from_artery.id + roi.name + "enter_capillars_point"
            ] = closest_point_roi
        distance = calculate_distance_euclidian(
            from_artery.path[len(from_artery.path) - 1],
            artery_nearest_object_cache[
                from_artery.id + roi.name + "enter_capillars_point"
            ],
        )
        entry_time = standard_speed_function(
            distance,
            [from_artery, Arteriol(avg_diameter=0.5)],  # eigentlich avg 0.1
        )  # TODO speed?

        cell.time += entry_time
        cell.time_change_target = cell.time
        cell.time += speed_fun(
            calculate_distance_euclidian(
                artery_nearest_object_cache[
                    from_artery.id + roi.name + "enter_capillars_point"
                ],
                destination,
            )
        )
        cell.times += [
            entry_time,
            speed_fun(
                calculate_distance_euclidian(
                    artery_nearest_object_cache[
                        from_artery.id + roi.name + "enter_capillars_point"
                    ],
                    destination,
                )
            ),
        ]
        cell.path += [
            artery_nearest_object_cache[
                from_artery.id + roi.name + "enter_capillars_point"
            ],
            destination,
        ]  # add destination and closest point
        if cell.logging and False:
            print(
                "Traveled to VOI from",
                cell.path[len(cell.path) - 1],
                destination,
                cell.time,
            )
        if DBPATHTIMELEN:
            print(
                "len after TRAVELTO +2 dest and closest point, +2 time between",
                len(cell.times),
                len(cell.path),
            )
        return start_time + entry_time
    else:
        assert False
        cell.times += [
            speed_fun(
                calculate_distance_euclidian(cell.path[len(cell.path) - 1], destination)
            )
        ]
        cell.path += [destination]
        cell.time += speed_fun(
            calculate_distance_euclidian(cell.path[len(cell.path) - 1], destination)
        )


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
        cell.comp_change_times[cell.time] = -2
        # called after c1, comprelease, arteryend
        assert len(destination) == 3, "destination is no point"
        assert type(vein) == Vessel, "Vein has to be vessel"
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
    npoints, ntimes = cell.rw_to_destination(
        position=cell.path[-1], destination=destination, cutoff=1
    )
    cell.comp_change_times[cell.time] = -1
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
