
import numpy as np
import trimesh

# Methods to create vectorfield for flow based capillary walk
from numba import jit, cuda


# @jit(nopython=True)
def generate_arrays_parallel(
    roi_points_sorted,
    pitch,
    vein_indices=[],
    art_indices=[],
    use_dict=True,
    parallel=False,
):
    roi_points_sorted = np.round(roi_points_sorted, 3)
    print("Generating Array for flowwalk GPU")
    print("parallel")
    output_array = np.full(
        26 * len(roi_points_sorted), -1, dtype=np.int64
    )  # [-1 for _ in range(26*len(roi_points_sorted))]

    # initialise output array with neighbour relations
    insert = 26
    for i in range(len(roi_points_sorted)):
        # assert insert==26, insert
        insert = 0
        for a in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if not (a == 0 and j == 0 and k == 0):
                        target = (
                            np.round(roi_points_sorted[i][0] + pitch * a, 3),
                            np.round(roi_points_sorted[i][1] + pitch * j, 3),
                            np.round(roi_points_sorted[i][2] + pitch * k, 3),
                        )

                        # binarysearch for point in points

                        left, right = 0, len(roi_points_sorted) - 1
                        found = False
                        while left <= right:
                            mid = (left + right) // 2

                            changed = False
                            for l in range(len(target)):
                                if roi_points_sorted[mid][l] < target[l]:
                                    left = mid + 1
                                    changed = True
                                    break
                                elif roi_points_sorted[mid][l] > target[l]:
                                    right = mid - 1
                                    changed = True
                                    break

                            if not changed:
                                found = True
                                break

                        if not found:
                            mid = -1
                        output_array[i * 26 + insert] = mid
                        insert = insert + 1

    # neighbours end
    chunk_size = 26
    num_chunks = (len(output_array) + chunk_size - 1) // chunk_size
    collected_nbs = output_array.reshape(num_chunks, chunk_size)

    # collect accumulated directions
    collected_dirs = np.full(26 * len(roi_points_sorted), 0, dtype=np.float64)
    collected_dirs = collected_dirs.reshape(num_chunks, chunk_size)

    artery_indices = art_indices
    vein_indices = vein_indices

    # set art directions
    source_voxel_indices = artery_indices + vein_indices

    # resulting_borders = []
    # min_borders = []
    # max_borders = []

    # initialise complementary index lookup data
    collected = []
    for a in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if not (a == 0 and j == 0 and k == 0):
                    target = (
                        np.round(0 + pitch * a, 3),
                        np.round(0 + pitch * j, 3),
                        np.round(0 + pitch * k, 3),
                    )
                    collected.append(target)
    reverse_index_map = (
        dict()
    )  # maps every direction index to opposite to 'reverse' the negative values
    for i, point in enumerate(collected):
        for j, point2 in enumerate(collected):
            if (
                point[0] == -point2[0]
                and point[1] == -point2[1]
                and point[2] == -point2[2]
            ):
                reverse_index_map[i] = j

    # initialise direction "collector" arrays, one per index per inlet/outlet
    num_dir_collectors = len(source_voxel_indices)
    all_dir_collectors = np.zeros(num_dir_collectors * 26 * len(roi_points_sorted))
    all_dir_collectors = all_dir_collectors.reshape(
        num_dir_collectors, num_chunks, chunk_size
    )

    for a_i, artery_index in enumerate(source_voxel_indices):
        neg = False  # artery
        if artery_index in vein_indices:
            neg = True
        dir_collector = all_dir_collectors[a_i]

        next_bound_index = 0
        center = artery_index
        outer_bound = collected_nbs[artery_index]  # nbs
        # bound_neighbours = set()
        prev_bound = {center}
        prev_bound_prev = prev_bound

        # set initial chances, every side that has a neighbour gets set to one/Q if specified
        direction_chance = 1
        if neg:
            direction_chance = -1

        # set all near ones
        for direction_i in range(26):
            if (
                collected_nbs[artery_index][direction_i] != -1
            ):  # there exists a voxel at neighbourplace
                dir_collector[artery_index][direction_i] = direction_chance
                # repeat direction in close proximity
                dir_collector[collected_nbs[artery_index][direction_i]][
                    direction_i
                ] = direction_chance

            else:
                dir_collector[artery_index][direction_i] = 0
                # set first block for all directions

        # set first bound around startvoxel
        rem_neg = lambda arr: [elem for elem in arr if elem >= 0]

        prev_bound = set([center])
        prev_bound_prev = prev_bound
        nbs = [rem_neg(collected_nbs[i]) for i in prev_bound]
        prev_bound_neighbours = [x for x in [x for xs in nbs for x in xs]]

        next_bound = set(
            [
                index
                for index in prev_bound_neighbours
                if index not in prev_bound and index not in prev_bound_prev
            ]
        )  # point is not from older iteration
        # next bound holds all indices of voxels around startvoxel that are existent

        next_bound_index += 1
        # layer #1
        # add neighbourig directions
        direction_changes = []
        to_add = np.zeros(26 * len(next_bound))
        to_add = to_add.reshape(len(next_bound), 26)
        direction_changes = []
        for i_point, point in enumerate(next_bound):
            # every voxel surrounding:
            nbs = [i for i in rem_neg(collected_nbs[point]) if i in next_bound]
            # divide by 2 to make all axis and diagonals the most probablle. Leading to even lines emerging from the center
            # copy the previously elongated directions to themselves from neighbours to create "mixed" rays

            # nbdirections
            neighbouring_directions_for_point = sum([dir_collector[i] / 2 for i in nbs])
            to_add[i_point] += neighbouring_directions_for_point
        for k, point in enumerate(
            next_bound
        ):  # update the values later to not falsify data for later iterations
            for i, d in enumerate(to_add[k]):
                if collected_nbs[point][i] not in nbs:
                    dir_collector[point][i] += d
                else:
                    ...  # assert False

            print("firstbound", dir_collector[i_point])

        # set coinsecutive bounds:

        prev_bound_prev = prev_bound
        prev_bound = next_bound
        ic = 0
        while True or ic < 2:
            next_bound_index += 1
            ##resulting_borders.append(next_bound)
            nbs = [rem_neg(collected_nbs[i]) for i in prev_bound]

            prev_bound_neighbours = [x for x in [x for xs in nbs for x in xs]]
            print("prev bound nbs", prev_bound_neighbours, "prev_bound", prev_bound)
            next_bound = set(
                [
                    index
                    for index in prev_bound_neighbours
                    if index not in prev_bound
                    and index not in prev_bound_prev
                    and index != -1
                ]
            )  # point is not from older iteration
            # print("next_bound len", len(next_bound), prev_bound)
            for point in next_bound:  # copy incoming directions
                prevs = [i for i in rem_neg(collected_nbs[point]) if i in prev_bound]
                directions = sum(
                    [1 / next_bound_index**2 * dir_collector[i] for i in prevs]
                )
                # dir_collector[point]+=directions

                # update values in own dir_collector for this artindex
                print("directions", directions, prevs)
                for k, dir1 in enumerate(directions):
                    if collected_nbs[point][k] not in next_bound:  # not in same layer
                        dir_collector[point][k] += dir1
                    else:
                        # print("was in bound",ic)
                        dir_collector[point][k] += 0

            """ max_borders.append(
                max(
                    [max([x[-1] for x in voxel_nb_indices[i][1]]) for i in next_bound] + [0]
                )
            )
            minarray = [
                min([x[-1] for x in voxel_nb_indices[i][1] if x[-1] > 0])
                for i in next_bound
                if voxel_nb_indices[i][1][0][-1] > 0
            ]
            if len(minarray) > 0:
                min_borders.append(min(minarray))
            else:
                min_borders.append(0) """

            prev_bound_prev = prev_bound
            prev_bound = next_bound
            # colored_pointgraph(voxel_nb_indices,voxel_nb_indices)
            if len(next_bound) == 0:
                # print("b", ic)
                break
            ic += 1
        # print(max_borders)
        # return next_bound_index, min_borders, max_borders

        """ next_bound_index, min_borders, max_borders = set_consecutive_bounds(
            prev_bound,
            next_bound,
            next_bound_index,
            voxel_nb_indices,
            min_borders,
            max_borders,
        )  """

        # return next_bound_index, min_borders, max_borders
        #
        """  next_bound_index, min_borders, max_borders = set_direction_chances(
            artery_indices, roi_array, roi_points_sorted
        ) """

        # set_direction_chances(vein_indices, roi_array, roi_points_sorted, neg=True)

        # convert all negative directions in positive directions in point that they point to on the reverse vector
        if neg:
            reserve_dirs = np.zeros(26 * len(roi_points_sorted))
            reserve_dirs = reserve_dirs.reshape(num_chunks, chunk_size)
            for i in range(len(collected_nbs)):
                for direction_i in range(
                    26
                ):  # take negative out and add as positive in nb that had negative to it pointing to self
                    if dir_collector[i][direction_i] < 0:
                        set_in_index = collected_nbs[i][direction_i]
                        # if set in index is in mesh and not outside:
                        if set_in_index != -1:
                            rev_dir = reverse_index_map[direction_i]
                            # assert rev_dir==25-direction_i
                            # print("dir", set_in_index, rev_dir, direction_i, 25-direction_i)
                            dir_collector[set_in_index][rev_dir] -= dir_collector[i][
                                direction_i
                            ]
                        dir_collector[i][direction_i] = 0

                        # neighbour that would get linked if positive is in nbs at index of direction that was set previously
            # dir_collector=reserve_dirs

    for d in range(len(all_dir_collectors)):  # add all calculated values together
        for i in range(len(collected_nbs)):
            collected_dirs[i] += all_dir_collectors[d][i]

    print("collected dirs", dir_collector[4])
    print("collected val", collected_dirs[4])

    # set all veins to not connect to other points:
    for vein_index in vein_indices:
        # vein_index = binary_search(vein_point, roi_points_sorted)
        for i in range(26):
            collected_dirs[vein_index][i] = 0
    return collected_nbs, collected_dirs
    # return np.asarray(roi_points_sorted), np.asarray(roi_array)
