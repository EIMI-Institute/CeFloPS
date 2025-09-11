import math
import numpy as np
import random
import trimesh


# general functions


def calculate_angle_difference(vector_a, vector_b):  # θ = cos-1 [ (a · b) / (|a| |b|) ]
    return math.degrees(
        np.arccos(np.dot(vector_a, vector_b) / (mag(vector_a) * mag(vector_b)))
    )


def moving_average(array, window_size):
    """moving_average Use a moving average of window size on an array

    Args:
        array (_type_): _description_
        window_size (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")

    half_window = window_size // 2
    result = [0] * len(array)

    for i in range(len(array)):
        valsum = 0
        count = 0

        for j in range(
            max(0, int(i - half_window)), min(len(array), int(i + half_window) + 1)
        ):
            valsum += array[j]
            count += 1

        result[i] = valsum / count

    return result


def create_vector(source_to_target, length, keep_distance_relation=False):
    """create_vector creates a vector with the given length from a 2 point array that points from the first to the second point

    Args:
        source_to_target (list): 2 point array
        length (float): length of the returned vector
        keep_distance_relation (bool, optional): If true, does not normalize vector. Defaults to False.

    Returns:
        _type_: vector with given length
    """
    vector = np.asarray(source_to_target[1]) - source_to_target[0]
    vetor_initial_length = mag(vector)
    if keep_distance_relation:
        vector = (vector / vetor_initial_length) * length
        for i, p in enumerate(vector):
            if p == max(vector):
                break
        vector[i] = vector[i] + 10 / vetor_initial_length

        return vector  # * (10 / vetor_initial_length)
    return (vector / vetor_initial_length) * length


def mag(vector):
    """mag Generate magnitude of 3 dimenstional vector

    Args:
        vector (array/list): 3 dimensional vector

    Returns:
        float: magnitude of vector
    """
    return math.sqrt(
        vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]
    )


def point_in(p, set_of_points):
    if len(set_of_points) > 0:
        assert len(p) == len(set_of_points[0]), "Punkte nicht gleichdimensional"
    return any(
        (point[0] == p[0] and point[1] == p[1] and point[2] == p[2])
        for point in set_of_points
    )


def calc_row_idx(k, n):
    return int(
        math.ceil(
            (1 / 2.0) * (-((-8 * k + 4 * n**2 - 4 * n - 7) ** 0.5) + 2 * n - 1) - 1
        )
    )


def elem_in_i_rows(i, n):
    return i * (n - 1 - i) + (i * (i + 1)) // 2


def calc_col_idx(k, i, n):
    return int(n - elem_in_i_rows(i + 1, n) + k)


def condensed_to_square(k, n):
    i = calc_row_idx(k, n)
    j = calc_col_idx(k, i, n)
    return i, j


def count(element, arr):
    """
    returns count of element in array arr
    """
    counter = 0
    for e in arr:
        if e == element:
            counter = counter + 1
    return counter


def pick_color(index):
    kelly_colors = dict(
        vivid_yellow=[255, 179, 0],
        strong_purple=[128, 62, 117],
        vivid_orange=[255, 104, 0],
        very_light_blue=[166, 189, 215],
        vivid_red=[193, 0, 32],
        grayish_yellow=[206, 162, 98],
        medium_gray=[129, 112, 102],
        # these aren't good for people with defective color vision:
        vivid_green=[0, 125, 52],
        strong_purplish_pink=[246, 118, 142],
        strong_blue=[0, 83, 138],
        strong_yellowish_pink=[255, 122, 92],
        strong_violet=[83, 55, 122],
        vivid_orange_yellow=[255, 142, 0],
        strong_purplish_red=[179, 40, 81],
        vivid_greenish_yellow=[244, 200, 0],
        strong_reddish_brown=[127, 24, 13],
        vivid_yellowish_green=[147, 170, 0],
        deep_yellowish_brown=[89, 51, 21],
        vivid_reddish_orange=[241, 58, 19],
        dark_olive_green=[35, 44, 22],
    )
    for i, key in enumerate(kelly_colors):
        if i == index % 20:
            return kelly_colors[key]


def same_counts(a1, a2):
    a1.sort()
    a2.sort()
    return a1 == a2


def norm_vector(vector):
    return np.asarray(vector) / mag(vector)


def dot_product(vectora, vectorb):
    return np.dot(norm_vector(vectora), norm_vector(vectorb))


def normalize(array):
    total = sum(array)
    if total == 0:
        return array  # zero array is already normalized / avoid division by zero
    out = array.copy()
    for i in range(0, len(array)):
        out[i] = out[i] / total
    return out


def make_choice(items, probabilities):
    choice = random.random()

    assert (
        sum(probabilities) > 0
    ), f"Probabilities must be greater 0 but is {sum(probabilities)} with {probabilities}"
    if sum(probabilities) != 1:
        probabilities = normalize(probabilities)

    for i, item in enumerate(items):
        if sum(probabilities[: i + 1]) > choice:
            return item


def single_dir_vector(vector_length, axis, dimension):
    out = np.zeros(dimension)
    out[axis] = vector_length
    return out


""" def flatten(l):
    match l:
        case (list() | tuple() | set()) as items:
            for item in items:
                yield from flatten(item)
        case _:
            yield l """


# ------------------------------------TODO
def random_step_3D(position, vectors, stepsize, speed_function, multiplier=1):
    """random_step_3D Takes a list of attraction vectors and calculates the corresponding probable step

    Args:
        position (_type_): start_position
        vectors (_type_): attraction vectors
        stepsize (_type_): length of stepvector
        speed_function (_type_): speed function to calculate time for step
        multiplier (int, optional): Multiplier to allow for larger steps/ faster cells. Defaults to 1.

    Returns:
        _type_: next_position, time_taken
    """
    # available directions
    stepdirections = [
        (stepsize, 0, 0),
        (-stepsize, 0, 0),
        (0, stepsize, 0),
        (0, -stepsize, 0),
        (0, 0, stepsize),
        (0, 0, -stepsize),
    ]
    # chances for each axis and each direction on those so that -1 and +1 cant cancel each other resulting in no movement of a cell
    stepchances = np.asarray(
        [
            sum([vector[0] for vector in vectors if vector[0] >= 0] + [0]),
            abs(sum([vector[0] for vector in vectors if vector[0] <= 0] + [0])),
            sum([vector[1] for vector in vectors if vector[1] >= 0] + [0]),
            abs(sum([vector[1] for vector in vectors if vector[1] <= 0] + [0])),
            sum([vector[2] for vector in vectors if vector[2] >= 0] + [0]),
            abs(sum([vector[2] for vector in vectors if vector[2] <= 0] + [0])),
        ]
    )
    stepchances = [float(chance) for chance in stepchances]
    stepchances = normalize(stepchances)

    step = random.choices(stepdirections, k=1, weights=stepchances)[0]

    next_point = np.asarray(position) + step
    time = speed_function(stepsize, multiplier=multiplier)
    return next_point, time


def random_walk_3D(
    roi, cell, walk_time, concentrations, vectors, close_rois
):  # TODO call by reference? WICHTIG MEMORY
    # waehle alle veneneintrittspunkte in der Naehe und laufe jeweils einen schritt, wobei diejenigen schritte,
    #  die einen neuen wegpunkt näher an einer nahen vene produzieren prääferiert werden und direkte
    # Rückwärtsbewegung ausgeschlossen wird

    STEPLENGTH = 0.05
    walked_path = [cell.position]  # TODO check that a ell may never reenter an ARTERY
    time_target = cell.time + walk_time

    # solange nicht im direkten perimeter eines kompartiments
    while (
        distance_to_next_roi(walked_path[len(walked_path) - 1], close_rois)
        > (STEPLENGTH * 2)
        and cell.time < time_target
    ):  # check if another roi got entered
        # get distance to next vein in each axis direction
        min_vein_distance = min(
            [
                calculate_distance_euclidian(walked_path[len(walked_path) - 1], vpoint)
                for vpoint in next_vein_entries
            ]
        )
        clostest_vein_entry = [
            vein
            for vein in next_vein_entries
            if calculate_distance_euclidian(walked_path[len(walked_path) - 1], vein)
            == min_vein_distance
        ][0]

        stepdirections = [
            (STEPLENGTH, 0, 0),
            (-STEPLENGTH, 0, 0),
            (0, STEPLENGTH, 0),
            (0, -STEPLENGTH, 0),
            (0, 0, STEPLENGTH),
            (0, 0, -STEPLENGTH),
        ]
        # chances for each axis and each direction on those so that -1 and +1 cant cancel each other resulting in no movement of a cell
        # concatienation of [0] so that the array for sum has always a min size of 1
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
        # add 0.1 and normalize
        # stepchances=stepchances+np.random.rand(6)/2 # random up to 25% TODO
        stepchances = normalize(stepchances)
        """ print(
            "picking ",
            stepdirections,
            stepchances,
            "pick:",
            stepdirections[np.random.choice(len(stepdirections), 1, p=stepchances)[0]],
        ) """

        step = stepdirections[
            np.random.choice(len(stepdirections), 1, p=stepchances)[0]
        ]
        walked_path.append(np.asarray(walked_path[len(walked_path) - 1]) + step)
        cell.time += STEPLENGTH / roi.speed

    cell.path.extend(walked_path)
    if cell.time > time_target:
        # continue travel of cell in next roi
        cell.set_position((next_roi, walked_path[len(walked_path) - 1]))
        new_roi.add_cell()
        roi.sub_cell()
        cell.traverse_cell(concentrations)


def pitch_transformed(point, pitch):
    options = [pitch, 0, -pitch]
    pitched = np.asarray(
        [
            [a, b, c]
            for a in options
            for b in options
            for c in options
            if not (a == b == c == 0)
        ]
    )
    for i in range(len(pitched)):
        pitched[i] = point + pitch
    return pitched


def filled(array, points, pitch):
    # array = set(array)

    centers_from_points = np.asarray(
        [
            [round(point[0], pitch), round(point[1], pitch), round(point[2], pitch)]
            for point in points
        ]
    )
    return [point in array for point in centers_from_points]


def rround(x, unit):
    if x % unit > unit / 2:
        return x - (x % unit) + unit
    else:
        return x - (x % unit)


def get_block_position(point, voxel_geometry):
    pitch = voxel_geometry.pitch
    centers_from_points = np.asarray(
        [round(point[0], pitch), round(point[1], pitch), round(point[2], pitch)]
    )
    if centers_from_points in voxel_geometry.points:
        ...


def random_walk_voxel_centers(roi, cell):
    # take adjacent voxels and compute the distance to targetvoxels.
    # take the reverse as chance for each adjacent voxel
    voxel_geometry = roi.geometry
    pitch = voxel_geometry.pitch
    options = [pitch, 0, -pitch]
    check_index_points = np.asarray(
        [
            [a, b, c]
            for a in options
            for b in options
            for c in options
            if not (a == b == c == 0)
        ]
    )
    check_index_points = check_index_points + np.asarray(
        roi.get_block_position(cell.position)
    )
    print(check_index_points)
    non_tran = []
    check_index_chances = []
    for i, point in enumerate(check_index_points):
        check_index_chances.append(0)
        if point in roi.points:
            chance[i] += roi.get_attraction()
        else:
            # to check if it goes toward or outward from centre of roi group all nontraversable blocks
            non_tran.append(point)  # changes

    point = non_tran[0]
    # if point has another adjacent block
    n_t_a = [point]
    n_t_b = []
    added = True
    check = 0
    while added:
        added = False
        for p in pitch_transformed(n_t_a[check], pitch):
            if p in non_tan:
                non_tan.remove(p)
                n_t_a.append(p)
                added = True
        if check < len(n_t_a):
            check += 1
    n_t_b = non_tan

    # check distance of the two structures to the roi
    if calculate_distance_euclidian(
        f.geometric_mean(n_t_a), roi.centre
    ) < calculate_distance_euclidian(f.geometric_mean(n_t_b), roi.centre):
        # set chance to exit into n_t_b cubes
        for outer_point in n_t_b:
            chance[np.where(check_index_points == outer_point)] = (
                vessels.get_attraction()
            )  # TODO take all near vessels with higher strength
    else:
        for outer_point in n_t_a:
            chance[np.where(check_index_points == outer_point)] = (
                vessels.get_attraction()
            )  # TODO take all near vessels with higher strength

    # step
    chosen_destination = make_choice(...)
    cell.time += (
        calculate_distance_euclidian(cell.position, destination)
        / constants.capillary_speed
    )
    cell.position = chosen_destination

    if chosen_destination not in compatiment.points:
        cell.roi = cell.roi.outer_roi
        cell.roi.outer_roi.add_cell()
        roi.sub_cell()


def random_step_voxel_centers(roi, cell):
    # take adjacent voxels and compute the distance to targetvoxels.
    # take the reverse as chance for each adjacent voxel
    voxel_geometry = roi.geometry
    pitch = voxel_geometry.pitch[0]
    options = [pitch, 0, -pitch]
    check_index_points = np.asarray(
        [
            [a, b, c]
            for a in options
            for b in options
            for c in options
            if not (a == b == c == 0)
        ]
    )
    check_index_points = check_index_points + np.asarray(
        roi.get_block_position(cell.position)
    )
    print(check_index_points)
    non_tran = []
    check_index_chances = []
    for i, point in enumerate(check_index_points):
        check_index_chances.append(0)
        if point in roi.points:
            chance[i] += roi.get_attraction()
        else:
            # to check if it goes toward or outward from centre of roi group all nontraversable blocks
            non_tran.append(point)  # changes

    point = non_tran[0]
    # if point has another adjacent block
    n_t_a = [point]
    n_t_b = []
    added = True
    check = 0
    while added:
        added = False
        for p in pitch_transformed(n_t_a[check], pitch):
            if p in non_tan:
                non_tan.remove(p)
                n_t_a.append(p)
                added = True
        if check < len(n_t_a):
            check += 1
    n_t_b = non_tan

    # check distance of the two structures to the roi
    if calculate_distance_euclidian(
        f.geometric_mean(n_t_a), roi.centre
    ) < calculate_distance_euclidian(f.geometric_mean(n_t_b), roi.centre):
        # set chance to exit into n_t_b cubes
        for outer_point in n_t_b:
            chance[np.where(check_index_points == outer_point)] = (
                vessels.get_attraction()
            )  # TODO take all near vessels with higher strength
    else:
        for outer_point in n_t_a:
            chance[np.where(check_index_points == outer_point)] = (
                vessels.get_attraction()
            )  # TODO take all near vessels with higher strength

    # step
    chosen_destination = make_choice(...)
    cell.time += (
        calculate_distance_euclidian(cell.position, destination)
        / constants.capillary_speed
    )
    cell.position = chosen_destination

    if chosen_destination not in compatiment.points:
        cell.roi = cell.roi.outer_roi
        cell.roi.outer_roi.add_cell()
        roi.sub_cell()


def calculate_distance_euclidian(pointA, pointB):
    """calculates the distance between two triplets that represent 3D points"""
    return math.sqrt(
        (pointA[0] - pointB[0]) * (pointA[0] - pointB[0])
        + (pointA[1] - pointB[1]) * (pointA[1] - pointB[1])
        + (pointA[2] - pointB[2]) * (pointA[2] - pointB[2])
    )


def get_organ_for_point(point):
    # TODO export to lookup list
    organ_names = []
    print("Found stl Files: ")
    for name in glob.glob(str("../*organ*")):
        stlRegex = re.compile(r".stl$")
        mo1 = stlRegex.search(name)
        if mo1 != None:
            # if("pulmonary" not in name):
            organ_names.append(name)
    print(organ_names)
    organs = []
    for name in organ_names:
        organs.append(trimesh.load(name))

    # find out in which organs those vessels end
    vessels_organ_dist = [[] for i in range(len(organs))]
    for i, organ in enumerate(organs):
        vessel_end_hits = np.asarray(
            trimesh.proximity.signed_distance(organ, [point])
        )  # trimesh.proximity.signed_distance(vessel.submesh,selection )
        # vessel_end_hits=[select_indices[i] for i, x in enumerate(x >= 0 for x in vessel_end_hits) if x == True]
        # vessel_end_hits=[select_indices[i] for i in np.where(vessel_end_hits>=0)[0]]
        vessels_organ_dist[i] = vessel_end_hits

    for i, organ in enumerate(organs):
        if vessels_organ_dist[i][0] >= 0:
            print("Found organ", organ_names[i])
            return organ_names[i]


def step(point, length, xp, xn, yp, yn, zp, zn):
    choice = random.random()
    total = sum([xp, xn, yp, yn, zp, zn])
    xp = xp / total
    xn = xn / total
    yp = yp / total
    yn = yn / total
    zp = zp / total
    zn = zn / total
    if xp >= choice:
        return [point[0] + length, point[1], point[2]]
    if xp + xn > choice:
        return [point[0] - length, point[1], point[2]]
    if xp + xn + yp > choice:
        return [point[0], point[1] + length, point[2]]
    if xp + xn + yp + yn > choice:
        return [point[0], point[1] - length, point[2]]
    if xp + xn + yp + yn + zp > choice:
        return [point[0], point[1], point[2] + length]
    if 1 >= choice:
        return [point[0], point[1], point[2] - length]


def get_closest_dim_dist(
    reference, dimension, sign, vein_entries
):  # ("x","p",next_vein_entries)
    print(reference, vein_entries)
    dist = 0
    if dimension == "x":
        if sign == "p":
            k = [
                reference[0] - vein_entry[0]
                for vein_entry in vein_entries
                if reference[0] - vein_entry[0] <= 0
            ]
            if len(k) > 0:
                dist = max(k)

        else:
            k = [
                reference[0] - vein_entry[0]
                for vein_entry in vein_entries
                if reference[0] - vein_entry[0] >= 0
            ]
            if len(k) > 0:
                dist = min(k)
    if dimension == "y":
        if sign == "p":
            k = [
                reference[1] - vein_entry[1]
                for vein_entry in vein_entries
                if reference[1] - vein_entry[1] <= 0
            ]
            if len(k) > 0:
                dist = max(k)

        else:
            k = [
                reference[1] - vein_entry[1]
                for vein_entry in vein_entries
                if reference[1] - vein_entry[1] >= 0
            ]
            if len(k) > 0:
                dist = min(k)
    if dimension == "z":
        if sign == "p":
            k = [
                reference[2] - vein_entry[2]
                for vein_entry in vein_entries
                if reference[2] - vein_entry[2] <= 0
            ]
            if len(k) > 0:
                dist = max(k)

        else:
            k = [
                reference[2] - vein_entry[2]
                for vein_entry in vein_entries
                if reference[2] - vein_entry[2] >= 0
            ]
            if len(k) > 0:
                dist = min(k)
    dist = abs(dist)
    print(dist)
    return dist
