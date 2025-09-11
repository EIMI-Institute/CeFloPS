from CeFloPS.simulation.sim_configs.abstract_config import AbstractStateMachineVectorized
from CeFloPS.simulation.common.vessel2 import Link,Vessel
import sympy
import numpy as np
from CeFloPS.data_processing.geometric_fun import closest_point_on_circle, random_point_on_circle
from CeFloPS.simulation.common.functions import calculate_distance_euclidian
from numba import jit
import random
from CeFloPS.simulation.sim_configs.transition_functions.vessel_transitions import get_nearest_voi_index, get_nearest_veinend
from scipy.linalg import expm
import CeFloPS.simulation.settings as settings
from CeFloPS.simulation.sim_configs.abstract_config import vectorized_compartment_change_restricted

def normalize(array):
    total = sum(array)
    if total == 0:
        return array  # zero array is already normalized / avoid division by zero
    out = array.copy()
    for i in range(0, len(array)):
        out[i] = out[i] / total
    return out


def flow_walk_step(roi, pos_index, speed=1, already_visited=set(), iteration=0):
    roi_array = roi.vectorfield.get_points()
    roi_points_sorted = roi.geometry.get_points()

    #print(f"At {pos_index}:", roi_array[pos_index][0])
    #print("weights:", roi_array[pos_index][1])
    possible = [(i, w) for i, w in zip(roi_array[pos_index][0], roi_array[pos_index][1]) if i != -1]
    #print("Possible:", possible)
    positive = [(i, w) for i, w in possible if w > 0]
    #print("Positive:", positive)
    selection = [int(i) for i, w in positive]
    #print(f"SELECTION at {pos_index}:", selection)
    if len(selection) == 0 or iteration == -1:
        #print(f"STUCK at {pos_index}!", roi_array[pos_index][0], roi_array[pos_index][1],roi.name)
        return None
    #print(pos_index,"AROUND",roi_array[pos_index][0])
    chances = [
        roi_array[pos_index][1][k]
        for k, i in enumerate(roi_array[pos_index][0])
        if roi_array[pos_index][1][k] > 0 and i != -1
    ]
    #print("Chances",chances)







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


class DefStateMachineVec(AbstractStateMachineVectorized):
    def t_position_time_voi_vectorized(self, indices, simulation, bound_mask):
        """
        Vectorized flow-walk position/time update for VOI cells at `indices`.
        Only update positions for unbound cells.
        """

        #print("CAll VOI", indices,bound_mask)
        # Indices of *unbound* cells
        to_update = indices[~bound_mask]
        if to_update.size > 0:

            # For each, do flow_walk_step on their region/reference index.
            # Example assumes simulation.current_regions and reference_indices arrays.
            roi_arr     = simulation.current_regions[to_update]
            ref_idx_arr = simulation.reference_indices[to_update]
            speeds      = simulation.roi_speeds[roi_arr]

            results = [flow_walk_step(
                    simulation.rois[roi],
                    ref_idx,
                    speed=speed,
                ) for roi, ref_idx, speed in zip(roi_arr, ref_idx_arr, speeds)
            ]

            # Some results may be None (unable to advance)
            update_mask = [r is not None for r in results]
            good_ixs = np.array(to_update)[update_mask]
            res = [r for r in results if r is not None]
            if res:
                new_indices, new_points, times_taken = zip(*res)
                new_indices = np.array(new_indices)
                new_points = np.array(new_points)
                times_taken = np.array(times_taken)

                # Updates für normale Zellen
                simulation.current_positions[good_ixs] = new_points
                simulation.current_times[good_ixs]    += times_taken
                simulation.reference_indices[good_ixs] = new_indices
             #for reaching cells: set reference index to 0, curernt region to blood and current_vessel_id to the one thats at the position thats referenced!
            # in the roi.outlets it should match the 2 element of the 3 tuples in there and then the first item is the vessel and we use the simulation.vessel_to_id
            #check transitions
            returner_mask = np.logical_not(update_mask)
            returner_ixs = np.array(to_update)[ returner_mask]
            #print("returner_ixs:", returner_ixs)
            #print("roi_arr.shape:", roi_arr.shape)
            #print("roi_arr[returner_mask]:", roi_arr[returner_mask])
            if len(returner_ixs) > 0:
                # Setze reference_index auf 0 für Returner
                roi_return = roi_arr[returner_mask]
                for sim_idx, roi_idx in zip(returner_ixs, roi_return):
                    roi = simulation.rois[roi_idx]
                    #update current vessel
                    found_vessobj=None
                    for vessel, pathidx, voigeomidx in roi.outlets:
                        if voigeomidx == simulation.reference_indices[sim_idx]:
                            found_vessobj=vessel

                            simulation.reference_indices[sim_idx] = pathidx
                            break
                    assert found_vessobj!=None, (roi.name, roi.outlets, simulation.reference_indices[sim_idx], simulation.current_positions[sim_idx],roi.geometry.get_points()[voigeomidx])
                    simulation.current_vessel_ids[sim_idx]=simulation.vessel_obj_to_id[found_vessobj]
                    simulation.current_regions[sim_idx]=simulation.blood_region_id
        # For bound cells: advance time, append last position
        to_wait = indices[bound_mask]

        simulation.current_times[to_wait]    += 1




    def p_resolve_link_chances_vectorized(self, links):#TODO actually vectorize?
        """
        Vectorized link probability assignment for cells at `indices`.
        """
        """ debug=False
        for l in links:
            if "sys_pul_art_" in l.source_vessel.associated_vesselname:
                debug=False
                print("vec resolve",[(str(l),type(l)) for l in links])
                print("vec resolve",[(str(l.source_vessel),str(l.target_vessel),type(l)) for l in links]) """
        return  p_link_chances_q(links)


    def t_position_time_vessel_vectorized(
    self, indices, simulation, p_link_chances_q=None, debug=True
    ):
        """
        Vectorized updating of vessel cells—advance position and manage object/index transitions.
        Each cell advances to the next node along its vessel (or transitions to VOI/tissue).
        Handles normal fallback, compartment transitions, inactivation, and logging.
        """
        #print("move",simulation.reference_indices[indices])
        vessel_ids = simulation.current_vessel_ids[indices]
        in_vessel_mask = vessel_ids != -1
        vessel_indices = indices[in_vessel_mask]
        vessel_ids = vessel_ids[in_vessel_mask]
        if len(vessel_indices) == 0:
            return

        # --- main batch arrays for currently advancing cells ---
        starts = simulation.current_positions[vessel_indices].copy()   # shape (M,3)
        ref_indices = simulation.reference_indices[vessel_indices].copy()
        r_indices = simulation.rad_indices[vessel_indices].copy()
        thetas = simulation.thetas[vessel_indices].copy()
        current_time = simulation.current_times[vessel_indices].copy()
        time_left = np.full_like(current_time, fill_value=simulation.TICK_TIME / 1000.0)

        vessels = [simulation.vessel_id_to_obj[v_id] for v_id in vessel_ids]
        volume_ids = np.asarray([simulation.vessel_id_to_volume_id_map[vid] for vid in vessel_ids])
        vessel_profiles = [simulation.vessel_profiles_dict[vid] for vid in volume_ids]

        tick_time = simulation.TICK_TIME / 1000.0
        N = len(ref_indices)
        finished_mask = np.zeros(N, dtype=bool)
        finished_next_objects = [None] * N
        finished_next_indices = [None] * N

        reshuffle_mask = (ref_indices == 0)
        indices_to_reshuffle = np.where(reshuffle_mask)[0]   # Liste der Indizes zum updaten

        for k in indices_to_reshuffle:
            #print("Update r upon entry in", vessels[k].associated_vesselname)
            profile = vessel_profiles[k]   # profile of cell
            rs_len = len(profile['rs'])
            rs_prob_pdf = profile['rs_prob_pdf']
            r_indices[k] = random.choices(
                range(rs_len),
                weights=rs_prob_pdf,
                k=1
            )[0]

        for k in range(N):
            # Get next object/index according to the vessel's topology and simulation logic
            vessel = vessels[k]
            ref_idx = ref_indices[k] + 1  # advance one reference  and step until reaching the corresponding location

            debug=False
            if "sys_pul_art_" in vessel.associated_vesselname:
                ...
                #debug=True
                #print("sys pul art")
            res = vessel.determine_next_target_index(ref_idx, p_link_chances_q,debug=debug)
            # Fallback/Kompatibilität zu alter Rückgabe:
            if type(res) == tuple:
                next_obj, next_idx = res
            else:
                if res == 0:  # go to closest vein
                    _, next_obj = get_nearest_veinend(
                        simulation.end_veins,
                        vessel,
                        vessel.path[len(vessel.path) - 1],
                        simulation.artery_nearest_object_cache,
                    )
                    next_idx = 0
                else:  # go to closest voi (region)
                    next_obj = res
                    next_idx = get_nearest_voi_index(
                        vessel,
                        simulation.artery_nearest_object_cache,
                        roi=next_obj,
                    )
            """ if isinstance(res, tuple):
                next_obj, next_idx = res
            else:
                next_obj, next_idx = res, 0 """
            if ((hasattr(next_obj, "path") and next_obj != vessel)) or not hasattr(next_obj, "path"):
                finished_mask[k] = True
                finished_next_objects[k] = next_obj
                finished_next_indices[k] = next_idx
                continue
            #print("next_obj idx",next_obj, next_idx,next_obj!=vessel)
            # If a next node exists, get its data
            if hasattr(next_obj, "path") and len(next_obj.path) > next_idx:
                if next_obj!=vessel:print("Walking to next vessel")
                next_point = np.array(next_obj.path[next_idx])
                reached_target = False
                while not reached_target:
                    # Fallback for normal if needed:
                    try:
                        normal_candidate = vessel_profiles[k]['normals'][next_idx]
                        if normal_candidate is None or (
                            hasattr(normal_candidate, "__len__") and len(normal_candidate) == 0
                        ):
                            raise ValueError
                        normal = np.asarray(normal_candidate)
                    except Exception:
                        fallback = next_point - starts[k]
                        norm = np.linalg.norm(fallback)
                        if norm < 1e-12:
                            normal = np.array([1.0, 0.0, 0.0])
                        else:
                            normal = fallback / norm

                    # jedes Mal mit aktuellem r_indices/thetas den Zielkreis berechnen!
                    radius_bins = vessel_profiles[k]['rs']
                    radius = radius_bins[r_indices[k]] * 1000
                    theta = thetas[k]
                    target_point = point_on_circle(next_point, normal, radius, theta)
                    speed = (
                        vessel_profiles[k]["v_r"](radius_bins[r_indices[k]]) * 1000
                    )  # mm/s

                    vec_to_target = target_point - starts[k]
                    dist_to_target = np.linalg.norm(vec_to_target)
                    max_move = speed * time_left[k]



                    """ print("Tick time (s):", tick_time)
                    print("Speed (mm/s):", speed)
                    print("Time left (s):", time_left[k])
                    print("Attempted move (mm):", max_move)
                    print("Current position:", starts[k])
                    print("Target position:", target_point) """

                    if debug and False:
                        print(
                            f"Cell {k}: At {starts[k]}, running toward ring {next_idx} [{next_point}]"
                        )
                        print(
                            f"Vec to target: {vec_to_target}, dist: {dist_to_target}, max_move: {max_move}, time_left: {time_left[k]}"
                        )

                    if dist_to_target < 1e-12:
                        # Schon da
                        ref_indices[k] = next_idx
                        reached_target = True
                        break

                    if speed > 0 and max_move >= dist_to_target - 1e-12:
                        # Kann Zielring in time_left erreichen
                        starts[k] = target_point
                        t_used = dist_to_target / (speed + 1e-12)
                        current_time[k] += t_used
                        time_left[k] -= t_used
                        ref_indices[k] = next_idx
                        reached_target = True  # Gehe aus while raus, Zielkreis erreicht
                    else:
                        # Nicht genug Zeit, laufe maximal bis Zeit verbraucht
                        if dist_to_target > 0 and speed > 0:
                            move = vec_to_target / dist_to_target * max_move
                            starts[k] += move
                            current_time[k] += time_left[k]
                            time_left[k] = 0
                        else:
                            # Stecke fest, keine Bewegung möglich (z.B. speed 0)
                            current_time[k] += time_left[k]
                            time_left[k] = 0


                    #BREAK on exceeding time
                    if current_time[k] > settings.TIME_LIMIT_S:
                        finished_mask[k] = True
                        reached_target = True
                        break

                    # Nach jedem Schritt: falls Zeit abgelaufen, aber Ziel noch nicht erreicht
                    if time_left[k] <= 1e-12 and not reached_target:
                        # Resample radius/theta für neuen Teilweg
                        ###r_indices[k] = np.random.randint(0, len(vessel_profiles[k]["rs"]))
                        thetas[k] = np.random.uniform(0, 2 * np.pi)
                        # Compartment transition
                        global_cell_idx = vessel_indices[k]
                        simulation.current_times[global_cell_idx] = current_time[k]
                        new_state = vectorized_compartment_change_restricted(
                            simulation, [global_cell_idx], debug=(False and k < 3), nochange_until_s=settings.CHANGE_LIMIT
                        )[0]
                        simulation.cell_states[global_cell_idx] = new_state
                        if new_state == 2:
                            # Cell eliminated
                            #simulation.current_times[global_cell_idx] = -1
                            simulation.cell_states[global_cell_idx]=2
                            finished_mask[k] = True
                            finished_next_objects[k] = next_obj
                            finished_next_indices[k] = next_idx
                            reached_target = True
                            break
                        else:
                            # Laden neues time_left auf
                            time_left[k] = tick_time
                            # while-Schleife läuft, Target gleich, aber r/theta neu!

        # Write all updates back to simulation
        #print("Before:", simulation.current_positions[vessel_indices])
        simulation.current_positions[vessel_indices] = starts
        #print("After:", simulation.current_positions[vessel_indices])
        simulation.rad_indices[vessel_indices] = r_indices
        simulation.thetas[vessel_indices] = thetas
        simulation.current_times[vessel_indices] = current_time
        simulation.reference_indices[vessel_indices] = ref_indices
        simulation.time_until_tick[vessel_indices] = time_left
        #print("finished_mask:", finished_mask)
        #print("Any finished?", np.any(finished_mask))
        #print("finished_next_objects:", finished_next_objects)
        # Logging, region/vessel transition for "finished" cells
        for idx in np.where(finished_mask) [0]:#either eliminated or change vessel for other voi/vessel
            global_idx = vessel_indices[idx]
            next_object = finished_next_objects[idx]
            next_index = finished_next_indices[idx]
            # Update region and vessel if next_object as appropriate
            if next_object is not None and simulation.current_times[global_idx] != -1:#TODO what with eliminated here?
               # print("REACHED UPDATE CODE")
                simulation.reference_indices[global_idx] = next_index
                if hasattr(next_object, "path"):
                        assert not hasattr(next_object, "geometry")
                        #print("setting next", next_object.associated_vesselname)
                        #print("setting next", simulation.vessel_obj_to_id[next_object])
                        # Still a vessel, stays in blood region
                        simulation.current_regions[global_idx] = simulation.blood_region_id
                        simulation.current_vessel_ids[global_idx] = simulation.vessel_obj_to_id[next_object]
                elif hasattr(next_object, "name"):
                    #print("next is",next_object.name )
                    region_id = simulation.region_name_to_id.get(next_object.name, None)
                    #assert False, (region_id,next_object.name)
                    simulation.current_vessel_ids[global_idx] = -1

                    if region_id is not None:
                        simulation.current_regions[global_idx] = region_id
                    else:
                        print(f"[WARN] Unknown region '{next_object.name}' for cell {global_idx}, setting region to -1")
                        simulation.current_regions[global_idx] = -1
                        assert False
                else:
                    print(f"[WARN] next_object has neither path nor name (cell {global_idx}), setting region to -1")
                    simulation.current_regions[global_idx] = -1
                    assert False
        """ for global_idx in vessel_indices:
            print(f"Writeback for global_idx={global_idx}: vessel={simulation.current_vessel_ids[global_idx]}, region={simulation.current_regions[global_idx]}, ref_idx={simulation.reference_indices[global_idx]}")
        """

    def t_location_VOI_VEIN_vectorized(self, indices, simulation):
        """
        Vectorized version to move cells from VOI to nearest vein endpoint.
        """
        # Find nearest for all
        positions = simulation.current_positions[indices]
        curr_rois = simulation.current_regions[indices]
        nearest = [
            np.argmin([calculate_distance_euclidian(voipoint, pos)
                        for veinpoint, veinvessel, voiindex, voipoint in simulation.rois[roi].veins])
            for pos, roi in zip(positions, curr_rois)
        ]
        for idx, n in zip(indices, nearest):
            vein_info = simulation.rois[simulation.current_regions[idx]].veins[n]
            targetpoint, targetvessel, voiindex, voipoint = vein_info
            travel_to_vein_vec(idx, targetpoint, targetvessel, simulation)
            # Mark new region:
            simulation.current_regions[idx] = simulation.blood_region_id

    def p_substate_vectorized(self, indices, simulation, sim_times):
        """
        Calls transition-chance logic for all cells at indices.sim_time
        """
        change_time = 350  # hardcoded for now
        results = [
            (p_substate_no_change(simulation, ix, -1) if sim_time < change_time
             else p_substate_continuous_markov(simulation, ix, min(simulation.cell_times[ix]-change_time, sim_time)))
            for ix, sim_time in zip(indices, sim_times)
        ]
        # Unzip to (ps, cs)
        ps, cs = zip(*results)
        # Convert to arrays as needed
        return np.array(ps), np.array(cs)
    def t_substate_vectorized(self):
        pass

import numpy as np

""" def get_next_target_vector_vectorized(
    start_points,          # (N, 3)
    rindices,              # (N,)
    next_objects,          # (N,)
    next_indices,          # (N,)
    volume_ids,               # (N,) or None
    vessel_profiles_dict,  # {vol.id: {'v_r': ..., 'rs': ...}}
    TICK_TIME,             # float
):
    N = start_points.shape[0]
    next_points = np.stack([
        np.asarray(obj.path[idx], dtype=np.float64) for obj, idx in zip(next_objects, next_indices)
    ])
    speeds = np.array([
        vessel_profiles_dict[vol_idx]['v_r'](
            vessel_profiles_dict[vol_idx]['rs'][r_idx]
        ) * 1000
        for vol_idx, r_idx in zip(volume_ids, rindices)
    ])

    distance_vectors = next_points - start_points
    distances = np.linalg.norm(distance_vectors, axis=1)
    max_move_dist = speeds * TICK_TIME



    # Move fractions: fraction of distance to target you can take this tick
    move_fractions = np.minimum(1.0, max_move_dist / np.clip(distances, 1e-12, None))
    move_vectors = distance_vectors * move_fractions[:,None]
    moved_distances = np.linalg.norm(move_vectors, axis=1)
    time_lefts = TICK_TIME - (moved_distances / speeds)

    just_finished = (move_fractions == 1.0)   # covers both perfectly and early arrivals

    return move_vectors, time_lefts, just_finished """

def point_on_circle(center, normal, radius, theta):
    """
    Returns a 3D point on a circle given by center, normal, radius, and angle theta.
    center: (3,)
    normal: (3,)
    radius: float
    theta: float, angle [rads]
    """
    normal = np.asarray(normal)
    center = np.asarray(center)
    normal = normal / np.linalg.norm(normal)
    # Get 2 orthogonal vectors in plane
    if np.allclose(normal, [0, 0, 1]):
        u = np.array([1, 0, 0])
    else:
        u = np.cross([0, 0, 1], normal)
        if np.linalg.norm(u) < 1e-8:
            u = np.array([1, 0, 0])
        u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    return center + radius * (np.cos(theta) * u + np.sin(theta) * v)


@jit(nopython=True)
def calculate_movement_vector(start_point, target_point, speed, interval):
    # distance vector from start to target
    distance_vector = target_point - start_point

    # euclidean distance to the target point is vector norm
    distance = np.linalg.norm(distance_vector)

    # maximum movement distance allowed within the time interval
    max_move_distance = speed * interval

    # determine movement vector
    if distance > max_move_distance:
        # Normalize distance vector to turn it into a unit vector
        direction = distance_vector / distance
        # scale direction by the maximum distance we can move
        movement_vector = direction * max_move_distance
    else:
        # If the distance to the target is less than or equal to maximum movement distance,
        # move directly towards the target
        movement_vector = distance_vector

    # Calculate the total distance possible in one move
    total_distance_possible = np.linalg.norm(movement_vector)

    # Calculate the initial number of steps needed
    if total_distance_possible > 0:
        steps_needed = int(np.ceil(distance / total_distance_possible))
        time_per_step = (
            total_distance_possible / speed
        )  # Time taken to move this distance
    else:
        steps_needed = 0  # No movement possible if movement_vector is zero
        time_per_step = 0

    # Updated steps considering a maximum check of the last two steps
    current_point = start_point.copy()
    updated_steps = 0

    for step in range(steps_needed):
        # Calculate potential new position after the movement
        new_point = current_point + movement_vector

        # Check if we are making valid progress toward the target
        if updated_steps >= steps_needed - 2:  # Only check for last two steps
            # Check if moving to new_point is actually closer than current_point
            current_distance = np.linalg.norm(target_point - current_point)
            new_distance = np.linalg.norm(target_point - new_point)

            # If the new distance doesn't decrease sufficiently, stop the process
            if new_distance >= current_distance:
                break

        # Move to the new point
        current_point = new_point
        updated_steps += 1

    return movement_vector, updated_steps, time_per_step


def p_link_chances_q(links, simulation=None):
        # define tlink chances if tlinks are included
        # chances=[link.target_vessel.get_volume_by_index(link.target_index) for link in links if type(link)==Link]
        normal_links = [link for link in links if type(link) == Link]
        no_vols, q_chances = check_if_all_options_have_volumes(
            [
                (link.target_vessel, link.target_index, link.source_index)
                for link in normal_links
            ]
        )
        chances = normalize(q_chances)
        # tlinks:
        tlinks = [link for link in links if type(link) != Link]
        t_chances = [
            (
                settings.TCHANCE
                if l.source_index != len(l.source_vessel.path) - 1
                else l.target_tissue.volume_ml
            )
            for l in tlinks
        ]
        t_chances = normalize(t_chances)
        #print(f"[PROBABILITY LINKS] from {[link.source_vessel.associated_vesselname[-32:] for link in normal_links][0]} --> {normal_links + tlinks, chances + t_chances}")
        #print(f"[PROBABILITY LINKS - TARGETS] {[link.target_vessel.associated_vesselname[-32:] for link in normal_links] + [l.target_tissue.name for l in tlinks]}")
        return normal_links + tlinks, chances + t_chances

def check_if_all_options_have_volumes(selection_to_choose_from):
    q_chances = []
    no_vols = False  # set to True if not all have vols
    for (
        target_vesselq,
        target_indexq,
        source_indexq,
    ) in selection_to_choose_from:
        try:
            if (
                target_vesselq.get_volume_by_index(target_indexq) != None
                and type(
                    target_vesselq.get_volume_by_index(target_indexq).get_symval(
                        target_vesselq.get_volume_by_index(target_indexq).Q_1
                    )
                )
                != sympy.Symbol
            ):
                q_chances.append(
                    target_vesselq.get_volume_by_index(target_indexq).get_symval(
                        target_vesselq.get_volume_by_index(target_indexq).Q_1
                    )
                )
            else:
                no_vols = True
        except:
            no_vols = True
            break
    return no_vols, q_chances


def get_next_target_vector(
        self,
        start_point,
        rindex,
        next_object,
        next_index,
        start_refpoint=None,
        normal=None,
        random=False,
        logging=True,
    ):
        """get_next_target_vector Calculate a next target vector as

        Args:
            index (_type_): _description_
        Returns:
            tuple: (next_trav_object,index_at_next_object)
        """
        vol= self.volumes[0]
        assert len(self.volumes)==1, "We assume one vessel one volume relationship, else we have to pass the vol to use to this function!"
        if type(next_object) is not Vessel:
            try:
                next_point = next_object.geometry.get_points()[next_index]
            except:
                assert False, next_object.__dict__
        else:
            if settings.SHOW_R_CHANGES and logging:
                if normal is None:
                    normal = (
                        -np.asarray(next_object.path[next_index], dtype=np.float64)
                        + start_refpoint
                    )
                # logger.print(np.asarray(next_object.path[next_index],dtype=np.float64), np.asarray(start_point,dtype=np.float64),self.profiles[(vol.id, "rs")][rindex],normal)
                # logger.print("r",self.profiles[(vol.id, "rs")][rindex])
                if vol != None:
                    if tuple(start_point) != tuple(start_refpoint):
                        next_point = closest_point_on_circle(
                            np.asarray(next_object.path[next_index], dtype=np.float64),
                            np.asarray(start_point, dtype=np.float64),
                            self.profiles[(vol.id, "rs")][rindex],
                            normal,
                        )
                    else:
                        next_point = random_point_on_circle(
                            np.asarray(next_object.path[next_index], dtype=np.float64),
                            normal,
                            self.profiles[(vol.id, "rs")][rindex],
                        )
                else:
                    # assert False, "Not implemented yet"
                    next_point = next_object.path[next_index]
            else:
                next_point = next_object.path[next_index]

        if vol == None:
            # try if next object has a vol:
            try:
                vol = next_object.get_volume_by_index(next_index)
                speed = (
                    next_object.profiles[(vol.id, "v_r")](
                        next_object.profiles[(vol.id, "rs")][rindex]
                    )
                    * 1000
                )
            except Exception as e:
                # assert False, e
                # Fallback to interp speed
                speed = interpol_speed(
                    next_object.type, next_object.diameters[next_index]
                )
        else:
            try:
                speed = (
                    vol.vessel.profiles[(vol.id, "v_r")](
                        vol.vessel.profiles[(vol.id, "rs")][rindex]
                    )
                    * 1000
                )
            except:
                vol.vessel.register_volumes()#try creating llambda functions again...
                try:
                    speed = (
                        vol.vessel.profiles[(vol.id, "v_r")](
                            vol.vessel.profiles[(vol.id, "rs")][rindex]
                        )
                        * 1000
                    )
                except:
                    assert False, (vol.__dict__,vol.vessel.profiles)
        # random_point_on_circle(center, normal, radius)
        movement_vector, updated_steps, time_per_step = calculate_movement_vector(
            np.asarray(start_point),
            np.asarray(next_point),
            speed,
            interval=0.005,  # TODO
        )
        return movement_vector, updated_steps, time_per_step
