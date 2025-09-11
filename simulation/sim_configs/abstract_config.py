from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from CeFloPS.simulation.common.functions import (
    normalize,
)
import CeFloPS.simulation.settings as settings
from CeFloPS.simulation.common.vessel_functions import standard_speed_function
from CeFloPS.simulation.common.vessel2 import TissueLink, Link, Vessel
from CeFloPS.simulation.sim_configs.transition_functions.three_dim_connection import (
    travel_to,
)
import random
from scipy.linalg import expm
from CeFloPS.simulation.common.functions import calculate_distance_euclidian

class Location(Enum):
    VOI = 1

    ARTERY = 2

    VEIN = 3


class AbstractStateMachine(ABC):
    @abstractmethod
    def t_position_time_voi(self, cell, bound, simulation):
        """t_position_time_voi Transition position and time for voi locations depending on the state of the cell.

        Args:
            cell (Cell): cell-object for cellstate
            bound (bool): Indicates whether the current substate allows for movement
            simulation (Simulation): simulation-object for being able to obtain simulation data
        """
        return

    @abstractmethod
    def p_resolve_link_chances(self, links, simulation=None):
        """p_resolve_link_chances Abstract probability function for connections between vessels and tissues

        Args:
            links (list of TissueLink and Link objects): Links between vessels or Vessels and Tissues
            simulation (Simulation): simulation-object for being able to obtain simulation data
        """
        ...

    @abstractmethod
    def t_position_time_vessel(self, cell, simulation, pfun):
        """t_position_time_vessel Transition position and time in vessels, currently also implements the location change from vessels to vois

        Args:
            cell (Cell): cell-object for cellstate
            simulation (Simulation): simulation-object for being able to obtain simulation data
            pfun (function): Probability function mapping links to probabilities for them
        """
        return

    def t_cellstate(self, cell, simulation):
        """t_cellstate Transition Method for cellstate transitions

        Args:
            cell (Cell): cell-object for cellstate
            simulation (Simulation): simulation-object for being able to obtain simulation data
        """
        while cell.time < simulation.time:
            #print(simulation.time,"cell",cell.get_state())
            # VOIs
            if cell.roi != cell.blood:  # Location.VOI:
                # do substate transitions only if cell has contact to tissue -> is in voi
                bound, location_change = self.t_substate(
                    cell,
                    self.p_substate,
                    simulation,
                    time_taken=cell.time_taken(cell.roi),
                )

                if location_change:
                    #print("Change TTV",cell.roi.name)
                    self.t_location_VOI_VEIN(cell)
                    cell.roi = cell.blood
                else:
                    k=self.t_position_time_voi(cell, bound, simulation)
                    #print(k)
                    if k==-1:
                        self.t_location_VOI_VEIN(cell)
                        cell.roi = cell.blood
                        #assert False,cell.roi.name
                continue
            # Vessels
            # if at end of vessel or at decisionpoint use transition of location, else move inside of vessel
            # assert len(cell.current_position) == 2, cell.current_position
            else:
                # cell in blood
                # no locationchange in a vessel possible, only at ends, no substatechange possible.
                new_position = self.t_position_time_vessel(
                    cell, simulation, self.p_resolve_link_chances
                )  # update cell to step one time
                if new_position == None:
                    # wait
                    return
                # changelocation if was at end and new position is not in vessels anymore
                next_target_object, next_target_index = new_position
                if hasattr(next_target_object, "geometry"):
                    # #print("ntarget",type(next_target_object))
                    new_position = next_target_object.geometry.get_points()[
                        next_target_index
                    ]
                    cell.roi = next_target_object
                    # cell.radint_in_cell[0]=None
                cell.reference_index = next_target_index
                cell.current_position = new_position

    def change_comp(self, cell, n_comp):
        """change_comp Method to change compartment adding and subtracting cells from blood and VOI counts

        Args:
            cell (Cell): cell that changes compartment
            n_comp (None or Compartment): new compartment, None if blood
        """
        if cell.compartment == None:
            cell.blood.sub_cell()  # out of blood/capillaries into coompartment
        else:
            cell.compartment.sub_cell()
        #print("NEWCOMP", str(n_comp)[0:100])
        ##print("CURRENT_ROI_CELL", cell.__dict__)
        n_comp.add_cell()  # this is list of all vessels!TODO

        comp_index = 0
        if n_comp != cell.blood:
            if len(n_comp.k_outs) == 2:
                comp_index = 1
            else:
                comp_index = 2
        if n_comp == cell.blood:
            if cell.compartment == None:
                name = "STAYBLOOD"
            else:
                name = n_comp.name
        else:
            name = n_comp.descriptor
        # #print(name, "->", comp_index, cell.time) DEBUGGING
        cell.comp_change_times[cell.time] = comp_index

        if n_comp == cell.blood:
            cell.compartment = None
            # cell.roi = cell.blood
        else:
            cell.compartment = n_comp

    # ----------------Location Transition-----------------
    @abstractmethod
    def t_location_VOI_VEIN(self, cell):
        """t_location_VOI_VEIN transition Location from VOI to Vein, directly transition cellstates

        Args:
            cell (Cell): cell-object for cellstate
        """
        ...

    # ----------------substate Transition-----------------
    def t_substate(self, cell, p_substate, simulation, time_taken) -> (bool, bool):
        """t_substate Transition for substates; compartment changes, implemented for 2TCM

        Args:
            cell (Cell): cell-object for cellstate
            p_substate (function): function for stochastic state transition probability calculation
            simulation (Simulation): simulation-object for being able to obtain simulation data, not used for most methods
            time_taken (float): time to use to determine compartment interaction

        Returns:
            (bool,bool): First bool sets if a cell is bound to tissue and second signals a location change
        """

        if cell.loc != "voi":  # sample locations
            cell.update_cell_cvolume("vois", cell.time)  # sample locations
        t = simulation.time / 60
        ps, cs = p_substate(cell, simulation, time_taken)
        Ca, C1, C2 = cs# so CA contains  TODO vessels?

        p = normalize(ps)
        ##print(f"[COMPARTMENT CHANGE] {cs,p}")
        if sum(p) != 0:
            chosen_compartment = np.random.choice(cs, 1, p=p)[0]
        else:
            chosen_compartment = Ca  # TODO replace by getting the current compartment, this is used to deny all changes out of blood usually

        # if chosen compartment is blood, move, else do not move and unregister if in c2
        if cell.compartment != chosen_compartment:  # changed compartment
            old_roi = cell.roi
            old_compartment=cell.compartment
            #print("change comp from statemachine", "olr_roi, chosen", old_roi.name, type(chosen_compartment))
            self.change_comp(cell, chosen_compartment)
            if chosen_compartment == C2:
                simulation.unregister(cell)
                return True, False  # in c2 bound and no locationchange
            elif chosen_compartment == C1:
                return True, False  # in c1 bound and no locationchange
            elif chosen_compartment == Ca:
                # change to blood
                #print("Old_compartment CA",old_compartment==Ca,old_compartment)
                if not settings.RETURN_AFTER_C1_RELEASE or old_compartment==None:
                    # change to blood in voi
                    cell.roi = old_roi
                    return (
                        False,
                        False,
                    )  # in Ca no longer bound but no locationchange if not returnsetting
                else:
                    return True, True  # in Ca with a locationchange to veins
            else:
                assert False  # should not occur
        else:
            if chosen_compartment == Ca:
                return False, False  # in Ca with no locationchange
            else:
                return True, False  # in c1 or c2

    @abstractmethod
    def p_substate(self, cell, simulation, time_taken) -> (list, list):
        """p_substate Abstract transitionchances for substates

        Args:
            cell (Cell): cell-object for cellstate
            p_substate (function): function for stochastic state transition probability calculation
            simulation (Simulation): simulation-object for being able to obtain simulation data, not used for most methods
            time_taken (float): time to use to determine compartment interaction
        Returns:
            (list,list): A list of compartments and one of the chances to transition into them
        """
        ...


class AbstractStateMachineVectorized(ABC):
    @abstractmethod
    def t_position_time_voi_vectorized(self, indices, simulation, bound_mask):
        """
        Vectorized version: given a list/array of indices for current cells,
        operate on those in the simulation arrays, and update them accordingly.
        'bound_mask' indicates which of those indices represent bound-state cells.
        """
        pass

    @abstractmethod
    def p_resolve_link_chances_vectorized(self, links):
        """
        Compute connection probabilities in vectorized fashion for a vector of cell indices.
        Returns array of probabilities (or selections).
        """
        pass

    @abstractmethod
    def t_position_time_vessel_vectorized(self, indices, simulation, pfun):
        """
        Vectorized version for all vessel cells (by indices in main arrays).
        Returns updated positions, indices, etc.
        """
        pass

 




    def traverse_to_veinentry_fw(self,returned_indices, returned_regions, simulation):
            """
            Do flow-walk step for all returned indices.
            Log new times/positions.
            Only allow state transitions to elimination.
            If the step result is None, the cell has reached its target and should not be advanced further.
            """
            def flow_walk_step(roi, pos_index, speed=1, already_visited=set(), iteration=0):
                roi_array = roi.vectorfield.get_points()
                roi_points_sorted = roi.geometry.get_points()
                selection = [
                    int(i)
                    for k, i in enumerate(roi_array[pos_index][0])  # nbs
                    if roi_array[pos_index][1][k] > 0
                    and i != -1
                    and roi_array[pos_index][1][k] not in already_visited
                ]  # possible indexes from current index
                # print("AROUND",roi_array[pos_index][0])
                chances = [
                    roi_array[pos_index][1][k]
                    for k, i in enumerate(roi_array[pos_index][0])
                    if roi_array[pos_index][1][k] > 0 and i != -1
                ]
                if len(selection) == 0 or iteration == -1:

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
            returned_indices = np.array(returned_indices)
            to_update = returned_indices.copy()
            # Boolean mask: which cells are still being updated/advanced
            still_advancing = np.ones(to_update.shape, dtype=bool)
            tissue_tags = {idx: simulation.region_id_to_roi[reg].name
                        for idx, reg in zip(returned_indices, returned_regions)}

            while np.any(still_advancing):
                # Active indices this round
                current_indices = to_update[still_advancing]
                mask_in_indices = np.isin(returned_indices, current_indices)
                roi_arr     = returned_regions[mask_in_indices]
                #print("returning from", roi_arr)
                ref_idx_arr = simulation.reference_indices[current_indices]
                speeds      = simulation.roi_speeds[roi_arr]

                # Step
                results = [flow_walk_step(
                                simulation.rois[roi],
                                ref_idx,
                                speed=speed,
                        ) for roi, ref_idx, speed in zip(roi_arr, ref_idx_arr, speeds)
                ]

                # Good cells are those where result is not None
                good_mask = np.array([r is not None for r in results])
                good_indices = current_indices[good_mask]
                if good_mask.any():
                    new_indices, new_points, times_taken = zip(*[r for r in results if r is not None])
                    simulation.current_positions[good_indices] = new_points
                    simulation.current_times[good_indices]    += times_taken
                    simulation.reference_indices[good_indices] = new_indices
                    #print("Moved", times_taken, "to", new_points, "from",
                    #    simulation.current_positions[good_indices])
                # If no cell could be advanced in this round, break loop
                else:
                    break

                # For cells that reached their target (result is None), we want to "finish" their advance:
                # We don't set time to -1, just don't consider them in the next loop
                reached_target_mask = ~good_mask  # opposite of good_mask
                # Update still_advancing: set to False for any index that returned None this round
                update_indices_in_bool = np.where(still_advancing)[0] # which indices in full to_update
                still_advancing[update_indices_in_bool[reached_target_mask]] = False

                # Now check if any were eliminated (state==2) among the updated/advanced ones
                if good_indices.size > 0:
                    new_states = vectorized_compartment_change_restricted(simulation, good_indices, debug=False, nochange_until_s=settings.CHANGE_LIMIT)
                    simulation.cell_states[good_indices] = new_states
                    eliminated_mask = (simulation.cell_states[good_indices] == 2)
                    eliminated_indices = np.array(good_indices)[eliminated_mask]
                    #simulation.current_times[eliminated_indices] = -1
                    # Remove eliminated cells from further updating too
                    still_advancing[np.isin(to_update, eliminated_indices)] = False


                # Logging every cell that was active this round (regardless of eliminated or reached target)
                for cell_idx in current_indices:
                    tag_code = simulation.tag_to_int(tissue_tags[cell_idx])
                    if simulation.cell_states[cell_idx]==2:
                        tag_code=-1
                    if simulation.last_tags[cell_idx] != tag_code:
                        simulation.tag_log.log(
                            (cell_idx, simulation.current_times[cell_idx], tag_code)
                        )
                        simulation.last_tags[cell_idx] = tag_code

                    # --- Log position only if changed   ---
                    pos_has_changed = True  
                    if pos_has_changed and simulation.tracked_mask[cell_idx]:
                        simulation.position_log.log(
                            (cell_idx, simulation.current_times[cell_idx], *simulation.current_positions[cell_idx])
                        )

                    # --- Log region if changed ---
                    region_id = simulation.current_regions[cell_idx]
                    if simulation.last_regions[cell_idx] != region_id:
                        simulation.region_log.log(
                            (cell_idx, simulation.current_times[cell_idx], region_id)
                        )
                        simulation.last_regions[cell_idx] = region_id

                    # log state

                    if simulation.cell_states[cell_idx]!=simulation.last_states[cell_idx]:
                        simulation.last_states[cell_idx]=simulation.cell_states[cell_idx]
                        simulation.state_log.log(
                            (cell_idx, simulation.current_times[cell_idx],simulation.cell_states[cell_idx] )
                        ) 

            # mapping indices
            for sim_idx, prev_region_id in zip(to_update, returned_regions):
                if simulation.cell_states[sim_idx]!=2:
                    roi = simulation.region_id_to_roi[prev_region_id]
                    found = False
                    for vessel, pathidx, voigeomidx in roi.outlets:
                        if voigeomidx == simulation.reference_indices[sim_idx]:
                            simulation.current_vessel_ids[sim_idx] = simulation.vessel_obj_to_id[vessel]
                            simulation.reference_indices[sim_idx] = pathidx
                            found = True
                            break
                    if not found:
                        print(f"WARNING: No vessel found for cell {sim_idx} at exit point!")
                        print(simulation.current_regions[sim_idx],simulation.cell_states[sim_idx])
                else:
                    #simulation.current_times[sim_idx]=-1

                    simulation.current_vessel_ids[sim_idx]=-1

    def t_cellstate_vectorized(self, indices, simulation):
        """
        Vectorized state transition routine for a set of cell indices.
        Handles both VOI (tissue) and vessel cells according to current region state.
        """
        """ print("-----------------")
        print("Cvessel",simulation.current_vessel_ids[indices])
        print("CIndex",simulation.reference_indices[indices])
        print("CTime",simulation.current_times[indices])
        print("CRegion",simulation.current_regions[indices],"name",simulation.region_id_to_roi[simulation.current_regions[indices][0]].name)
        print("-----------------") """
        #print(1)
        DEBUGLOWNUM=False
        """ if len(indices)<=2:
            DEBUGLOWNUM=True """
        cur_times   = simulation.current_times
        if DEBUGLOWNUM:print(simulation.reference_indices[indices],simulation.current_vessel_ids[indices])
        is_vessel = simulation.current_vessel_ids[indices]!=-1 #(cur_regions[indices] != simulation.blood_region_id)
        is_voi = ~is_vessel

        # ---- Handle VOI/tissue cells by compartment stepping + move ---
        if np.any(is_voi):
            voi_indices = indices[is_voi]
            prev_states = simulation.cell_states[voi_indices].copy()
            #print("Walking in VOIS!", len(voi_indices),voi_indices,simulation.current_times[voi_indices],simulation.current_positions[voi_indices])

            new_states = vectorized_compartment_change_restricted(simulation, voi_indices,debug=False, nochange_until_s=settings.CHANGE_LIMIT)
            simulation.cell_states[voi_indices] = new_states 



            # Define "bound" mask for C1, C2 (or others as your PK model defines):
            BOUND_STATES = set([3 + 2*i for i in range(simulation.n_organs)]) \
                        | set([4 + 2*i for i in range(simulation.n_organs)]) \
                        | set([2])
            bound_mask = np.isin(new_states, list(BOUND_STATES))
            if DEBUGLOWNUM: print("Bound",bound_mask)
            not_eliminated = simulation.cell_states[voi_indices] != 2

            # Move to blood (vessel) if required
            newly_in_blood = (new_states == 1) & (prev_states != 1) & not_eliminated
            returned = set()
            if np.any(newly_in_blood):
                returned_indices = voi_indices[newly_in_blood]
                returned_regions = simulation.current_regions[returned_indices].copy()  # Use copy to ensure independence

                # Update region to blood for those indices
                simulation.current_regions[returned_indices] = simulation.blood_region_id

                # Add these indices to "returned"
                returned.update(returned_indices)

                # Pass both indices and regions to your function
                self.traverse_to_veinentry_fw(returned_indices, returned_regions, simulation)

            # Prepare for the next part
            voi_indices_arr = np.array(voi_indices)  # for indexing
            not_returned_mask = ~np.isin(voi_indices_arr, list(returned))  # filter out returned
            not_minus_one_mask = simulation.current_times[voi_indices_arr] != -1
            final_mask = not_returned_mask & not_minus_one_mask   

            final_indices = voi_indices_arr[final_mask]



            #  filter the bound_mask to those indices
            bound_mask_filtered = bound_mask[final_mask]
            currently_in_blood_no_vessel = (
                (simulation.current_regions[final_indices] == simulation.blood_region_id) &
                (simulation.current_vessel_ids[final_indices] == -1)
            )

            final_indices = final_indices[~currently_in_blood_no_vessel]
            bound_mask_filtered = bound_mask_filtered[~currently_in_blood_no_vessel]
            if DEBUGLOWNUM: print("Bound_parameter",bound_mask_filtered)
            # Now call with this filtered list
            self.t_position_time_voi_vectorized(
                final_indices, simulation, bound_mask_filtered
            )
            blood_no_vessel_indices = voi_indices_arr[
                (simulation.current_regions[voi_indices_arr] == simulation.blood_region_id) &
                (simulation.current_vessel_ids[voi_indices_arr] == -1)
            ]
            if blood_no_vessel_indices.size > 0:
                states = simulation.cell_states[blood_no_vessel_indices]
                times  = simulation.current_times[blood_no_vessel_indices]
                all_eliminated = np.all((states == 2) | (times == -1))
                if not all_eliminated:
                    print(simulation.current_regions[blood_no_vessel_indices])
                    print(states)
                    print(times)
                    print(simulation.current_vessel_ids[blood_no_vessel_indices])
                    assert False


            for idx in voi_indices:
                region_id = simulation.current_regions[idx]
                roi = simulation.region_id_to_roi[region_id]
                #print(f" current_regions[{idx}] = {region_id}, roi = {roi}")

                if simulation.current_vessel_ids[idx] != -1 and simulation.cell_states[idx]!=2:
                    vobj = simulation.vessel_id_to_obj[simulation.current_vessel_ids[idx]]
                    ref_idx = simulation.reference_indices[idx]
                    node_tags = vobj.node_tags
                    #print(f"idx={idx}, vessel_id={simulation.current_vessel_ids[idx]}, len(node_tags)={len(node_tags)}, ref_idx={ref_idx}")
                    if 0 <= ref_idx < len(node_tags):
                        tag = node_tags[ref_idx]
                    else:
                        print(f"WARNING: Index {ref_idx} out of bounds for vessel {simulation.current_vessel_ids[idx]} with {len(node_tags)} tags.")
                        print(simulation.current_regions[idx],simulation.cell_states[idx])
                        tag = simulation.last_tags[idx]
                else:
                    tag = simulation.region_id_to_roi[simulation.current_regions[idx]].name
                    cell_idx=idx
                    #if simulation.cell_states[idx]==2: simulation.current_times[idx]=-1
                    tag_code = simulation.tag_to_int(tag)
                    if simulation.cell_states[idx]==2:
                        tag_code=-1

                    if simulation.last_tags[cell_idx] != tag_code:
                        simulation.tag_log.log(
                            (cell_idx, simulation.current_times[cell_idx], tag_code)
                        )
                        simulation.last_tags[cell_idx] = tag_code
                    simulation.last_tags[cell_idx]=tag_code

                    # --- Log position only if changed---
                    pos_has_changed = True  
                    if pos_has_changed and simulation.tracked_mask[cell_idx]:
                        simulation.position_log.log(
                            (cell_idx, simulation.current_times[cell_idx], *simulation.current_positions[cell_idx])
                        )

                    # --- Log region if changed ---
                    region_id = simulation.current_regions[cell_idx]
                    if simulation.last_regions[cell_idx] != region_id:
                        simulation.region_log.log(
                            (cell_idx, simulation.current_times[cell_idx], region_id)
                        )
                        simulation.last_regions[cell_idx] = region_id
                    # log state

                    if simulation.cell_states[cell_idx]!=simulation.last_states[cell_idx]:
                        simulation.last_states[cell_idx]=simulation.cell_states[cell_idx]
                        simulation.state_log.log(
                            (cell_idx, simulation.current_times[cell_idx],simulation.cell_states[cell_idx] )
                        )
 
        # ---- Handle vessel cells by vessel walker ---
        if np.any(is_vessel):
            vessel_indices = indices[is_vessel]
            self.t_position_time_vessel_vectorized(
                vessel_indices,
                simulation,
                self.p_resolve_link_chances_vectorized if hasattr(self, "p_resolve_link_chances_vectorized") else None
            )
            for idx in vessel_indices:
                if simulation.current_vessel_ids[idx] != -1 and simulation.cell_states[idx]!=2:
                    vobj = simulation.vessel_id_to_obj[simulation.current_vessel_ids[idx]]
                    ref_idx = simulation.reference_indices[idx]
                    node_tags = vobj.node_tags
                    #print(vobj.associated_vesselname)
                    #print(f"idx={idx}, vessel_id={simulation.current_vessel_ids[idx]}, len(node_tags)={len(node_tags)}, ref_idx={ref_idx}")
                    if 0 <= ref_idx < len(node_tags):
                        tag = node_tags[ref_idx]
                    else:



                        print(f"WARNING: Index {ref_idx} out of bounds for vessel {simulation.current_vessel_ids[idx]} with {len(node_tags)} tags.")
                        print(simulation.current_regions[idx],simulation.cell_states[idx])
                        tag = simulation.last_tags[idx]

                else:
                    tag = simulation.region_id_to_roi[simulation.current_regions[idx]].name
                #if simulation.cell_states[idx]==2: simulation.current_times[idx]=-1
                tag_code = simulation.tag_to_int(tag)
                if simulation.cell_states[idx]==2:
                        tag_code=-1
                cell_idx=idx
                if simulation.last_tags[cell_idx] != tag_code:
                        simulation.tag_log.log(
                            (cell_idx, simulation.current_times[cell_idx], tag_code)
                        )
                        simulation.last_tags[cell_idx] = tag_code
                simulation.last_tags[cell_idx]=tag_code

                # --- Log position only if changed (or always, if one wants all positions) ---
                pos_has_changed = True # TODO timer based or something
                if pos_has_changed and simulation.tracked_mask[cell_idx]:
                    simulation.position_log.log(
                        (cell_idx, simulation.current_times[cell_idx], *simulation.current_positions[cell_idx])
                    )

                # --- Log region if changed ---
                region_id = simulation.current_regions[cell_idx]
                if simulation.last_regions[cell_idx] != region_id:
                    simulation.region_log.log(
                        (cell_idx, simulation.current_times[cell_idx], region_id)
                    )
                    simulation.last_regions[cell_idx] = region_id
                # log state

                if simulation.cell_states[cell_idx]!=simulation.last_states[cell_idx]:
                    simulation.last_states[cell_idx]=simulation.cell_states[cell_idx]
                    simulation.state_log.log(
                        (cell_idx, simulation.current_times[cell_idx],simulation.cell_states[cell_idx] )
                    )
    def change_comp_vectorized(self, indices, new_comps, simulation):
        """
        Vectorized compartment change for cells at indices: remove/add cells in VOI/blood.
        new_comps: array of new compartment ids
        """
        old_comps = simulation.current_compartments[indices] 
        for comp_id in np.unique(old_comps):
            simulation.compartment_counts[comp_id] -= np.sum(old_comps == comp_id)
        for comp_id in np.unique(new_comps):
            simulation.compartment_counts[comp_id] += np.sum(new_comps == comp_id)
        # Store the new compartment:
        simulation.current_compartments[indices] = new_comps
 

    @abstractmethod
    def t_location_VOI_VEIN_vectorized(self, indices, simulation):
        """
        Move VOI cells at these indices to blood/vessel (location transition).
        Update .current_regions, .current_positions, etc.
        """
        pass
    @abstractmethod
    def t_substate_vectorized(self, indices, simulation):
        """
        Vectorized compartment/substate transition for cells at these indices.

        Returns:
            bound_mask: Bool array (shape=indices.shape)
            location_change_mask: Bool array (shape=indices.shape)
        """
        pass
        # All cells: get current time_taken, p_substate, etc. in vectorized fashion
        times = simulation.current_times[indices]
        # For each cell, compute p_substate result:
        ps, cs = self.p_substate_vectorized(indices, simulation, times)
        # Normalize ps rows (per-cell)
        p_choice = (ps.T / np.sum(ps, axis=1)).T

        rng = np.random.default_rng()
        chosen = np.array([np.random.choice(cs[i], p=p_choice[i]) for i in range(len(indices))])

        # Figure out bound/location-change for each
        comp = simulation.current_compartments[indices]
        CA, C1, C2 = simulation.compartment_ids  # ints, provided by simulation

        bound_mask = chosen != CA
        location_change_mask = np.zeros(len(indices), dtype=bool)

        change_indices = (comp != chosen)
        # For those with compartment change,
        # if chosen is C2, unregister (simulate "deletion" or removal from further simulation)
        C2_mask = (chosen == C2)
        C1_mask = (chosen == C1)
        CA_mask = (chosen == CA)

        # Remove from simulation if going to C2
        to_remove = change_indices & C2_mask
        simulation.unregister_indices(indices[to_remove])

        # For C1, bound, no location change
        # For CA, handle location change depending on your configuration
        # (not shown here: see your original code, e.g., settings.RETURN_AFTER_C1_RELEASE etc.)

        # Return bound_mask, location_change_mask (according to your logic)
        # Fill location_change_mask based on your policy:
        # location_change is for CA with config/logic conditions
        return bound_mask, location_change_mask

    @abstractmethod
    def p_substate_vectorized(self, indices, simulation, times):
        """
        Vectorized p_substate transition calculation for all indices.

        Returns:
            ps_list: [len(indices), K] array (transition probabilities per cell)
            cs_list: [len(indices), K] array/lists (compartment IDs per cell)
        """
        pass

import numpy as np
from scipy.linalg import expm

def vectorized_compartment_change_restricted(sim, cell_indices, debug=True, nochange_until_s=False):
    if not hasattr(vectorized_compartment_change_restricted, 'region_visit_counts'):
        vectorized_compartment_change_restricted.region_visit_counts = {}
        vectorized_compartment_change_restricted.max_visits = 1     # initialized to 1 to avoid divide-by-zero

    visit_counts = vectorized_compartment_change_restricted.region_visit_counts
    max_ignore_prob = 0.2
 
    n_cells = len(cell_indices)
    n_organs = sim.n_organs
    n_states = 3 + 2 * n_organs

    # prepare times
    cell_times = sim.current_times[cell_indices] / 60.0
    cell_last_times = sim.last_update_times[cell_indices] / 60.0
    dt_min = cell_times - cell_last_times

    valid_mask = dt_min > 0
    if nochange_until_s:
        allowed_mask = sim.current_times[cell_indices] > nochange_until_s
        process_mask = valid_mask & allowed_mask
    else:
        process_mask = valid_mask

    num_valid = np.sum(process_mask)
    if debug and cell_indices is not None:
        print(f"vectorized_compartment_change_restricted: n_cells={n_cells}, n_organs={n_organs}, n_states={n_states}")
        print(f"Num cells allowed to update: {num_valid} / {n_cells}")

    if num_valid == 0:
        if debug: print("No valid cells to update.")
        return sim.cell_states[cell_indices].copy()

    sim_indices = np.array(cell_indices)[process_mask]
    cell_times_valid = cell_times[process_mask]
    cell_last_times_valid = cell_last_times[process_mask]
    dt_min_valid = dt_min[process_mask]

    idxs = np.searchsorted(sim.t_pre, cell_times_valid, side="right")
    idxs = np.clip(idxs, 1, len(sim.t_pre)-1) - 1

    cell_kinj  = sim.k_inj_time[idxs]
    cell_kelim = sim.k_elim_time[idxs]
    current_regions = sim.current_regions[sim_indices]

    organ_fractions=sim.organ_fractions
    # ---- prepare generator matrix of the ctmc ----
    Qbase = np.zeros((n_states, n_states))
    for i in range(n_organs):
        region_idx = 3 + 2 * i
        trap_idx   = region_idx + 1
        Qbase[region_idx, 1] = sim.K2[i]
        Qbase[region_idx, trap_idx] = sim.K3[i]

    Qs = np.tile(Qbase, (num_valid, 1, 1))

    # elimination only from blood
    Qs[:, 1, 2] = cell_kelim

    # effective exchange rate blood<->Organ dependent on organ fraction
    kinj_eff = np.zeros(num_valid)
    for k, reg in enumerate(current_regions):
        organ_idx = sim.region_id_to_organ_idx.get(reg, None)
        # no blood case
        if organ_idx is not None and reg != 0 and organ_fractions[organ_idx] > 0:
            kinj_eff[k] = sim.K1[organ_idx] / organ_fractions[organ_idx]
        else:
            kinj_eff[k] = 0.0

    for k, reg in enumerate(current_regions):
        organ_idx = sim.region_id_to_organ_idx.get(reg, None)
        if organ_idx is not None and reg != 0 and organ_fractions[organ_idx] > 0:
            reg_C1 = 3 + 2*organ_idx
            Qs[k, 1, reg_C1] = kinj_eff[k]

    idx = np.arange(n_states)
    Qs[:, idx, idx] = 0
    Q_row_sums = np.sum(Qs, axis=2)
    Qs[:, idx, idx] = -Q_row_sums

    next_states = np.empty(num_valid, dtype=int)
    cell_state_vec = sim.cell_states[sim_indices]
    blood_idx = sim.region_Q_idx_mapping[sim.blood_region_id]
    elimination_idx = 2
    C1_indices = np.array([3 + 2*i for i in range(n_organs)])
    C2_indices = C1_indices + 1

    for k in range(num_valid):
        cell_idx = sim_indices[k]
        region = current_regions[k]

        key = (cell_idx, region)
        visits = visit_counts.get(key, 0) + 1
        visit_counts[key] = visits

        if visits > vectorized_compartment_change_restricted.max_visits:
            vectorized_compartment_change_restricted.max_visits = visits

        max_visits = vectorized_compartment_change_restricted.max_visits
        ignore_prob = max_ignore_prob * (visits / max_visits) if max_visits > 1 else 0.0

        if np.random.random() < ignore_prob:
            if debug:
                print(f"Cell {cell_idx} in region {region} (visits={visits}, max_visits={max_visits}) -- ignore_prob: {ignore_prob:.3f}, SKIPPED")
            next_states[k] = cell_state_vec[k]
            continue
        if dt_min_valid[k] > 10:
            print(f'WARNING: dt_min is very large for cell {sim_indices[k]}: {dt_min_valid[k]}')
            print(sim.current_times[sim_indices[k]],
                sim.current_vessel_ids[sim_indices[k]],
                sim.current_regions[sim_indices[k]],
                sim.cell_states[sim_indices[k]],
                nochange_until_s)
        if np.max(np.abs(Qs[k])) > 1e5:
            print(sim.current_times[sim_indices[k]],
                sim.current_vessel_ids[sim_indices[k]],
                sim.current_regions[sim_indices[k]],
                sim.cell_states[sim_indices[k]],
                nochange_until_s)
            print(f'WARNING: Qs is very large for cell {sim_indices[k]}: max(abs(Q))={np.max(np.abs(Qs[k]))}')
        P = expm(Qs[k] * dt_min_valid[k])
        cs = cell_state_vec[k]
        probs = P[cs]
        prob_mod = np.copy(probs)
        region_id = current_regions[k]
        c1 = sim.region_Q_idx_mapping.get(region_id, None)
        if cs in C2_indices:
            prob_mod[:] = 0.
            prob_mod[cs] = 1.0
        elif cs == blood_idx:
            allowed = np.zeros(n_states, dtype=bool)
            allowed[blood_idx] = True
            allowed[elimination_idx] = True
            if region_id != sim.blood_region_id and c1 is not None and c1 in C1_indices:
                allowed[c1] = True
            prob_mod[blood_idx] += np.sum(prob_mod[~allowed])
            prob_mod[~allowed] = 0.
        elif cs in C1_indices:
            c1 = cs
            c2 = c1 + 1
            allowed = np.zeros(n_states, dtype=bool)
            allowed[[blood_idx, c1, c2]] = True
            prob_mod[blood_idx] += np.sum(prob_mod[~allowed])
            prob_mod[~allowed] = 0.
        prob_mod /= prob_mod.sum()
        next_states[k] = np.random.choice(n_states, p=prob_mod)

    all_states = sim.cell_states[cell_indices].copy()
    all_states[process_mask] = next_states

    sim.cell_states[sim_indices] = next_states
    sim.last_update_times[sim_indices] = sim.current_times[sim_indices]

    return all_states
