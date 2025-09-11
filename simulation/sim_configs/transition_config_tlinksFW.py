from CeFloPS.simulation.sim_configs.abstract_config import AbstractStateMachine
import CeFloPS.simulation.sim_configs.transition_functions as transition_functions


# configuration that uses FW instead of RW
class DefStateMachine(AbstractStateMachine):
    def t_position_time_voi(self, cell, bound, simulation):
        return transition_functions.t_position_time_voi_f_walk(cell, bound, simulation)
        # set to use flowwalk instead of random walk

    def p_resolve_link_chances(self, links, simulation=None):
        return transition_functions.p_link_chances_q(links, simulation=simulation)

    def t_position_time_vessel(self, cell, simulation, pfun):
        return transition_functions.t_position_time_vessel(cell, simulation, pfun)

    def t_location_VOI_VEIN(self, cell):
        return transition_functions.t_location_VOI_VEIN_biased_rw(cell)

    def p_substate(self, cell, simulation, time_taken) -> (list, list):
        """Transitionchances for substates

        Args:
            cellstate (_type_): _description_
        """
        return transition_functions.p_substate_continuous_markov(
            cell, simulation, time_taken
        )
