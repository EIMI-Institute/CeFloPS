from .abstract_config import AbstractStateMachine
import transition_functions


class ReconStateMachine(AbstractStateMachine):
    def __init__(self):
        ReconStateMachine.t_position_time_vessel = (
            transition_functions.t_position_time_vessel
        )
        ReconStateMachine.t_position_time_voi = transition_functions.t_position_time_voi

        ReconStateMachine.p_substate = transition_functions.p_substate

        ReconStateMachine.t_location_ART_VEIN = transition_functions.t_location_ART_VEIN
        ReconStateMachine.t_location_ART_VOI = transition_functions.t_location_ART_VOI
        ReconStateMachine.t_location_VOI_VEIN = transition_functions.t_location_VOI_VEIN

        ReconStateMachine.resolve = transition_functions.resolve
        ReconStateMachine.p_arteries = transition_functions.p_arteries_q
        ReconStateMachine.p_arteries = transition_functions.p_arteries_r
        ReconStateMachine.p_arteries = transition_functions.p_arteries_attract
