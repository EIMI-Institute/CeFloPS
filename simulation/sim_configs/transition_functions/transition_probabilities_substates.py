from math import exp
import CeFloPS.simulation.settings as settings
import numpy as np
from scipy.linalg import expm


def get_adapted_k1(voivolume, k1):
    return (k1 * voivolume) / 4000  # ml blood


def P_C1C1(k2, k3, t):
    return exp(-(k2 + k3) * t)


def P_C1C2(k2, k3, t):
    return (k3 / (k2 + k3)) * (1 - exp(-(k2 + k3) * t))


def P_C1CA(k2, k3, t):
    return (k2 / (k2 + k3)) * (1 - exp(-(k2 + k3) * t))


def P_CAC1(k1, t):
    return 1 - exp(-(k1) * t)


def P_CACA(k1, t):
    return exp(-(k1) * t)


def p_substate_exp(
    self, cell, simulation, sim_time=-1
) -> ((float, float, float), (object, object, object)):
    """Probabilities for substates

    Args:
        cell (_type_): _description_
        simulation (_type_): _description_

    Returns:
        _type_: compartment list and their chances
    """
    compartment = cell.compartment

    if sim_time < 0:
        t = simulation.time / 60
    else:
        t = sim_time

    if compartment is not None:
        if len(compartment.k_outs) == 2:
            assert compartment.type == "C1"
            C1 = compartment
            C2 = C1.C_outs[1]
            Ca = C1.C_outs[0]
            assert Ca.type == 1
            assert C2.type == "C2"
            # in C1
            k2, k3 = compartment.k_outs

            p_1a = P_C1CA(k2, k3, t)
            p_12 = P_C1C2(k2, k3, t)
            p_11 = P_C1C1(k2, k3, t)

            cs = [Ca, C1, C2]

            assert cs[2].type == "C2", (cs[0].type, cs[1].type, cs[2].type)

            return [p_1a, p_11, p_12], cs
        else:
            # in C2
            assert len(compartment.k_outs) == 1
            assert compartment.k_outs[0] == 0
            C1 = compartment.C_outs[0]
            C2 = compartment
            Ca = C1.C_outs[0]
            cs = [Ca, C1, C2]
            return [0, 0, 1], cs  # C2 is final

    else:
        # Ca
        chance_per_s = P_CACA(self.roi.k1, simulation.time / 60)
        C1 = cell.roi.compartment_model
        C2 = C1.C_outs[1]
        Ca = C1.C_outs[0]
        cs = [Ca, C1, C2]
        return [chance_per_s, 1 - chance_per_s, 0], cs


def get_q_relative_k1(roi, k1):
    return (settings.q_total / roi.q) * roi.k1


def p_substate_exp_k1_prop_q(
    self, cell, simulation, sim_time=-1
) -> ((float, float, float), (object, object, object)):
    """Probabilities for substates using k1 relative to inflow

    Args:
        cell (_type_): _description_
        simulation (_type_): _description_

    Returns:
        _type_: _description_
    """
    compartment = cell.compartment

    if sim_time < 0:
        t = simulation.time / 60
    else:
        t = sim_time

    if compartment is not None:
        if len(compartment.k_outs) == 2:
            assert compartment.type == "C1"
            C1 = compartment
            C2 = C1.C_outs[1]
            Ca = C1.C_outs[0]
            assert Ca.type == 1
            assert C2.type == "C2"
            # in C1
            k2, k3 = compartment.k_outs

            p_1a = P_C1CA(k2, k3, t)
            p_12 = P_C1C2(k2, k3, t)
            p_11 = P_C1C1(k2, k3, t)

            cs = [Ca, C1, C2]

            assert cs[2].type == "C2", (cs[0].type, cs[1].type, cs[2].type)

            return [p_1a, p_11, p_12], cs
        else:
            # in C2
            assert len(compartment.k_outs) == 1
            assert compartment.k_outs[0] == 0
            C1 = compartment.C_outs[0]
            C2 = compartment
            Ca = C1.C_outs[0]
            cs = [Ca, C1, C2]
            return [0, 0, 1], cs  # C2 is final

    else:
        # Ca
        chance_per_s = P_CACA(get_q_relative_k1(self.roi.k1), simulation.time / 60)
        C1 = cell.roi.compartment_model
        C2 = C1.C_outs[1]
        Ca = C1.C_outs[0]
        cs = [Ca, C1, C2]
        return [chance_per_s, 1 - chance_per_s, 0], cs


def get_transition_chances_markov(k1, k2, k3, interval_s):
    Q = np.array([[-k1, k1, 0], [k2, -k2 - k3, k3], [0, 0, 0]])
    t = (1 / 60) * interval_s
    P = expm(Q * t)
    # ca ca c1 c2
    # c1 ca c1 c2
    # c2 ca c1 c2
    return P


def p_substate_continuous_markov(
    cell, simulation, sim_time=-1
) -> ((float, float, float), (object, object, object)):
    """Probabilities for substates assuming a continuous markov chain

    Args:
        cell (_type_): cell to get cellstate
        simulation (_type_): _description_

    Returns:
        _type_: _description_
    """
    compartment = cell.compartment
    #print(f"[P SUBSTATE] MARKOV  cell_roi {cell.roi.name},  compartment {cell.compartment}, compartment_model {cell.roi.compartment_model.__dict__}")
    if type(cell.roi.blood_roi)==list:
        cell.roi.blood_roi=cell.blood
        cell.roi.compartment_model.C_outs[0]=cell.blood
        assert cell.blood==simulation.blood_roi
    assert not any([type(x)==list for x in cell.roi.compartment_model.C_outs])

    if sim_time < 0:
        t = simulation.time / 60
    else:
        t = sim_time
    transition_chances = get_transition_chances_markov(
        cell.roi.k1,
        cell.roi.compartment_model.k_outs[0],
        cell.roi.compartment_model.k_outs[1],
        sim_time,
    )
    if compartment is not None:
        if len(compartment.k_outs) == 2:
            # in c1
            pa, p1, p2 = transition_chances[1]

            assert compartment.type == "C1"
            C1 = compartment
            C2 = C1.C_outs[1]
            Ca = C1.C_outs[0]
            assert Ca.type == 1
            assert C2.type == "C2"
            cs = [Ca, C1, C2]

            assert cs[2].type == "C2", (cs[0].type, cs[1].type, cs[2].type)

            return [pa, p1, p2], cs
        else:
            # in C2
            pa, p1, p2 = transition_chances[2]

            assert len(compartment.k_outs) == 1
            assert compartment.k_outs[0] == 0
            C1 = compartment.C_outs[0]
            C2 = compartment
            Ca = C1.C_outs[0]
            cs = [Ca, C1, C2]
            return [pa, p1, p2], cs  # C2 is final

    else:
        # Ca
        pa, p1, p2 = transition_chances[0]

        C1 = cell.roi.compartment_model
        C2 = C1.C_outs[1]
        Ca = C1.C_outs[0]
        cs = [Ca, C1, C2]
        return [pa, p1, p2], cs


def p_substate_no_change_until(cell, simulation, sim_time=-1, change_time=0):
    """Return no change until changetime, then let changes begin as if simtime is 0

    Args:
        cell (_type_): _description_
        simulation (_type_): _description_
        change_time (_type_): _description_
        sim_time (int, optional): _description_. Defaults to -1.
    """
    if sim_time < change_time:
        return p_substate_no_change(cell, simulation, sim_time=-1)
    # cell.time
    # they give time since last seen, this should be changed to be lastseentime=min(celltime-changetime,lastseen)
    return p_substate_continuous_markov(
        cell, simulation, sim_time=min(cell.time - change_time, sim_time)
    )


def p_substate_no_change(
    cell, simulation, sim_time=-1
) -> ((float, float, float), (object, object, object)):
    """Probabilities for substates assuming a continuous markov chain

    Args:
        cell (_type_): cell to get cellstate
        simulation (_type_): _description_

    Returns:
        _type_: _description_
    """
    compartment = cell.compartment
    pa, p1, p2 = 0, 0, 0
    if compartment is not None:
        if len(compartment.k_outs) == 2:
            # in c1

            assert compartment.type == "C1"
            C1 = compartment
            C2 = C1.C_outs[1]
            Ca = C1.C_outs[0]
            assert Ca.type == 1
            assert C2.type == "C2"
            cs = [Ca, C1, C2]

            assert cs[2].type == "C2", (cs[0].type, cs[1].type, cs[2].type)

            return [pa, p1, p2], cs
        else:
            # in C2

            assert len(compartment.k_outs) == 1
            assert compartment.k_outs[0] == 0
            C1 = compartment.C_outs[0]
            C2 = compartment
            Ca = C1.C_outs[0]
            cs = [Ca, C1, C2]
            return [pa, p1, p2], cs  # C2 is final

    else:
        # Ca

        C1 = cell.roi.compartment_model
        C2 = C1.C_outs[1]
        Ca = C1.C_outs[0]
        cs = [Ca, C1, C2]
        return [pa, p1, p2], cs
