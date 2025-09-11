import matplotlib.pyplot as pltvf
import networkx as nx
import numpy as np
import pandas
import itertools

# import CeFloPS.simulation.common.vessel_functions.get_traversable_links as get_traversable_links
import CeFloPS.simulation.settings as settings
import CeFloPS.simulation.common.functions as cf
import CeFloPS.simulation.common.unit_conv as unit_conv
from sympy import *
from io import BytesIO
import PIL.Image as Image

import matplotlib.pyplot as pltvf
import networkx as nx
import numpy as np
import pandas
import itertools

 


# from CeFloPS.simulation.common.volumesystem_simplification import *
def check_volume_LGS(volumes):
    """Check if a system of volumes is in a form that can be solvewith linear equations.
    Checks that no volume connects to interconnecting volumes

    Args:
        volumes: resistant_volumeiterable to validate
    """
    for volume in volumes:
        assert volume not in volume.next_vol, "volume loops to itself"
        for vol_target in volume.next_vol:
            for vol3 in vol_target.next_vol:
                assert (
                    vol3 not in volume.next_vol
                ), "targets are connected between each other -> one would be 0"
        assert len(volume.next_vol) == len(set(volume.next_vol))


def get_traversable_links(new_vessel):
    """Get traversable links for a vessel, based on if its classified as artery or vein

    Args:
        new_vessel (Vessel): Vesselobject which has an amount of associated links

    Returns:
        list of Link: traversable links of vessel
    """
    if new_vessel.type == "vein":
        try:
            if new_vessel.highest_link() != None:
                return [new_vessel.highest_link()]
            return []
        except:
            print("vein has no links or wrong direction")
            return []
    else:
        return new_vessel.next_links(0)


def get_traversable_links_ALL(new_vessel):
    if new_vessel.type == "vein":
        try:
            return links_at(new_vessel, len(new_vessel.path) - 1)
            """ if new_vessel.highest_link()!=None:
                return [new_vessel.highest_link()]
            return [] """
        except:
            print("vein has no links or wrong direction")
            return []
    else:
        return new_vessel.next_links(0)


def create_graph_from_vessels(vessels, vesselnames=False, all_end=True):
    """Create a networkx graph that shows the connection through links from all vessels given as parameter.
    Nodes are vessels and links are edges.

    Args:
        vessels (list of Vessel): The vessels to show
        vesselnames (bool, optional): If vesselnames should be shown in nodes. Defaults to False.
        all_end (bool, optional): If all highestlinks of veins should be used to construct the graph and not the one that gets used for traversal. Defaults to True.

    Returns:
        networkx.DiGraph: Graph
    """
    G = nx.DiGraph()
    for i, vessel in enumerate(vessels):
        if vesselnames:
            G.add_node(f"{vessel.id[0:17]}{vessel.associated_vesselname[94::]}")
        else:
            G.add_node(vessel.id[0:16])
        if all_end:
            for link in get_traversable_links_ALL(vessel):
                if vesselnames:
                    G.add_edge(
                        f"{vessel.id[0:17]}{vessel.associated_vesselname[94::]}",
                        f"{link.target_vessel.id[0:17]}{link.target_vessel.associated_vesselname[94::]}",
                    )
                else:
                    G.add_edge(f"{vessel.id[0:16]}", f"{link.target_vessel.id[0:16]}")
        else:
            for link in get_traversable_links(vessel):
                if vesselnames:
                    G.add_edge(
                        f"{vessel.id[0:17]}{vessel.associated_vesselname[94::]}",
                        f"{link.target_vessel.id[0:17]}{link.target_vessel.associated_vesselname[94::]}",
                    )
                else:
                    G.add_edge(f"{vessel.id[0:16]}", f"{link.target_vessel.id[0:16]}")
    return G


def previous_links(vessel, plink):
    """Get all links that are connected at a previous index of one vessel

    Args:
        vessel (Vessel): Vessel that holds links
        plink (Link): Upper bound link

    Returns:
        list of Link: List of all previous saved outgoing links
    """
    return [
        link
        for link in vessel.links_to_path
        if link.source_index <= plink.source_index and link != plink
    ]


def create_graph_from_vessels_link_index_nodes(vessels):
    G = nx.DiGraph()
    color_map = []
    lp = (
        lambda vessel, link: f"{vessel.id[0:17]}{vessel.associated_vesselname[94::]},{link.source_index}"
    )
    for i, vessel in enumerate(vessels):
        for link in vessel.links_to_path:
            G.add_node(lp(vessel, link))
            if link.source_index == 0:
                color_map.append("green")  # TODO check if node is doubled
            else:
                color_map.append("blue")

            if vessel.type == "vein":
                # add arrow TODO
                if link.source_index == 0:
                    G.add_edge(lp(vessel, link), lp(vessel, vessel.highest_link()))
                else:
                    G.add_edge(
                        lp(vessel, link),
                        f"{link.target_vessel.id[0:17]}{link.target_vessel.associated_vesselname[94::]}, {link.target_index}",
                    )
            else:
                # add arrow inside of vessel to link source
                for l2 in previous_links(vessel, link):
                    G.add_edge(lp(vessel, l2), lp(vessel, link))
                G.add_edge(  # add linkarrow
                    lp(vessel, link),
                    f"{link.target_vessel.id[0:17]}{link.target_vessel.associated_vesselname[94::]},{link.target_index}",
                )

        # link to each rechable link in vesssel
        # link for the acutual link
        """ for link in get_traversable_links(vessel):  # .links_to_path:#
                G.add_edge(
                    f"{vessel.id[0:17]}{vessel.associated_vesselname}",
                    f"{link.target_vessel.id[0:17]}{link.target_vessel.associated_vesselname}",
                )
            else:
                G.add_edge(f"{vessel.id}", f"{link.target_vessel.id}") """

    return G, color_map


def draw_vesselgraph(vessels):
    """Draw a graph from create_graph_from_vessels

    Args:
        vessels (list of Vessel): list of the vessels to be shown
    """
    # layout= nx.planar_layout(G, scale=1, center=None, dim=2)
    # pos=nx.nx_agraph.graphviz_layout(G, prog="dot")

    G = create_graph_from_vessels(vessels, True, False)
    pos = nx.nx_agraph.graphviz_layout(G)
    fig = plt.figure(1, figsize=(160, 90), dpi=60)
    nx.draw_networkx(G, pos, node_size=100, font_size=10, arrowsize=25)
    pos = nx.nx_agraph.graphviz_layout(g)
    nx.draw_networkx(
        g, pos, with_labels=True
    )  # ,layout)#,node_color=create_graph_from_vessels_link_index_nodes(veins)[1])#, layout=neato )
    list(nx.simple_cycles(G))
    plt.show()


from scipy.integrate import quad

PI = 3.14159265359
VISCOSITY = 0.0035


def velocity_profile(radius, length, p1, p2):
    """Generate velocity profile for a volume given through parameters

    Args:
        radius (float): radius of the volume in m
        length (float): length of the volume in m
        p1 (float): pressure at beginning of volume in Pa
        p2 (float): pressure at end of volume in Pa

    Returns:
        function: function that gives the velocity for an input r: the distance from center of volume
    """
    profile = lambda r: (1 - ((r**2) / (radius**2))) * v_max(radius, length, p1, p2)
    return profile


def v_max(radius, length, p1, p2):
    """Calculate max velocity for a volume specified by arguments

    Args:
        radius (float): radius of the volume in m
        length (float): length of the volume in m
        p1 (float): pressure at beginning of volume in Pa
        p2 (float): pressure at end of volume in Pa

    Returns:
        float: max velocity for given volume
    """
    return (1 / (4 * VISCOSITY)) * radius**2 * ((p1 - p2) / length)


def probability_profile(profile, radius):
    """Generate a probability profile for a given velocity profile and the volumes radius as the integral of its volumeflow function TODO

    Args:
        profile (_type_): _description_

    Returns:
        _type_: _description_
    """
    # integrate.quad(lambda r: q_over_r(r) / total, 0, r)[0]
    total = quad(lambda r: profile(r) * abs(r) * 2 * PI, 0, radius)[0]  # q over r

    pprofile = lambda x: quad(lambda r: (profile(r) * abs(r) * 2 * PI) / total, 0, x)[0]
    return pprofile


def create_profiles(vessel):
    """Generate velocity and probability profile

    Args:
        vessel (resistant_volume): volume that holds radius length and pressure information

    Returns:
        tuple of function: velocity_profile, probability_profile
    """
    return (
        velocity_profile(vessel.radius, vessel.length, vessel.p1, vessel.p2),
        probability_profile(
            velocity_profile(vessel.radius, vessel.length, vessel.p1, vessel.p2)
        ),
    )


def Q_from_speed(A, v):
    """Calculate the Flow Q from a given velocity v and Area A

    Args:
        A (float): Area where flow begins from in m²
        v (float): velocity in m/s

    Returns:
        float: Flowrate Q in m³/s
    """
    return A * v


def speed_from_Q(A, Q):
    """Calculate the Flow Q from a given velocity v and Area A

    Args:
        A (float): Area where flow begins from in m²
        v (float): velocity in m/s

    Returns:
        float: Flowrate Q in m³/s
    """
    return Q / A


def vessel_resistance(radius, l_m):
    """Calculate the resistace of a volume under the assumption of a constant viscosity

    Args:
        radius (float): radius of volume in m
        l_m (float): length of volume in m

    Returns:
        float: volumerestistance W in TODO
    """
    return (8 * VISCOSITY * l_m) / (PI * radius**4)


def Q_from_p(radius, length_m, p1, p2):
    """Calculate Flow Q from a volume given by parameters

    Args:
        radius (float): radius of the volume in m
        length_m (float): length of the volume in m
        p1 (float): pressure at beginning of volume in Pa
        p2 (float): pressure at end of volume in Pa

    Returns:
        float: flowrate Q in m³/s
    """
    return (p1 - p2) / vessel_resistance(radius, length_m)


def dQ(d, r, radius, length, p1, p2):
    """_summary_

    Args:
        d (_type_): _description_
        r (_type_): _description_
        radius (_type_): _description_
        length (_type_): _description_
        p1 (_type_): _description_
        p2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    return 2 * PI * r * d * (1 - (r**2 / radius**2)) * v_max(radius, length, p1, p2)


def makeLGS(volumes):
    """Generate a system of linear equations from volumes that is usable to get their pressure and Flowrates given an Entryflowrate

    Args:
        volumes (list of resistant_volume): list of all connected volumes

    Returns:
        list of sympy.Eq, search_variables: Equations and searched variables
    """
    searched_vars = []
    for vol in volumes:
        searched_vars += vol.variables_searched
    searched_vars = set(searched_vars)
    equations = [vol.eq_q for vol in volumes]
    assert False not in equations
    # equations.append(eq_sumQ)
    for vol in volumes:
        # print("volume: ", vol.name, "eq: ",vol.eq_q)
        equations += vol.q_equations
        assert False not in vol.q_equations, vol.q_equations
        # print("volume: ", vol.__dict__, "eq: ",vol.q_equations)

    # for equation in eq_sumQs:
    #    equations.append(equation)
    for i, q in enumerate(equations):
        assert not q.has(
            oo, -oo, zoo, nan
        ), f"Infinity value in equation{volumes[i].name}, {volumes[i].__dict__}"
        q = simplify(q)

        assert not q.has(oo, -oo, zoo, nan), "Infinity value in equation"
        assert q != False, "simplify is bad"
    equations = set(equations)
    return list(equations), searched_vars


def insert_start_vars(volumes, system_q, system_p1, start_volumes=[]):
    """Used in Equation-System-generation: Assign leading volumes system_p1 as pressure and system_q.

    Args:
        volumes (list of resistant_volume): volumelist that gets processed
        system_q (_type_): Flowrate through system
        system_p1 (_type_): Startpressure from system
        start_volumes (list, optional): Alternative way to specify leading volumes. Default is those volumes that are not linked by other volumes. Defaults to [].

    Returns:
        sympy.Eq: Equation about the start_volumes. Their Sum of Flow should equal the systems flow
    """
    n_volumes = []
    start_equation = []
    for vol in volumes:
        n_volumes += vol.next_vol
    if len(start_volumes) == 0:
        for vol in volumes:
            if vol not in n_volumes:
                start_volumes += [vol]
    # start_volume.update_eq_q()
    for start_volume in start_volumes:
        if len(start_volumes) == 1:
            start_volume.set_val_for_symbol(start_volume.Q_1, system_q)

        start_volume.set_val_for_symbol(start_volume.P_1, system_p1)
    if len(start_volumes) > 1:
        start_equation.append(
            Eq(system_q, sum([vol.get_symval(vol.Q_1) for vol in start_volumes]))
        )
    print("start_equation", start_equation)
    return start_equation


def insert_end_vars(volumes, system_q, system_p2, end_volumes=[]):
    """Used in Equation-System-generation: Sets the last volumes to have the same endpressure

    Args:
        volumes (list of resistant_volume): volumelist that gets processed
        system_q (float): Flowrate through system
        system_p1 (float): Endpressure from system
        end_volumes (list, optional): Alternative way to specify last volumes. Default is those volumes that do not link to other volumes. Defaults to [].
    """
    # set p2 berfore to p1 at current
    if len(end_volumes) == 0:
        for vol in volumes:
            if len(vol.next_vol) == 0:
                vol.set_val_for_symbol(vol.P_2, system_p2)
    else:
        for vol in end_volumes:
            if len(vol.next_vol) == 0:
                vol.set_val_for_symbol(vol.P_2, system_p2)


def write_qual_p21(volumes, equations=False):
    """Create all Equations about volumes that specify that for any volume v1 the following volume v2 should have the same starting pressure as v1s ending pressure

    Args:
        volumes (list of resistant_volume): List of volumes to process

    Returns:
        list of sympy.Eq: List of equations for pressureconnections

    """
    out = []
    if equations == True:
        for vol in volumes:
            # if len(vol.next_vol)==1:
            for next_v in vol.next_vol:
                # if ():#only if thery are in series, not in parallel
                out.append(Eq(next_v.get_symval(next_v.P_1), vol.get_symval(vol.P_2)))
                assert not Eq(
                    next_v.get_symval(next_v.P_1), vol.get_symval(vol.P_2)
                ).has(oo, -oo, zoo, nan), "Infinity value in equation"

                if Eq(next_v.get_symval(next_v.P_1), vol.get_symval(vol.P_2)) != False:
                    # assert False, ((sum([v.get_symval(v.Q_1) for v in parallelvols])), sum([p_volume.get_symval(p_volume.Q_1) for p_volume in vol.prevs]))
                    """print(
                        "IM TRUE PRESS",
                        Eq(next_v.get_symval(next_v.P_1), vol.get_symval(vol.P_2)),
                    )
                    print(next_v.get_symval(next_v.P_1), vol.get_symval(vol.P_2))"""
            # vol.equivalent_symbols[].add()
        return out

    else:
        for vol in volumes:
            # if len(vol.next_vol)==1:
            for next_v in vol.next_vol:
                if next_v.get_symval(next_v.P_1) != None:
                    # if already set: put equation for equality
                    out.append(
                        Eq(next_v.get_symval(next_v.P_1), vol.get_symval(vol.P_2))
                    )

                else:
                    next_v.set_val_for_symbol(next_v.P_1, vol.get_symval(vol.P_2))
                    print("setval", vol.variables)

        return []  # out


def create_Q_equations(volumes, system_q, end_volumes=[]):
    if len(end_volumes) == 0:
        for vol in volumes:
            vol.prevs = set()
            if len(vol.next_vol) == 0:
                end_volumes.append(vol)

    eq_sumQ = Eq(
        system_q, sum([vol.get_symval(vol.Q_1) for vol in end_volumes])
    )  # output system is equal to input
    assert not eq_sumQ.has(oo, -oo, zoo, nan), "Infinity value in equation"

    eq_sumQs = []

    # sum of all previous equal current flow

    # store all prevs
    for vol in volumes:
        for v in vol.next_vol:
            v.prevs.add(vol)

    # the flow of one volume is equal to the incoming flow from volumes. if multiple volumes flow into a set of volumes, then only the sum of all incoming and all outgoing flows will be qual
    """ for volume in volumes:
        change=True
        incoming_volumes = [volume]
        outgoing_volumes = [vol2 for vol2 in volume.next_vol]
        while(change):
            change=False
            for vol in outgoing_volumes:
                for inc in vol.prevs:
                    if inc not in incoming_volumes:
                        incoming_volumes.append(inc)
                        change=True
            for vol in incoming_volumes:
                for inc in vol.next_vol:
                    if inc not in outgoing_volumes:
                        outgoing_volumes.append(inc)
                        change=True
        next_q = Eq(
            sum([v.get_symval(v.Q_1) for v in incoming_volumes]),
            sum([p_volume.get_symval(p_volume.Q_1) for p_volume in outgoing_volumes]),
        )
        if next_q != False:
            # assert False, ((sum([v.get_symval(v.Q_1) for v in parallelvols])), sum([p_volume.get_symval(p_volume.Q_1) for p_volume in vol.prevs]))
            print("IM TRUE q groups", next_q)
            print([v.id for v in incoming_volumes],[p_volume.id for p_volume in outgoing_volumes],"-",sum([v.get_symval(v.Q_1) for v in incoming_volumes]),sum([p_volume.get_symval(p_volume.Q_1) for p_volume in outgoing_volumes]))
            if next_q != True:
                eq_sumQs.append(next_q) """
    # if all from next_vol have no other entries

    for volume in volumes:
        continue
        incoming = []  # q incoming in volume from previous volumes
        other = []  # q that lands in other volume
        for vol in volume.prevs:
            if len(vol.next_vol) == 1:
                incoming.append(vol.get_symval(vol.Q_1))
            else:
                incoming.append(
                    vol.get_symval(vol.Q_1)
                )  # splitq instead of sub better?
                for vol2 in vol.next_vol:
                    if vol2 != volume:
                        other.append(vol2.get_symval(vol2.Q_1))

        next_q = Eq(
            sum([v for v in incoming]) - sum([v for v in other]),
            volume.get_symval(volume.Q_1),
        )
        assert not next_q.has(oo, -oo, zoo, nan), "Infinity value in equation"

        if next_q != False:
            # assert False, ((sum([v.get_symval(v.Q_1) for v in parallelvols])), sum([p_volume.get_symval(p_volume.Q_1) for p_volume in vol.prevs]))
            # print("IM TRUE q groups", next_q)
            # print([v.id for v in incoming_volumes],[p_volume.id for p_volume in outgoing_volumes],"-",sum([v.get_symval(v.Q_1) for v in incoming_volumes]),sum([p_volume.get_symval(p_volume.Q_1) for p_volume in outgoing_volumes]))
            if next_q != True:
                eq_sumQs.append(next_q)
        else:
            ...
            """ print(
                "...............",
                [v for v in incoming],
                [v for v in other],
                volume.get_symval(volume.Q_1),
            ) """

    # print("eqs", eq_sumQs)

    return [eq_sumQ] + list(set(eq_sumQs))


def get_volume_data(vessel, lower_pathindex, higher_index):
    offset = 0
    # calc offset as path has 2 more elements than diameters (1 at beginning)
    if lower_pathindex != 0:
        ...
        # offset += 1 TODO remove as len diameters len pathpoints is given now
    """ if higher_index == len(vessel.path) - 1:
        offset += 1 """
    print(lower_pathindex)
    print(higher_index)
    print(higher_index + 1)
    try:
        print(vessel.diameters[lower_pathindex - offset : higher_index + 1])
    except:
        print(vessel.associated_vesselname)

    assert (
        sum(vessel.diameters[lower_pathindex - offset : higher_index + 1]) > 0
    ), vessel.__dict__

    rad = sum(vessel.diameters[lower_pathindex - offset : higher_index + 1]) / len(
        vessel.diameters[lower_pathindex - offset : higher_index + 1]
    )
    rad = 0.707 * (rad / 2)  # as diameters are saved
    print(lower_pathindex, higher_index)
    length = sum(
        [
            cf.calculate_distance_euclidian(vessel.path[i], vessel.path[i + 1])
            for i in range(lower_pathindex, higher_index)
        ]
    )
    if lower_pathindex != higher_index:
        assert length > 0
    assert len(
        [
            cf.calculate_distance_euclidian(vessel.path[i], vessel.path[i + 1])
            for i in range(lower_pathindex, higher_index)
        ]
    ) == -1 + len(vessel.diameters[lower_pathindex - offset : higher_index + 1])

    return rad, length  # in mm and mm


def Q_from_speed(A, v):
    return A * v


def vessel_resistance(radius, l_m):
    return (8 * VISCOSITY * l_m) / (PI * radius**4)


def Q_from_p(radius, length_m, p1, p2):
    return (p1 - p2) / vessel_resistance(radius, length_m)


class resistant_volume:
    VISCOSITY = 0.0035

    PI = 3.14159265359
    # newid = itertools.count().next

    def __init__(
        self,
        radius,
        length,
        identifier=None,
        Q=None,
        P1=None,
        P2=None,
        Q_in_part=None,
        lower_volumes=[],
        res=None,
        lower_type=None,
    ):
        if len(lower_volumes) > 0:
            assert (
                lower_type == "serial"
                or lower_type == "parallel"
                or lower_type == "delta"
            ), "Only parallel and serial implemented"

        self.lower_type = lower_type
        self.to_append = []  # vessel and index
        self.pathindices = []  # indices in vessel
        self.vessel = None
        self.prevs = set()
        self.length = length  # m
        self.A = (radius**2) * PI
        self.radius = radius  # m^2
        if res == None:
            self.resistance = (8 * VISCOSITY * self.length) / (
                PI * self.radius**4
            )  # Pa = N/m^2
        else:
            self.resistance = res
        if identifier != None:
            assert (
                type(identifier) != int
            ), "Do not give integer identifier, use given ones from class"
            self.id = identifier
        else:
            """resistant_volume.id_count+=1
            self.id = resistant_volume.id_count"""
            self.id = id(self)  # self.incr()
            # self.id = resistant_volume.newid()
        self.Q_1, self.W, self.P_1, self.P_2 = symbols(
            f"Q_{self.id}, W_{self.id}, P1_{self.id}, P2_{self.id}", positive=True
        )
        self.variables = {
            self.Q_1: Q,
            self.W: self.resistance,
            self.P_1: P1,
            self.P_2: P2,
        }
        self.prev = []
        self.next_vol = []
        """ if self.prev != None and self.prev.P2 != None:
            self.variables[self.P_1]== prev.variables[prev.P_2] """
        self.update_eq_q()
        self.q_equations = []
        self.q_splits = 0
        self.q_split_symbols = []
        self.lower_volumes = lower_volumes

    def get_q_for_equation(self, other_vol=None):
        if other_vol == None:
            self.q_split_symbols.append(
                sympy.Symbol(f"frac_{self.id}_{self.q_splits}")
            )  # addlater qsplits
            self.q_splits += 1
        else:
            self.q_split_symbols.append(
                sympy.Symbol(f"frac_{self.id}_{other_vol.id}")
            )  # addlater qsplits

        return self.q_split_symbols[-1]

    def solve_lower_volumes(self):
        if None in self.variables.values():
            print("Not yet solved volume cant solve lower ones")
            print(self.variables)
            if all(
                [
                    variable != None
                    for key, variable in self.variables.items()
                    if key != self.Q_1
                ]
            ):
                print("calc Q in delta!")
                self.set_val_for_symbol(
                    self.Q_1,
                    calcQ(
                        p1=self.get_symval(self.P_1),
                        p2=self.get_symval(self.P_2),
                        r=self.resistance,
                    ),
                )
            else:
                return
        p1 = self.get_symval(self.P_1)
        q = self.get_symval(self.Q_1)
        p2 = self.get_symval(self.P_2)
        if len(self.lower_volumes) > 0:
            # parallel
            if self.lower_type == "parallel":
                vols = self.lower_volumes

                for vol in vols:
                    if type(vol) == resistant_volume:
                        vol.set_val_for_symbol(vol.P_1, p1)
                        vol.set_val_for_symbol(vol.P_2, p2)
                        vol.set_val_for_symbol(
                            vol.Q_1, calcQ(p1=p1, p2=p2, r=vol.resistance)
                        )
                    vol.p1 = p1
                    vol.p2 = p2
                    vol.q = calcQ(p1=p1, p2=p2, r=vol.resistance)

            # serial
            elif self.lower_type == "serial":
                vols = self.lower_volumes
                left_side = vols[0]
                right_side = vols[1]

                if type(left_side) == resistant_volume:
                    left_side.set_val_for_symbol(left_side.P_1, p1)
                    left_side.set_val_for_symbol(
                        left_side.P_2, calc_p2(q=q, p1=p1, r=left_side.resistance)
                    )
                    left_side.set_val_for_symbol(left_side.Q_1, q)
                if type(right_side) == resistant_volume:
                    right_side.set_val_for_symbol(
                        right_side.P_1, calc_p1(q=q, p2=p2, r=right_side.resistance)
                    )
                    right_side.set_val_for_symbol(right_side.P_2, p2)

                    right_side.set_val_for_symbol(right_side.Q_1, q)
                # also set for form vols:
                left_side.q = q
                left_side.p1 = p1
                left_side.p2 = calc_p2(q=q, p1=p1, r=left_side.resistance)
                right_side.q = q
                right_side.p1 = calc_p1(q=q, p2=p2, r=right_side.resistance)
                right_side.p2 = p2

            elif self.lower_type == "delta":
                if self.delta_resolved == False:
                    # check if other parts are already solved, if not then let other one solve q
                    solvable = True
                    for vol in self.same_config:
                        if hasattr(vol, "q") and vol.q != None:
                            print(vol.q, vol.id)
                            continue
                        solvable = False
                    if not solvable:
                        return

                    # calculate pressures per configuration and pass values to delta config thats saved
                    a, b, c = self.same_config
                    ab, ac, bc = self.lower_volumes
                    p_1_a = a.get_symval(a.P_1)  # should be terminal p1 in delta
                    p_2_c = c.get_symval(c.P_2)  # should be terminal p2 in delta
                    p_2_b = b.get_symval(b.P_2)  # should be terminal p2 in delta
                    lower_p = p_2_c
                    higher_p = p_2_b
                    if p_2_b < lower_p:
                        higher_p = p_2_c
                        lower_p = p_2_b

                    # p1a ab p2c
                    ac.set_val_for_symbol(ac.P_1, p_1_a)
                    ac.set_val_for_symbol(ac.P_2, p_2_c)
                    ac.p1 = p_1_a
                    ac.p2 = p_2_c
                    # p1a ac p2b
                    ab.set_val_for_symbol(ac.P_1, p_1_a)
                    ab.set_val_for_symbol(ac.P_2, p_2_b)
                    ab.p1 = p_1_a
                    ab.p2 = p_2_b
                    # higher_p bc lower_p
                    bc.set_val_for_symbol(bc.P_1, higher_p)
                    bc.set_val_for_symbol(bc.P_2, lower_p)
                    bc.p1 = higher_p
                    bc.p2 = lower_p
                    # calculate Q with the given pressures
                    ab.q = calcQ(p1=ab.p1, p2=ab.p2, r=ab.resistance)
                    ac.q = calcQ(p1=ac.p1, p2=ac.p2, r=ac.resistance)
                    bc.q = calcQ(p1=bc.p1, p2=bc.p2, r=bc.resistance)

                    # set all other volumes in config as solved
                    for v in self.same_config:
                        v.delta_resolved = True
            else:
                assert False, "Not other tranform implemented"
        for vol in self.lower_volumes:
            if type(vol) == resistant_volume:
                vol.solve_lower_volumes()

    def update_eq_q(self):
        self.variables_searched = []
        # Equation for Q:
        self.eq_q = Eq(self.Q_1, (self.P_1 - self.P_2) / self.W)
        for key, value in self.variables.items():
            if value != None:
                self.eq_q = self.eq_q.subs(key, value)
                if type(value) == Symbol:
                    self.variables_searched.append(value)
                elif type(value) == float:
                    self.variables[key] = Float(value)
            else:
                self.variables_searched.append(key)

    def get_symval(self, key):
        out = self.variables[key]
        if out == None:
            out = key
        return out

    def make_q_equations(self, volumes):
        prev = set()
        if len(prev) > 0:
            for vol in volumes:
                if self in vol.next_vol:
                    prev.add(vol)
            eq = Eq(
                self.Q_1, sum([vol.get_q_for_equation(self) * vol.Q_1 for vol in prev])
            )

            self.q_equations.append(eq)

    def make_q_split_equations(self):
        if len(self.q_split_symbols) > 0:
            eq = Eq(sum([split_sym for split_sym in self.q_split_symbols]), 1)
            self.q_equations.append(eq)

    def set_val_for_symbol(self, sym, val, verbose=False):
        if sym in self.variables.keys():
            self.variables[sym] = val
            #

            self.update_eq_q()
        else:
            if verbose:
                print(sym, "not in vars")

    def print_self(self):
        return self.eq_q


def create_heart_volumes(vessels):
    for vessel in vessels:
        if "jump_vessel" in vessel.associated_vesselname:
            new_vol = resistant_volume(0.15, 0.15)
            new_vol.path_indices = (0, len(vessel.path) - 1)
            new_vol.vessel = vessel
            new_vol.next_vol = []
            new_vol.set_val_for_symbol(new_vol.W, 0.00000000000000001)
            # new_vol.set_val_for_symbol(new_vol.Q_1,0.00006)
            for link in get_traversable_links(vessel):
                print(vessel.associated_vesselname)
            vessel.volumes = [new_vol]


# xcreate links from 0 to linksources
def create_volumes(vessels):
    volumes = []
    # vols in vessel are associated with  pathpoints
    for vessel in vessels:
        if not "jump_vessel" in vessel.associated_vesselname:
            vessel.volumes = []
            # create vvolumes from 0 to first traversable link
            lower_pathindex = 0
            position = 0
            traversable_links = get_traversable_links(vessel)
            traversable_links = [
                link
                for link in traversable_links
                # if not "jump_vessel" in link.target_vessel.associated_vesselname
            ]
            traversable_links = sorted(
                list(set(traversable_links)), key=lambda x: x.source_index
            )
            while position < len(traversable_links):
                higher_index = vessel.links_to_path[position].source_index
                rad, length = get_volume_data(
                    vessel, lower_pathindex, higher_index
                )  # higher index included
                new_vol = resistant_volume(unit_conv.mm_m(rad), unit_conv.mm_m(length))
                new_vol.path_indices = (lower_pathindex, higher_index)
                new_vol.to_append.append(
                    (
                        traversable_links[position].target_vessel,
                        traversable_links[position].target_index,
                    )
                )
                while position < len(traversable_links) - 1:
                    if traversable_links[position + 1].source_index == higher_index:
                        position += 1
                        new_vol.to_append.append(
                            (
                                traversable_links[position].target_vessel,
                                traversable_links[position].target_index,
                            )
                        )
                    else:
                        break
                new_vol.vessel = vessel
                lower_pathindex = higher_index
                volumes.append(new_vol)
                position += 1
            if (
                lower_pathindex < len(vessel.path) - 1
            ):  # create volume from last link to end TODO are there any really short ends that are just faulty connections?
                higher_index = len(vessel.path) - 1
                rad, length = get_volume_data(vessel, lower_pathindex, higher_index)
                new_vol = resistant_volume(unit_conv.mm_m(rad), unit_conv.mm_m(length))
                new_vol.path_indices = (lower_pathindex, higher_index)
                new_vol.vessel = vessel
                lower_pathindex = higher_index
                volumes.append(new_vol)
    # connect the volumes and assign them their linked order
    for v in volumes:
        assert v.vessel != None
        v.next_vol = []
        v.vessel.volumes.append(v)
        v.vessel.volumes = list(set(v.vessel.volumes))


def split_volume(volume, target):
    assert target < len(volume.vessel.path) - 1, volume.pathindices
    assert target >= volume.path_indices[0] and target < volume.path_indices[1]
    higher_vol = None
    lower_vol = None
    vessel = volume.vessel

    lower_pathindex = volume.path_indices[0]
    higher_index = volume.path_indices[1]

    rad, length = get_volume_data(vessel, lower_pathindex, target)
    lower_vol = resistant_volume(unit_conv.mm_m(rad), unit_conv.mm_m(length))
    lower_vol.path_indices = (lower_pathindex, target)

    rad, length = get_volume_data(vessel, target, higher_index)
    higher_vol = resistant_volume(unit_conv.mm_m(rad), unit_conv.mm_m(length))
    higher_vol.path_indices = (target, higher_index)

    higher_vol.to_append = volume.to_append
    lower_vol.vessel = vessel
    higher_vol.vessel = vessel

    return lower_vol, higher_vol


def split_volumes_for_links(vessels):
    # create split volumes for each linking volume from antoher vessel if necessary
    for vessel in vessels:
        # if "jump" not in vessel.associated_vesselname:
        for link in get_traversable_links(vessel):
            # if "jump" not in link.target_vessel.associated_vesselname:
            to_rem = None
            target = link.target_index
            tvessel = link.target_vessel
            lower_pathindex = -1
            higher_index = -1
            higher_vol = None
            lower_vol = None
            found = False
            # create a volume where link hits the vessel
            for t_vol in tvessel.volumes:
                if (
                    t_vol.path_indices[0] <= target
                    and t_vol.path_indices[1] > target
                    and not any(tv.path_indices[1] == target for tv in tvessel.volumes)
                ):
                    found = True
                    assert target < len(tvessel.path)

                    lower_vol, higher_vol = split_volume(t_vol, target)
                    to_rem = t_vol

                    assert higher_vol != None, (
                        t_vol.path_indices,
                        [vol2.path_indices for vol2 in tvessel.volumes],
                        target,
                        len(tvessel.path),
                        lower_vol,
                        higher_vol,
                    )
                    assert lower_vol != None, "lowa!"
                    break
            if to_rem != None:
                tvessel.volumes.remove(t_vol)
                tvessel.volumes.append(lower_vol)
                tvessel.volumes.append(higher_vol)
            if target == 0:
                assert any([vo.path_indices[1] == 0 for vo in tvessel.volumes])
                assert found == True or any(
                    tv.path_indices[1] == target for tv in tvessel.volumes
                )
    for vessel in vessels:
        # if "jump" not in vessel.associated_vesselname:
        for link in get_traversable_links(vessel):
            # if "jump" not in link.target_vessel.associated_vesselname:
            assert exists_vollume_for_link(link), (
                "creation faulty"
                + link.target_vessel.associated_vesselname
                + "-  "
                + str(link.source_index)
                + "  - "
                + " ---> "
                + str(link.target_index)
                + " "
                + str([vol.path_indices for vol in link.target_vessel.volumes])
                + " self "
                + str([vol.path_indices for vol in link.source_vessel.volumes])
            )


def split_vessels_max_len(vessels):
    MAX_PATHINDEX_DISTANCE_VOLUMES = settings.MAX_PATHINDEX_DISTANCE_VOLUMES
    for vessel in vessels:
        # if "jump_vessel" not in vessel.associated_vesselname:
        removed = True
        while removed:
            add_vols = []
            to_rem = []
            removed = False
            for vol in vessel.volumes:
                if (
                    vol.path_indices[1] - vol.path_indices[0]
                    > MAX_PATHINDEX_DISTANCE_VOLUMES
                ):
                    lower_pathindex = vol.path_indices[0]
                    higher_index = vol.path_indices[1]
                    higher_index = lower_pathindex + (
                        (higher_index - lower_pathindex) // 2
                    )
                    # print(lower_pathindex,higher_index//2, len(vol.vessel.path))
                    rad, length = get_volume_data(vessel, lower_pathindex, higher_index)
                    lower_vol = resistant_volume(
                        unit_conv.mm_m(rad),
                        unit_conv.mm_m(length),
                    )
                    lower_vol.path_indices = (lower_pathindex, higher_index)
                    rad, length = get_volume_data(
                        vessel, higher_index, vol.path_indices[1]
                    )
                    higher_vol = resistant_volume(
                        unit_conv.mm_m(rad),
                        unit_conv.mm_m(length),
                    )
                    higher_vol.path_indices = (higher_index, vol.path_indices[1])
                    higher_vol.to_append = vol.to_append
                    to_rem.append(vol)
                    add_vols.append(lower_vol)
                    add_vols.append(higher_vol)
                    removed = True
            for v in to_rem:
                vessel.volumes.remove(v)
            for v in add_vols:
                v.vessel = vessel
                vessel.volumes.append(v)


def get_volumes(vessels):
    # sort by lower index of volumes
    for vessel in vessels:
        # if "jump" not in vessel.associated_vesselname:
        vessel.volumes = sorted(list(set(vessel.volumes)), key=lambda x: x.path_indices)
    volumes = []
    for vessel in vessels:
        #  if "jump" not in vessel.associated_vesselname:
        volumes += vessel.volumes
    for volume in volumes:
        volume.to_append = []
    return volumes


def intraconnect_vesselvols(vessels):
    # link volumes in vessel
    for vessel in vessels:
        # if "jump_vessel" not in vessel.associated_vesselname:
        for volume in vessel.volumes:
            # link highest to lowest
            volume.next_vol = []
            for vol2 in vessel.volumes:
                if vol2.path_indices[0] == volume.path_indices[1] and vol2 != volume:
                    volume.next_vol.append(vol2)


def interconnect_vesselvols(vessels):
    for vessel in vessels:
        # if "jump" not in vessel.associated_vesselname:
        for volume in vessel.volumes:
            volume.to_append = []

        for link in get_traversable_links(vessel):
            found_source = False
            found_target = False
            for volume in vessel.volumes:
                # if "jump" not in volume.vessel.associated_vesselname:
                if link.source_index == volume.path_indices[1]:
                    found_source = True
                    volume.to_append.append((link.target_vessel, link.target_index))
                # else:
                #    found_source = True

            for vol2 in link.target_vessel.volumes:
                if link.target_index == vol2.path_indices[1]:  # or (
                    #    link.target_index == 0 and vol2.path_indices[0] == 0
                    # ):

                    found_target = True
                    # vol2.to_append.append((link.target_vessel, link.target_index))

            assert found_source == True
            assert (
                found_target == True
                or "jump" in link.target_vessel.associated_vesselname
            ), (
                link.source_vessel.associated_vesselname,
                link.target_index,
                [vol.path_indices for vol in link.target_vessel.volumes],
                len(link.target_vessel.path),
                str(link),
            )


def vessel_has_vol_with_higherindex(vessel, index):
    print(vessel.associated_vesselname)
    return any([vol.path_indices[1] == index for vol in vessel.volumes])


def get_vol_for_vessel_ind(vessel, ind, lower=False):
    for volume in vessel.volumes:
        i = 1
        if lower or ind == 0:
            i = 0
        if volume.path_indices[i] == ind:
            return volume
    print(
        vessel.associated_vesselname, [vol.path_indices for vol in vessel.volumes], ind
    )


def create_graph_from_volumes_append(volumes, vesselnames=False):
    G = nx.DiGraph()
    if vesselnames:
        for i, vol in enumerate(volumes):
            G.add_node(f"{vol.id},{vol.vessel.associated_vesselname[-20::]}")

            for app in vol.to_append:
                vol2 = get_vol_for_vessel_ind(app[0], app[1], True)
                if vol2 != None:
                    G.add_edge(
                        f"{vol.id},{vol.vessel.associated_vesselname[-20::]}",
                        f"{vol2.id},{vol2.vessel.associated_vesselname[-20::]}",
                    )
                """ if vol2.path_indices[1]==app[1]:
                for
                G.add_edge(
                    f"{vol.id},{vol.vessel.associated_vesselname[-20::]}",
                    f"{vol2.id},{vol2.vessel.associated_vesselname[-20::]}",
                ) """
    else:
        for i, vol in enumerate(volumes):
            G.add_node(f"{vol.id}")

            for vol2 in vol.next_vol:
                G.add_edge(f"{vol.id}", f"{vol2.id}")
    return G


def apply_connections(volumes):
    deg = False
    """volumes=[]
    for vessel in vessels:
        volumes+=vessel.volumes"""
    fully_connected = lambda x: len(x.to_append) == 0
    while not all([fully_connected(volume) for volume in volumes]):
        print(
            [
                [v.associated_vesselname[-30::] for v, t in volume.to_append]
                for volume in volumes
                if fully_connected(volume) == False
            ]
        )

        print(
            len(
                [
                    fully_connected(volume)
                    for volume in volumes
                    if fully_connected(volume) == True
                ]
            ),
            len(
                [
                    fully_connected(volume)
                    for volume in volumes
                    if fully_connected(volume) == False
                ]
            ),
            "/",
            len(volumes),
        )
        if (
            len(
                [
                    fully_connected(volume)
                    for volume in volumes
                    if fully_connected(volume) == False
                ]
            )
            == 10
        ):
            deg = True
        """ for vessel in vessels:
            if "jump" not in vessel.associated_vesselname: """
        for volume in volumes:
            if not fully_connected(volume):
                targets = []
                rem = []
                for target in volume.to_append:
                    if deg:
                        ...
                    t_vessel, t_index = target
                    """ if "jump" in t_vessel.associated_vesselname:
                        rem.append(target)
                        break """
                    # if "jump" not in t_vessel.associated_vesselname:

                    for target_volume in t_vessel.volumes:
                        if target_volume.path_indices[1] == t_index and fully_connected(
                            target_volume
                        ):
                            volume.next_vol += target_volume.next_vol
                            rem.append(target)

                for r in rem:
                    volume.to_append.remove(r)


def generate_volumes(
    vessels,
):
    for vessel in vessels:
        if hasattr(vessel, "volumes"):
            for vol in vessel.volumes:
                assert vol != None
    create_heart_volumes(vessels)
    # print(sum([len(vessel.volumes) for vessel in vessels]))
    create_volumes(vessels)
    for vessel in vessels:
        if hasattr(vessel, "volumes"):
            for vol in vessel.volumes:
                assert vol != None
    split_volumes_for_links(vessels)
    split_vessels_max_len(vessels)
    volumes = get_volumes(vessels)
    for vessel in vessels:
        for link in get_traversable_links(vessel):
            assert exists_vollume_for_link(link, volumes), (
                "gathering faulty",
                link.target_vessel.associated_vesselname
                + "-  "
                + str(link.source_index)
                + "  - "
                + " ---> "
                + str(link.target_index)
                + " "
                + str([vol.path_indices for vol in link.target_vessel.volumes])
                + " self "
                + str([vol.path_indices for vol in link.source_vessel.volumes]),
                link.source_vessel.associated_vesselname,
            )
    intraconnect_vesselvols(vessels)
    interconnect_vesselvols(vessels)
    apply_connections(volumes)
    for vessel in vessels:
        volumes += vessel.volumes
    return list(set(volumes))


def get_volume_by_index(vessel, index):
    for volume in vessel.volumes:
        if volume.path_indices[0] <= index and volume.path_indices[1] >= index:
            return volume


def create_graph_from_volumes(volumes, vesselnames=False):
    G = nx.DiGraph()
    if vesselnames:
        for i, vol in enumerate(volumes):
            G.add_node(
                f"{vol.id},{vol.vessel.associated_vesselname[-20::]}",
                node_color="tab:red",
            )

            for vol2 in vol.next_vol:
                G.add_edge(
                    f"{vol.id},{vol.vessel.associated_vesselname[-20::]}",
                    f"{vol2.id},{vol2.vessel.associated_vesselname[-20::]}",
                )
    else:
        for i, vol in enumerate(volumes):
            G.add_node(f"{vol.id}")

            for vol2 in vol.next_vol:
                G.add_edge(f"{vol.id}", f"{vol2.id}")
    return G


# remove unreferecned
def no_refs(volume, volumes):
    for v in volumes:
        if volume in v.next_vol:
            return False
    return True


def exists_vollume_connection_for_link(link, volumes=[]):
    # check if volume for source exists
    if len(volumes) == 0:
        volumes = link.source_vessel.volumes
    for vol in volumes:
        if (
            vol.vessel == link.source_vessel
            and vol.path_indices[1] == link.source_index
        ):
            # print("source found")
            # check if corresponding target exists
            if any(
                [v in vol.next_vol for v in link.target_vessel.volumes]
            ):  # for volume in target: volume is also in next_vol of sourcevolume
                return True
            else:
                return False

    return False


def exists_vollume_for_link(link, volumes=[]):
    # check if volume for source exists
    if len(volumes) == 0:
        volumes = link.source_vessel.volumes
    for vol in volumes:
        if (
            vol.vessel == link.source_vessel
            and vol.path_indices[1] == link.source_index
        ):
            # print("source found")
            # check if corresponding target exists
            if any(
                [
                    v.path_indices[1] == link.target_index
                    for v in link.target_vessel.volumes
                ]
            ):  # for volume in target: volume is also in next_vol of sourcevolume
                return True
            else:
                return False

    return False


# def get_volume_for_link(link, volumes=[]):


def get_vollume_for_link(link, volumes=[]):
    if len(volumes) == 0:
        volumes = link.source_vessel.volumes
    for vol in volumes:
        if (
            vol.vessel == link.source_vessel
            and vol.path_indices[1] == link.source_index
        ):
            for v in link.target_vessel.volumes:
                if v.path_indices[1] == link.target_index:
                    return v, vol
            return None, vol
        else:
            return None, None  #


import matplotlib.pyplot as plt
import networkx as nx
import math


def orderOfMagnitude(number):
    if type(number) == float or type(number) == int or type(number) == np.float64:
        try:
            return math.floor(math.log(number, 10))
        except:
            return number
    return number


def set_prevs(volumes):
    for vol in volumes:
        vol.prevs = set()
    for vol in volumes:
        for v2 in vol.next_vol:
            v2.prevs.add(vol)


def colored_volumegraph(
    volumes,
    restype=False,
    spacing=1,
    print_attr=None,
    print_attr_only=False,
    print_attrs=[],
    attr_funs=[],
    path="",
    font_size=8,
    node_size_multiplier=1,
    highlight_vols=[],
    ret_image=False,
    verbose=False,
    arrowsize=25,
):
    volumes = list(volumes)
    if not all([hasattr(vol, "prevs") for vol in volumes]):
        print("SETPREVS")
        set_prevs(volumes)

    def ident(vol):
        lies_outside_of_volumes = False
        try:
            ident = [i for i in range(len(volumes)) if volumes[i] == vol][
                0
            ]  # str(vol.id)#str(vol.id)
            ident = vol.id
        except:
            print("Volume in next but not in volumes", vol.vessel.associated_vesselname)
            ident = vol.id
            lies_outside_of_volumes = True
        return ident, lies_outside_of_volumes

    if print_attr_only == True:
        assert all(
            [hasattr(volume, print_attr) for volume in volumes]
        ), "All volumes need to have printattr if print_attr_only is set"

    def name(volume):
        """name return the name of the volume in the graph

        Args:
            volume (_type_): _description_

        Returns:
            _type_: _description_
        """
        if len(print_attrs) == 0:
            prefix = ""
            if hasattr(volume, "name"):
                prefix = f"{volume.name}\n"
            appendix = ""
            if print_attr != None:
                if hasattr(volume, print_attr):
                    if print_attr_only:
                        return f"\n {getattr(volume, print_attr)}  \n {orderOfMagnitude(getattr(volume, print_attr))}"
                    appendix = f"\n {getattr(volume, print_attr)}  \n {orderOfMagnitude(getattr(volume, print_attr))}"

            if hasattr(volume, "path_indices") and not hasattr(volume, "lower_volumes"):
                if (
                    hasattr(volume, "vessel")
                    and volume.vessel != None
                    and hasattr(volume.vessel, "associated_vesselname")
                ):
                    main = f"{volume.path_indices},{volume.vessel.associated_vesselname[-20::]}"
                else:
                    main = str(int(id(volume)))
            elif hasattr(volume, "lower_volumes"):
                main = ""  # f"lowerlen {len(volume.lower_volumes)}"
            else:
                main = str(int(id(volume)))
        else:
            name = ""
            for i, attr in enumerate(print_attrs):
                if attr == "vessel.associated_vesselname":
                    if len(attr_funs) == len(print_attrs):
                        name = (
                            name
                            + f"{attr_funs[i](volume.vessel.associated_vesselname[-20::])} \n"
                        )
                    else:
                        name = name + f"{volume.vessel.associated_vesselname[-20::]} \n"
                elif hasattr(volume, attr):
                    if len(attr_funs) == len(print_attrs):
                        name = name + f"{attr_funs[i](getattr(volume, attr))} \n"
                    else:
                        name = name + f"{getattr(volume, attr)}"[0:8] + "\n"
            return name
        return prefix + main + appendix[0:8]

    labeldict = {}
    for vol in volumes:
        labeldict[ident(vol)[0]] = name(vol)

    fig = plt.figure(1, figsize=(160, 90), dpi=60)
    G = nx.DiGraph()
    nodelist = []
    entrylist = []
    outlist = []
    highlightlist = []
    outlier = []
    for volume in volumes:
        if volume not in highlight_vols:
            if ident(volume)[1] == False:
                if no_refs(volume, volumes):
                    entrylist.append(ident(volume)[0])
                elif len(volume.next_vol) == 0:
                    outlist.append(ident(volume)[0])
                else:
                    nodelist.append(ident(volume)[0])
            else:
                outlier.append(ident(volume)[0])
        else:
            highlightlist.append(ident(volume)[0])
    # nodes
    G.add_nodes_from(nodelist, color="blue")
    G.add_nodes_from(entrylist, color="green")
    G.add_nodes_from(outlist, color="red")
    G.add_nodes_from(highlightlist, color="yellow")
    G.add_nodes_from(outlier, color="grey")
    print("nodes pre egdes:", len([node for node in G]))

    if verbose:
        print("Entry:")
    for elem in entrylist:
        if verbose:
            print(elem)
    for vol in volumes:
        for vol2 in vol.next_vol:
            if verbose:
                print(
                    vol.resistance,
                    len(vol.next_vol),
                    "->",
                    vol2.resistance,
                    len(vol2.next_vol),
                )
            G.add_edge(
                ident(vol)[0],
                ident(vol2)[0],
            )
    if path != "":
        fig = plt.figure(1, figsize=(160, 90), dpi=60)
    if restype:
        node_size = 200
        pos = circuit_layout(G, node_size=200)
        fig, ax = plt.subplots()
        color = nx.get_node_attributes(G, "color")
        color_map = [color[node] for node in G]
        # Draw the network with rectangular nodes
        # nx.draw(G, pos, ax=ax, with_labels=True, node_size=node_size, node_color='lightblue', font_size=8,arrowsize=20, node_shape='s')
        nx.draw_networkx(
            G,
            pos,
            node_size=[
                (
                    300 * node_size_multiplier
                    if node not in entrylist
                    else 1200 * node_size_multiplier
                )
                for node in G
            ],
            font_size=font_size,
            node_color=color_map,
            arrowsize=25,
            labels=labeldict,
            with_labels=True,
            node_shape="s",
        )
        # Find starting nodes (nodes with no predecessors)
        starting_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]

        # Add arrows for starting nodes
        for n in starting_nodes:
            # Create an invisible node above the starting node
            invisible_node_pos = (pos[n][0], pos[n][1] + 1.1)

            # Draw an arrow from the invisible node to the starting node
            ax.annotate(
                "",
                xy=(pos[n][0], pos[n][1] + 0.1),
                xycoords="data",
                xytext=invisible_node_pos,
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="->",
                    shrinkA=0,
                    shrinkB=0,
                    patchA=None,
                ),
            )
    else:
        if True:
            print("nodes:", len([node for node in G]))
            print(
                "nodelistlengths",
                len(nodelist),
                len(entrylist),
                len(outlist),
                len(outlier),
                len(highlightlist),
            )

        if len(G) == 1:
            pos = {[node for node in G][0]: (0, 0)}
        else:
            try:
                pos = nx.nx_agraph.graphviz_layout(
                    G, args=f"-Gsplines=true -Gsep={spacing}.0"
                )
            except Exception as e:
                print(e)

                def get_vol_x_y(node):
                    def get_volend_coord(volume):
                        return volume.resistance, volume.resistance

                    for volume in volumes:
                        if ident(volume) == node:
                            if hasattr(volume.vessel, "path"):
                                print("haspath")
                                return volume.vessel.path[volume.path_indices[1]][0:2]
                            else:
                                return (
                                    sum(
                                        [
                                            np.asarray(get_volend_coord(v))
                                            for v in volume.next_vol
                                        ]
                                    )
                                    - 10 / len(volume.next_vol)
                                    + sum(
                                        [
                                            np.asarray(get_volend_coord(v))
                                            for v in volume.prevs
                                        ]
                                    )
                                    - 10 / len(volume.prevs)
                                )[0:2]

                pos = nx.random_layout(
                    G
                )  # dict((node, get_vol_x_y(node)) for node in G)
                print("pos", pos)
        # print(nx.get_node_attributes(G, "color"))
        color = nx.get_node_attributes(G, "color")
        try:
            color_map = [color[node] for node in G]
            nx.draw_networkx(
                G,
                pos,
                node_size=[
                    (
                        300 * node_size_multiplier
                        if node not in entrylist
                        else 1200 * node_size_multiplier
                    )
                    for node in G
                ],
                font_size=font_size,
                node_color=color_map,
                arrowsize=arrowsize,
                labels=labeldict,
                with_labels=True,
            )
        except:
            print("len nodes", len([node for node in G]), len(volumes))
            nx.draw_networkx(
                G,
                pos,
                node_size=[
                    (
                        300 * node_size_multiplier
                        if node not in entrylist
                        else 1200 * node_size_multiplier
                    )
                    for node in G
                ],
                font_size=font_size,
                arrowsize=arrowsize,
                labels=labeldict,
                with_labels=True,
            )

    # plt.tight_layout()
    print(list(nx.simple_cycles(G)))

    plt.axis("off")
    if not ret_image:
        plt.show()
    else:
        fig.canvas.draw()
        image = Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
    if path != "":
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    if ret_image:
        return image


def circuit_layout(G, node_size=700):
    pos = {}
    levels = {}
    horizontal_spacing = 2  # Adjust horizontal spacing as needed
    vertical_spacing = 1  # Adjust vertical spacing as needed

    def add_to_level(level, node):
        if level not in levels:
            levels[level] = []
        levels[level].append(node)

    def place_nodes(node, level=0):
        if node in pos:
            return
        add_to_level(level, node)
        width = len(levels[level])
        x = (width - 1) * horizontal_spacing
        y = -level * vertical_spacing
        pos[node] = (x, y)
        for succ in G.successors(node):
            place_nodes(succ, level + 1)

    # Initialize the placement with the first node
    root_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    for root in root_nodes:
        place_nodes(root)

    # Adjust the x positions to center-align each level
    for level, nodes in levels.items():
        min_x = min(pos[node][0] for node in nodes)
        max_x = max(pos[node][0] for node in nodes)
        offset = (max_x - min_x) / 2
        for node in nodes:
            x, y = pos[node]
            pos[node] = (x - offset, y)

    return pos


def get_traversable_vols(volumes):
    traversables = [[v for v in vol.next_vol] + [vol] for vol in volumes]
    # print("trav",traversables)
    part_in = lambda array, inarray: any([part in array for part in inarray])
    add_unadded = lambda inp, array: [
        array.append(data) for data in inp if (data not in array)
    ]
    result = []
    while len(traversables) > 0:
        while len(traversables[0]) == 0:
            traversables.remove(traversables[0])
            # print("rem")
            if len(traversables) == 0:
                break
        if len(traversables) == 0:
            break
        collect = traversables[0]
        for i, item in enumerate(traversables):
            if part_in(collect, item):
                add_unadded(item, collect)
                traversables[i] = []
        result.append(collect)

    change = True
    while change:
        change = False
        for part in result:
            t = part
            for i, partb in enumerate(result):
                if partb is not t:
                    if part_in(partb, part):
                        add_unadded(partb, part)
                        result.remove(partb)
                        change = True
    return result


# show startacccessible regions
def get_start_regs(volumes):
    start_vols = [vol for vol in volumes if no_refs(vol, volumes)]
    regions = []
    print("start_vols", len(start_vols))
    print([len(set(vol.next_vol)) for vol in start_vols])
    for vol in start_vols:
        region = [vol]
        to_check = [vol]
        while len(to_check) > 0:
            check = to_check.pop(0)
            region += check.next_vol
            to_check += check.next_vol  # as its cyclefree
        regions.append(set(region))
    return regions


def save_connections(volumes):
    return dict((vol, vol.next_vol.copy()) for vol in volumes)


def restore_connections(restoration_dict):
    for key, value in restoration_dict.items():
        key.next_vol = value
    return list(restoration_dict.keys())


def solve_q(volumes, p1, q, p2=Symbol("p2Sys"), verbose=False):
    assert type(p1) != Symbol
    p1val = p1
    p1 = Symbol("p1Sys")

    def get_vol_key(volume, var):
        for key, value in volume.variables.items():
            if key == var or value == var:
                return key

    e = calculate_simplified_network(volumes, verbose=verbose)
    result = solGS(e, q, p1, p2, add_equation=Eq(p1, p1val))
    equations = result[0]
    if verbose:
        print("-----------equations-----------", len(equations))
    for eq in equations:
        if verbose:
            print(eq)
    if verbose:
        print("--------------vars---------------", len(equations))
    equations = result[1]

    for eq in equations:
        if verbose:
            print(eq)
    print("->Solution:", result[2])
    for volume in e:
        # print(volume.__dict__)
        if hasattr(volume, "variables"):
            for solution in [result[2]]:
                # start and endvalues have different keys
                for var in volume.variables_searched:
                    if var in solution.keys():
                        volume.set_val_for_symbol(
                            get_vol_key(volume, var), solution[var]
                        )
                        # print("set ",var,solution[var])

    for vol in e:
        print(vol.get_symval(vol.Q_1), vol.get_symval(vol.P_1), vol.get_symval(vol.P_2))
        print("subsolving")
        vol.solve_lower_volumes()


def get_start_volumes(volumes):
    n_volumes = []
    start_volumes = []
    for vol in volumes:
        n_volumes += vol.next_vol
    if len(start_volumes) == 0:
        for vol in volumes:
            if vol not in n_volumes:
                start_volumes += [vol]
    return start_volumes


def solGS(volumes, system_q, system_p1, system_p2, add_equation=None, verbose=False):
    # get start_volumes
    start_volumes = get_start_volumes(volumes)
    # Q_in = sum(Q_startvols)
    start_eq_q = Eq(system_q, sum([s.Q_1 for s in start_volumes]))
    if verbose:
        print("starteq", start_eq_q)
    # same for end volumes
    end_volumes = [vol for vol in volumes if len(vol.next_vol) == 0]
    end_eq_q = Eq(system_q, sum([s.Q_1 for s in end_volumes]))
    if verbose:
        print("end_eq", end_eq_q)
    # set p1 and p2 of system equations:
    for start_volume in start_volumes:
        if not hasattr(start_volume, "no_p_change"):
            start_volume.set_val_for_symbol(start_volume.P_1, system_p1)
    for end_volume in end_volumes:
        if not hasattr(end_volume, "no_p_change"):
            end_volume.set_val_for_symbol(end_volume.P_2, system_p2)

    # set p1 and p2 for volumes that follow after each other
    p_quations = write_qual_p21(volumes, equations=True)

    # Kirchhoffsche Equations
    # volume equations
    for vol in volumes:
        vol.make_q_equations(volumes)
        assert vol.eq_q != False, vol.name
    for vol in volumes:
        vol.make_q_split_equations()
        assert vol.eq_q != False, vol.name

        # assert not next_q.has(oo, -oo, zoo, nan),"Infinity value in equation"

    vol_equations = [vol.eq_q for vol in volumes]
    system_eq = [
        start_eq_q,
        end_eq_q,
    ] + p_quations
    for eq in system_eq:
        assert not eq.has(
            oo, -oo, zoo, nan
        ), "Infinity value in equations for system (ends)"
        assert eq != False

    vol_equations, searched_vars = makeLGS(volumes)  # q=pdelta/res for each vol
    # vol_equations=[vol.eq_q for vol in volumes]

    for eq in vol_equations:
        assert not eq.has(
            oo, -oo, zoo, nan
        ), "Infinity value in equations for system (volumes)"
        assert eq != False
    equations = system_eq + vol_equations
    if verbose:
        print("Before set ", len(equations))
    equations = set(equations)
    if verbose:
        print("After set ", len(equations))
    for eq in equations:
        assert not eq.has(
            oo, -oo, zoo, nan
        ), "Infinity value in equations for system (volumes)"

    if add_equation != None:
        if verbose:
            print("added equation", add_equation)
        equations.add(add_equation)
    if verbose:
        print("------------Equations-------------")
    for eq in sorted([str(eq) for eq in equations]):
        if verbose:
            print(eq)
    if verbose:
        print("------------VARS-------------")
    for eq in sorted([str(eq) for eq in searched_vars]):
        if verbose:
            print(eq)
    assert False not in equations
    # print("Equations")
    # equations=sorted(equations,key=lambda x:list(x))
    # [print(we) for we in equations]

    return equations, searched_vars, solve(equations)  # ,searched_vars)


# methods to manually solve equations for q:
# system: set solved, solvable, unsolved
# vol: are all dependencies in solved AND required volumes have attributes set? -> solvable
# system: set solved, solvable, unsolved
# vol: are all dependencies in solved AND required volumes have attributes set? -> solvable
class volume_solver:
    def __init__(self, volumes, p1, q, vocal=False):
        self.initial_wrap = volume_wrap(volumes, p1, q, self)
        self.initial_wrap.solvable = True
        self.wraps = set()
        self.initial_wrap.unwrap(self)  # all resistances
        self.solved = set()
        self.unsolved = set(self.wraps)
        self.solvable = self.get_solvable_wraps()

    def solve(self, vocal=False):
        while len(self.solvable) > 0:
            new_intel = dict()
            for wrap in self.solvable:
                wrap.single_solve(self)
            self.solved = self.solvable | self.solved
            self.unsolved = self.unsolved - self.solvable
            for wrap in self.unsolved:
                wrap.update_solvable(self)

            if vocal:
                print()
                print("--------solved----------")
                for wrap in self.solvable:
                    print(str(wrap), "Q", wrap.q, "P1", wrap.p1, "P2", wrap.p2)
                print("------------------")
                print()

            print("solved/all_wraps", len(self.solved), "/", len(self.wraps))

            self.solvable = self.get_solvable_wraps()
        if vocal:
            print("unsolved: ")
            for wrap in self.unsolved:
                print(str(wrap))
                print("depends on ", [str(w) for w in wrap.depends_on])
                print()

    def get_solvable_wraps(self):
        solvable = set()
        for wrap in self.unsolved:
            if wrap not in self.solved and wrap.solvable:
                solvable.add(wrap)
        return solvable

    def get_wrap(self, wrap_id):
        for wrap in self.wraps:
            if wrap.id == wrap_id:
                return wrap
        return None


class dependency:
    def __init__(self, element, attr):
        self.element = element
        self.attr = attr

    def value(self):
        if hasattr(self.element, self.attr):
            res = getattr(self.element, self.attr)
            if type(res) == dependency:
                return res.value()
            return res
        return None

    def __str__(self):
        return "dep(" + str(self.element) + "," + str(self.attr) + ")"


""" def __str__(self):
        return str([vol.resistance for vol in self.volumes])+" "+str([len(self.pivot_vols),len(self.end_vols)])+str(self.resistance)#+str(self.q) """
import sympy


# return one wrap below
def calcQ(p1, p2, r):
    # print("calcq p1,p2,r", p1, p2, r)
    if r == 0:
        r = 0.00000000000000000000000000000000000001
    q = sympy.Symbol("q")
    qeq = sympy.Eq(q, (p1 - p2) / r)
    return float(sympy.solve(qeq)[0])


def calc_p2(q, p1, r):
    # print("calc_p2: q, p1, r", q, p1, r)
    if r == 0:
        # print("return p1")
        return p1
    p2 = sympy.Symbol("p2")
    qeq = sympy.Eq(q, (p1 - p2) / r)
    assert len(sympy.solve(qeq)) == 1, (sympy.solve(qeq), qeq)
    return float(sympy.solve(qeq)[0])


def calc_p1(q, p2, r):
    if r == 0:
        # print("return p1")
        return p2
    p1 = sympy.Symbol("p1")
    qeq = sympy.Eq(q, (p1 - p2) / r)
    assert len(sympy.solve(qeq)) == 1, (sympy.solve(qeq), qeq)
    return float(sympy.solve(qeq)[0])


import sympy


class volume_wrap:
    def __init__(self, volumes, p1, q, solving_system, p2=None):
        self.solving_system = solving_system
        self.depends_on = []  # all required previously solved wraps
        assert len(volumes) > 0
        self.volumes = list(set(volumes))  # .copy()
        self.q = q
        self.p1 = p1
        # no_ref=lambda vol, vols: any([vol not in v.next_vol for v in vols])
        self.pivot_vols = [vol for vol in volumes if no_refs(vol, self.volumes)]
        self.end_vols = [
            vol
            for vol in volumes
            if len([v for v in vol.next_vol if v in volumes]) == 0
        ]
        self.system_opens = True

        if len(self.end_vols) == 1:
            self.last_element = self.end_vols[0]
            self.system_opens = False
            self.end_vols_pre = [
                vol for vol in volumes if self.last_element in vol.next_vol
            ]
        self.helper_node = False
        self.pivot_vol = self.pivot_vols[0]
        if len(self.pivot_vols) > 1:
            self.helper_node = True
            self.pivot_vol = form_vol(
                following=self.pivot_vols, res=0, p2=p1, p1=p1
            )  # this essentially bundles the beginging volumes together to aloow for resistance calculations
            self.pivot_vol.name = "empty_pivot"
            # print("use NEW")
        #:print("PIVOT ", self.pivot_vol.resistance)
        self.resistance = get_resistance(
            self.pivot_vol, self.volumes + [self.pivot_vol]
        )
        self.p2 = p2
        self.id = self.ident()
        self.solvable = False
        self.pivot_vol_next_vol_involumes = [
            vol for vol in self.pivot_vol.next_vol if vol in self.volumes
        ]

    def q_solvable(self):
        val = lambda x: x.value() if type(x) == dependency else x
        # print(type(self.q))
        if type(self.q) != list:
            if val(self.q) != None:
                return True
            else:
                return False
        # print("q",self.q)
        for start_pressure, p2, next_resistance in self.q:
            if (val(start_pressure) == None) or (val(p2) == None):
                print("missing:", str(start_pressure), str(p2))
                return False
        return True

    def update_solvable(self, system):
        val = lambda x: x.value() if type(x) == dependency else x

        self.solvable = False
        if all([dependent_wrap in system.solved for dependent_wrap in self.depends_on]):
            # print("dependency solved")
            if self.q_solvable():
                if val(self.p1) != None:
                    if val(self.p2) != None or type(self.p2) != dependency:
                        # print("q solvable",self.q)
                        # update vals

                        self.solvable = True

    def ident(self):
        """return str(sum([id(vol) for vol in self.pivot_vols])) + str(
            sum([id(vol) for vol in self.end_vols])
        )"""
        return self.resistance + sum([id(vol) for vol in self.volumes])

    def __str__(self):
        appendix = ""
        if hasattr(self, "create_method"):
            appendix = " " + str(self.create_method)
        if all([hasattr(vol, "name") for vol in self.volumes]):
            return (
                str([v.name for v in self.volumes]) + appendix
            )  # str(len(self.volumes))+","+str(len(self.pivot_vols))+","+str(len(self.end_vols))+","
        return (
            "#vols "
            + str(len(self.volumes))
            + " #pivot "
            + str(len(self.pivot_vols))
            + " #end "
            + str(len(self.end_vols))
            + " p1 "
            + str(self.p1)
            + " r "
            + str(self.resistance)
            + appendix
        )

    def single_solve(self, system, verbose=False):
        if verbose:
            print(
                "single solve",
                len(self.volumes),
                self.resistance,
                self.system_opens,
                str(self),
            )
        self.q_value = 0
        val = lambda x: x.value() if type(x) == dependency else x

        # print(self.solvable, self.q)
        if type(self.q) == list:
            # print("list detected!!!!!!!!!!!!");print(self.q);print("--------------------------")
            for start_pressure, p2, next_resistance in self.q:
                # print(val(start_pressure), val(p2), val(next_resistance))
                # try:
                self.q_value += calcQ(val(start_pressure), val(p2), next_resistance)
                assert next_resistance == self.resistance
                """ except:
                    assert False,self.q """
        else:
            self.q_value = val(self.q)
        self.q = self.q_value
        # always calc opening
        # print("calcp2vol",(self.q, val(self.p1), self.pivot_vol.resistance))
        self.pivot_vol.p2 = calc_p2(self.q, val(self.p1), self.pivot_vol.resistance)
        self.pivot_vol.q = self.q
        self.pivot_vol.p1 = val(self.p1)
        if verbose:
            print("q", self.q, val(self.q))
            print("calcp2", val(self.p1))
        if self.p2 == None:
            self.p2 = calc_p2(self.q, val(self.p1), self.resistance)
        else:
            self.p2 = val(self.p2)
            assert self.p2 != None
        if verbose:
            print(
                "self,q",
                str(self),
                self.q_value,
                "p1p2r",
                val(self.p1),
                val(self.p2),
                val(self.resistance),
            )

        if verbose:
            print(
                "calculated p2 wrap",
                val(self.p2),
                "q,p1,res",
                (self.q, val(self.p1), self.resistance),
                str(self),
            )
        if verbose:
            print(
                "set p1, p2, q for pivot:",
                str(self.pivot_vol),
                self.pivot_vol.p1,
                self.pivot_vol.p2,
                self.pivot_vol.q,
            )
        for vol in self.end_vols:
            vol.p2 = val(self.p2)
        for vol in self.pivot_vols:
            vol.p1 = val(self.p1)

        # set q if p2 is already given
        v_to_finsolve = None
        if len(self.volumes) == 1:
            v_to_finsolve = self.volumes[0]
            assert hasattr(v_to_finsolve, "p2") and hasattr(v_to_finsolve, "p1")
            v_to_finsolve.q = calcQ(
                val(v_to_finsolve.p1), val(v_to_finsolve.p2), v_to_finsolve.resistance
            )
            # v_to_finsolve.p1=val(self.p1)
        if (
            not self.system_opens
        ):  # set q p1 for last element and p2 for pre endvol elements
            self.last_element.p1 = calc_p1(
                self.q, val(self.p2), self.last_element.resistance
            )
            self.last_element.q = self.q
            self.last_element.p2 = val(self.p2)

            assert [self.last_element] == self.end_vols  # also in end vols
            # print("calculated p2", val(self.p2), "q,p1,res",(self.q,val(self.p1), self.resistance))
            if verbose:
                print(
                    "set p1, p2, q for endvol:",
                    str(self.last_element),
                    self.last_element.p1,
                    self.last_element.p2,
                    self.last_element.q,
                )

            for vol in self.end_vols_pre:
                vol.p2 = val(self.last_element.p1)

        if verbose:
            print("----------" + str(self) + "-----")

        if verbose:
            print(
                "single solveresult",
                val(self.q),
                val(self.p1),
                val(self.p2),
                self.resistance,
            )
        if verbose:
            print("---------------")

    def following(self, vol):
        next_vol = [v for v in vol.next_vol if v in self.volumes]
        return flatten(
            [next_vol + [self.following(v) for v in vol.next_vol if v in self.volumes]]
        )

    def split_volume_para_node_paras(self, next_vol):
        """
        split into wraps for each node where all nodes join tpgether
        """
        d = get_next_to_each_other(next_vol, volume_scope=self.volumes)
        for key, values in d.items():
            if all([v in values for v in next_vol]):
                join_node = key
        q = dependency(self.pivot_vol, "q")
        assert type(q) != list, "splitting should be done first, so q is set"
        join_following = self.following(join_node)  # TODO redundant: + [join_node]
        vol_to_join = [vol for vol in self.volumes if vol not in join_following] + [
            join_node
        ]
        wrap1 = volume_wrap(
            vol_to_join, self.p1, q, self.solving_system
        )  # pivot to join
        wrap2 = volume_wrap(
            join_following, dependency(join_node, "p2"), q, self.solving_system
        )  # join to end
        return wrap1, wrap2

    def one_merge_node(self, next_vol):
        """one_merge_node return if for next_vol all resistors connect to one at any point

        Args:
            next_vol (_type_): _description_
        """
        d = get_next_to_each_other(next_vol, volume_scope=self.volumes)
        # if all from next_vol are joined at one join_node, return true
        for values in d.values():
            if all([v in values for v in next_vol]):
                # print("all in values", [v.name for v in values], [v.name for v in next_vol])
                return True
        return False

    def unwrap(self, system, verbose=False):
        # if beginning volume(r/f) is single and end volume is single(r), then strip end -> both get calculated
        # if beginning volume(r/f) is single and end volume is multiple then strip beginning -> only beginning calculated
        # if both are multiple(first is always single per creation, but if it has multiple next volumes), then check if inbetween is a mergepoint and split first part from second part(s)
        if len(self.volumes) > 1:
            """if (
                self.system_opens or self.last_element == self.pivot_vol
            ):"""  # one beginning, multiple at end -> strip first
            # check if both beginning and end are parallel and if thats the case then split this volume
            if (
                len(self.pivot_vol_next_vol_involumes) > 1
                and len(self.end_vols) > 1
                and len(self.pivot_vol_next_vol_involumes) != len(self.end_vols)
                and self.one_merge_node(self.pivot_vol_next_vol_involumes)
            ):
                print("split")
                split_volumes = self.split_volume_para_node_paras(
                    self.pivot_vol_next_vol_involumes
                )
                for wrap in split_volumes:
                    wrap.create_method = "split"
                    wrap.bigger = self
                    wrap.unwrap(system)

            elif len(self.end_vols) == 1:
                # check if end is single and split end away
                # split away the end and split lower wrap further
                volumes_wo_end = list(set(self.volumes))
                volumes_wo_end.remove(self.last_element)  # remove last element
                # q is the same as last element q that is solvable!
                # p2 is the same as p1 last element
                # p1 is the same as self p1
                assert len(volumes_wo_end) == len(self.volumes) - 1
                if len(volumes_wo_end) > 0:
                    split_wrap = volume_wrap(
                        volumes_wo_end,
                        dependency(self.pivot_vol, "p1"),
                        dependency(self.last_element, "q"),
                        self.solving_system,
                    )
                    split_wrap.create_method = "endloss"
                    split_wrap.depends_on.append(self)
                    split_wrap.bigger = self

                    split_wrap.unwrap(system)
            elif len(self.pivot_vol_next_vol_involumes) == 1:
                # serial beginning split first vol
                volumes_wo_first = list(set(self.volumes))
                volumes_wo_first.remove(self.pivot_vol)  # remove first element
                # q is the same as first element q that is solvable!
                # p2 is the same as p2 self
                # p1 is the same as self pivot  p2
                if len(volumes_wo_first) > 0:
                    split_wrap = volume_wrap(
                        volumes_wo_first,
                        dependency(self.pivot_vol, "p2"),
                        dependency(self.pivot_vol, "q"),
                        self.solving_system,
                    )
                    split_wrap.create_method = "firstloss"
                    split_wrap.depends_on.append(self)
                    split_wrap.bigger = self
                    split_wrap.unwrap(system)
            elif len(self.pivot_vol_next_vol_involumes) > 1:
                # parallel beginning split into strands
                lower_wraps = []
                # use strand_vol or create a new volume for connected strands
                d = get_next_to_each_other(
                    self.pivot_vol_next_vol_involumes, volume_scope=self.volumes
                )
                strands, merge_nodes = get_strands(d, self.pivot_vol_next_vol_involumes)

                if True and all([hasattr(vol, "name") for vol in self.volumes]):
                    print(
                        "strands:",
                        [
                            (
                                [
                                    vol.name if hasattr(vol, "name") else vol
                                    for vol in strand[0]
                                ]
                                if type(strand) == tuple
                                else strand.name
                            )
                            for strand in strands
                        ],
                        "from vols",
                        [vol.name for vol in self.volumes],
                        "wrap",
                        str(self),
                        "own pivot_elem",
                        [vol.name for vol in self.pivot_vols],
                        "endvols",
                        len(self.end_vols),
                    )
                    for strand in strands:
                        if type(strand) == tuple:
                            print(
                                "one strand is pivot",
                                len(set(self.pivot_vols).difference(strand[0])) == 0
                                and len(set(self.pivot_vols)) == len(strand[0]),
                            )
                # for each strand create a wrap that has either strandvol as pivot or holds all startvols if they merge later(1n1x wrap)
                # p1 is pivot p2
                # q is calculated with p2
                # p2 is self p2
                for strand in strands:
                    if type(strand) != tuple:
                        assert strand != []
                        strand_vols = self.following(strand) + [strand]
                        strand_vols = list(set(strand_vols))
                        next_resistance = get_resistance(
                            strand, volumescope=self.volumes
                        )
                        q = (
                            dependency(self.pivot_vol, "p2"),
                            dependency(self, "p2"),
                            next_resistance,
                        )
                        if len(strand_vols) > 0:
                            new_wrap = volume_wrap(
                                strand_vols,
                                dependency(self.pivot_vol, "p2"),
                                [q],
                                self.solving_system,
                            )  # ,p2=dependency(self,"p2"))
                            new_wrap.create_method = "strand_sep_single_strand"

                            lower_wraps.append(new_wrap)
                    else:
                        strand_vols = []
                        start_is_same_as_pivot_elems = False
                        if len(set(self.pivot_vols).difference(strand[0])) == 0 and len(
                            set(self.pivot_vols)
                        ) == len(strand[0]):
                            start_is_same_as_pivot_elems = True

                        for strand_vol_starter in strand[0]:
                            strand_vols += self.following(strand_vol_starter) + [
                                strand_vol_starter
                            ]
                        strand_vols = list(set(strand_vols))
                        if len(strand_vols) > 0:
                            print("in strand 0: ", [vol.name for vol in strand[0]])
                            resistance_vol = form_vol(
                                res=0, following=list(strand[0]), p1=None, p2=None
                            )
                            resistance_vol.name = "next_res_calculate_vol"
                            next_resistance = get_resistance(
                                resistance_vol,
                                volumescope=strand_vols + [resistance_vol],
                            )
                            q = (
                                dependency(self.pivot_vol, "p2"),
                                dependency(self, "p2"),
                                next_resistance,
                            )
                            new_wrap = volume_wrap(
                                strand_vols,
                                dependency(self.pivot_vol, "p2"),
                                [q],
                                self.solving_system,
                            )  # ,p2=dependency(self,"p2"))
                            new_wrap.create_method = "strand_sep_multi"

                            lower_wraps.append(new_wrap)

                for wrap in lower_wraps:
                    wrap.depends_on.append(self)
                    wrap.bigger = self
                    wrap.unwrap(system)
            else:
                assert False
        assert system.get_wrap(self.id) == None
        system.wraps.add(self)

    def create_wrap(
        self,
        follow,
        start_pressure,
        p2,
        next_resistance,
        last=False,
        verbose=False,
        register_prev_qs=False,
    ):
        q = [(start_pressure, p2, next_resistance)]
        new_wrap = volume_wrap(
            follow,
            start_pressure,
            q,
            self.solving_system,
            # div q according to resistance share in strands
        )
        # if verbose:print("new_wrap",new_wrap, "depends on ", self)
        wrap_in_system = self.solving_system.get_wrap(new_wrap.id)

        if last:
            if verbose:
                print("Last:", str(self), "->", str(new_wrap), len(new_wrap.end_vols))
                print("dropped:", str(self.last_element))
                print()  # TODO if mehrfachdrop: self.q abhängig von neuen (inputs)!
        else:
            if verbose:
                print(
                    "First:", str(self), "->", str(new_wrap), len(new_wrap.end_vols)
                )  # , [v.name for v in new_wrap.end_vols])

        if wrap_in_system == None:
            real_wrap = new_wrap
        else:
            real_wrap = wrap_in_system
        if wrap_in_system == None:
            new_wrap.depends_on.append(
                self
            )  # depends on start p2 solve and potentially end p1 solve
            return new_wrap

        else:
            # add that system requires self
            # wrap_in_system.q.append(q[0])
            print("QADD")
            wrap_in_system.depends_on.append(self)

        """ if wrap_in_system != None:
            print("EIEIEI")
            ...#assert False, len(new_wrap.volumes)
        else:
            return new_wrap """


def flatten(items, seqtypes=(list, tuple)):
    for i, x in enumerate(items):
        while i < len(items) and isinstance(items[i], seqtypes):
            items[i : i + 1] = items[i]
    return items


def calculate_parallel_resistance(list_of_resistances):
    """
    Calculates the total resistance of a circuit where all resistances are in parallel.

    Parameters:
    - list_of_resistances: A list of resistances in the circuit.

    Returns:
    - The total resistance of the circuit.
    """
    if not list_of_resistances:
        return 0  # No resistances = no total resistance
    reciprocal_sum = sum(1 / R for R in list_of_resistances if R > 0)
    if reciprocal_sum == 0:
        return float(
            "inf"
        )  # If all resistances are 0, the total resistance is infinite
    # reverse value
    total_resistance = 1 / reciprocal_sum
    return total_resistance


def no_refs(volume, volumes):
    for v in volumes:
        if volume in v.next_vol:
            return False
    return True


class Object(object):
    pass


class form_vol(resistant_volume):
    VISCOSITY = 0.0035

    PI = 3.14159265359

    def __init__(self, res, p2, following, p1, Q=None, name=None):
        super().__init__(
            radius=1, length=1, identifier=None, res=res, P1=p1, P2=p2
        )  # to obtain the same identifier
        self.resistance = res
        self.next_vol = following
        if p2 != None:
            self.p2 = p2
        if p1 != None:
            self.p1 = p1
        self.path_indices = [0, 0]
        self.id = id(self)
        self.vessel = Object()
        self.vessel.type = "typeless_formvol_vessel"

        self.vessel.associated_vesselname = "ImaginaryConnectionVessel"

        self.to_append = []  # vessel and index
        self.prevs = set()
        if name != None:
            self.name = name
            self.id = name
        self.Q_1, self.W, self.P_1, self.P_2 = symbols(
            f"Q_{self.id}, W_{self.id}, P1_{self.id}, P2_{self.id}", positive=True
        )
        self.variables = {
            self.Q_1: Q,
            self.W: self.resistance,
            self.P_1: p1,
            self.P_2: p2,
        }
        self.update_eq_q()
        self.q_equations = []
        self.q_splits = 0
        self.q_split_symbols = []

    def get_q_for_equation(self, other_vol=None):
        if other_vol == None:
            self.q_split_symbols.append(
                sympy.Symbol(f"frac_{self.id}_{self.q_splits}")
            )  # addlater qsplits
            self.q_splits += 1
        else:
            self.q_split_symbols.append(
                sympy.Symbol(f"frac_{self.id}_{other_vol.id}")
            )  # addlater qsplits

        return self.q_split_symbols[-1]

    def make_q_equations(self, volumes):
        prev = set()
        for vol in volumes:
            if self in vol.next_vol:
                prev.add(vol)
        if len(prev) > 0:
            eq = Eq(
                self.Q_1, sum([vol.get_q_for_equation(self) * vol.Q_1 for vol in prev])
            )
            self.q_equations.append(eq)

    def make_q_split_equations(self):
        if len(self.q_split_symbols) > 0:
            eq = Eq(sum([split_sym for split_sym in self.q_split_symbols]), 1)
            self.q_equations.append(eq)

    def update_eq_q(self):
        self.variables_searched = []
        # Equation for Q:
        self.eq_q = Eq(self.Q_1, (self.P_1 - self.P_2) / self.W)
        for key, value in self.variables.items():
            if value != None:
                self.eq_q = self.eq_q.subs(key, value)
                if type(value) == Symbol:
                    self.variables_searched.append(value)
                elif type(value) == float:
                    self.variables[key] = Float(value)
            else:
                self.variables_searched.append(key)

    def get_symval(self, key):
        out = self.variables[key]
        if out == None:
            out = key
        return out

    def set_val_for_symbol(self, sym, val):
        if sym in self.variables.keys():
            self.variables[sym] = val
            # else:
            # print(sym, "not in vars")

            self.update_eq_q()

    def print_self(self):
        return self.eq_q

    def __str__(self):
        if hasattr(self, "name"):
            return str(self.name)
        else:
            return str(id(self))


def get_next_strand_end(marked_nodes, node_before):
    marked_nodes = set(marked_nodes)

    def following(vol):
        return flatten([vol.next_vol + [following(v) for v in vol.next_vol]])

    path = following(node_before)
    for element in path:
        if element in marked_nodes and element != node_before:
            return element
    return node_before


def lst_distance(l1, l2):
    """lst_distance Generate the distance between two lists as the relative amount of steps taken between them until they have the same element at pointer
    e.g. a=[1,2,3,4]; b=[6,5,3]; d(a,b)=sum(2+1/ len(a),2+1/len(b))=sum(3/4,3/3)=7/4

    Args:
        l1 (_type_): _description_
        l2 (_type_): _description_
    """
    k = -1
    for a, element in enumerate(l1):
        for b, element2 in enumerate(l2):
            if element == element2:
                return min(a, b)


def get_next_to_each_other(
    next_vols, path_cut_node=None, volume_scope=set(), verbose=False
):
    """
    for given volumes group by shared pathnode. if cut node if given only group if shared node comes before path_cut_node
    """
    if type(next_vols) == set:
        next_vols = list(next_vols)

    def following(vol, path_cut_node, volume_scope=set()):
        """following return all fllowing resistor nodes that do not include a path_cut node and are ion the volume scope

        Args:
            vol (_type_): _description_
            path_cut_node (_type_): _description_
            volume_scope (_type_, optional): _description_. Defaults to set().

        Returns:
            _type_: _description_
        """
        if path_cut_node in vol.next_vol:
            assert len(vol.next_vol) == 1
            return []
        next_vol = [
            v for v in vol.next_vol if v in volume_scope or volume_scope == set()
        ]
        return flatten(
            [
                next_vol
                + [following(v, path_cut_node, volume_scope) for v in vol.next_vol]
            ]
        )

    # finde common elements in following resistances und speichere mit first common node key
    next_to_each_other = (
        dict()
    )  # all volumes that have a path that share a node at some time
    lists = [following(vol, path_cut_node, volume_scope) for vol in next_vols]
    if verbose:
        print([[vol.resistance for vol in ls] for ls in lists])
    for v, lst in enumerate(lists):
        for v2, lst2 in enumerate(lists):
            if v != v2:
                vol = next_vols[v]
                vol2 = next_vols[v2]

                common_elements = set(lst2)

                # Find intersection between current set and next list
                common_elements &= set(lst)

                if len(common_elements) == 0:
                    continue
                else:
                    for element in lst:
                        if element in common_elements:
                            first_common_node = element
                            # if shared nodes then group
                            if first_common_node not in next_to_each_other:
                                next_to_each_other[first_common_node] = {vol}
                            next_to_each_other[first_common_node].add(vol2)
                            break

    return next_to_each_other


def get_resistance(volume, volumescope=set(), verbose=False):
    res.volume_scope = set(volumescope)
    # print("cal res, scope", volumescope)
    return res(volume, start=True, verbose=verbose)


def get_strands(d, next_vol):
    """return all connected startvols as sets and addinionally all paralllel ones"""
    parallels, merge_nodes = list(d.values()), list(d.keys())
    strands = []  # single or merging strands
    # all volumes that are assigned to a parallel are not single strands:
    for vol in next_vol:
        found = False

        for value in parallels:
            if vol in value:
                found = True
        if not found:
            strands.append(vol)
    # also return all volumes that connect together at ANY point -> largest set
    next_vol_in_dict = dict()  # per nextvol volume store the biggest join node set
    dvalues = list(d.values())
    for vol in next_vol:
        biggest_set = set()
        for volset in dvalues:
            if vol in volset:
                if len(volset) > len(biggest_set):
                    biggest_set = volset
        next_vol_in_dict[vol] = biggest_set
    groups = []
    for volset in next_vol_in_dict.values():
        stored = False
        for group in groups:
            if len(volset.difference(group)) == 0 and len(group) == len(
                volset
            ):  # len eq not needed
                stored = True
        if not stored:
            groups.append(volset)

    def merge_node_for_group(group, d):
        for node, g in d.items():
            if len(group.difference(g)) == 0 and len(group) == len(g):
                return node

    for group in groups:
        strands.append((group, merge_node_for_group(group, d)))
    """ substitute_lists = []
    # other volumes are forming a strand together
    for key in d:
        to_rem = []
        for key2 in d:
            if len(d[key2]) < len(d[key]):
                if len(d[key] & d[key2]) > 0:
                    to_rem.append(key2)

        for k2 in to_rem:
            for val in d[k2]:
                if val in d[key]:
                    d[key].remove(val)
        substitute_lists.append((key, key2))
    parallels, merge_nodes = list(d.values()), list(d.keys())
    for i, val in enumerate(parallels):
        strands.append((val, merge_nodes[i])) """
    return strands, merge_nodes


import numpy as np


class solve_object:
    def __init__(self, boolvaluecount, join_node, setsize):
        self.dependencies = [0 for i in range(boolvaluecount)]
        self.paravaluelist = []
        self.join_node = join_node
        self.nodes_to_cut = []
        self.solved = False
        self.n_lower_res = 0  # number of outer layers that depend on self

    def add_depend(self, i):
        self.dependencies[i] = 1

    def rem_append(self, i):
        self.dependencies[i] = 0

    def n_dependencies(self):
        return len([1 for i in self.dependencies if i == 1])

    def add_value(self, paravalue):
        self.paravaluelist.append(paravalue)

    def solve(self):
        self.solved = True
        self.value = calculate_parallel_resistance(self.paravaluelist)
        if len(self.nodes_to_cut) > 0:
            self.value += res(self.join_node, nodes_to_cut=self.nodes_to_cut)
        return self.value


def solve_parallel_strands(d):
    """solve_parallel_strands Returns the parallel resistance of a given node dict that stores join nodes and previous nodes that lead to it
        this solves up to the join:node, the node itself has to be added!
    Args:
        d (_type_): _description_

    Returns:
        _type_: _description_
    """
    paras = [solve_object(len(d), k, len(d[k])) for k in d]
    for i, nodeset in enumerate(d.values()):
        for j, nodeset2 in enumerate(d.values()):
            if i == j:
                continue
            if nodeset.issubset(nodeset2):  # i is sub of j
                paras[j].add_depend(i)

    for i, para in enumerate(paras):
        for j, b in enumerate(
            para.dependencies
        ):  # set own outer note to limit lower layer addition of ends
            if b == 1:
                paras[j].nodes_to_cut.append(para.join_node)
        # remove dependencies that lay in one another (dep on 3,1 when 3 depends on 1 remove 1)-> biggest subsets
        for depindex, b in enumerate(para.dependencies):
            if b == 0:
                continue
            dependency_node = paras[depindex]
            # rem all dependency nodes that the dependency node has
            for j, bb in enumerate(dependency_node.dependencies):
                if para.dependencies[j] == 1 and dependency_node.dependencies[j] == 1:
                    para.dependencies[j] = 0

    for i, para in enumerate(paras):
        for depindex, b in enumerate(para.dependencies):
            if b == 1:
                paras[depindex].n_lower_res += 1

    no_dependent_on_nodes = [para for para in paras if para.n_lower_res == 0]
    # added dependencies, now add res in lower ones:
    for i, para in enumerate(paras):
        if para.n_dependencies() == 0:
            para.paravaluelist = [
                res(node, outer_join_node=para.join_node, nodes_to_cut=[para.join_node])
                for node in d[para.join_node]
            ]
    ##print([(para.join_node.name,i, para.dependencies)for i,para in enumerate(paras)])
    # add values of strands that dont have a previous join node but also end in an upper layer:
    # node in d[join_node] but not in union(lower_layer_nodes)
    for para in paras:
        if para.n_dependencies() > 0:
            union_lower_layers = (
                set()
            )  # nodes that lead to same node and join on previous one
            for k, b in enumerate(para.dependencies):
                if b == 1:
                    union_lower_layers = union_lower_layers | d[paras[k].join_node]
            solo_values = d[para.join_node].difference(union_lower_layers)
            for node in solo_values:
                para.paravaluelist.append(
                    res(
                        node,
                        outer_join_node=para.join_node,
                        nodes_to_cut=[para.join_node],
                    )
                )  # outer join node to avoid returning join node if its the only one to the next join node

    solved = 0
    prev_solved = -1
    while solved < len(paras):
        if solved == prev_solved:
            print("NO SOLUTION")
            break

        prev_solved = solved
        for i, para in enumerate(paras):
            if para.n_dependencies() == 0 and para.solved == False:
                parasol = para.solve()
                # print(para.nodes_to_cut)
                solved += 1
                for para in paras:
                    if para.dependencies[i] == 1:
                        para.add_value(parasol)
                        para.rem_append(i)
        # print(solved, len(paras))
    ##print([(para.join_node.name, para.paravaluelist,para.value) for para in paras])
    ##print("result: ",[(solve_node.join_node,solve_node.value) for solve_node in no_dependent_on_nodes])
    return [
        (solve_node.join_node, solve_node.value) for solve_node in no_dependent_on_nodes
    ]


def res(
    volume,
    outer_join_node=None,
    nodes_to_cut=set(),
    start=False,
    verbose=False,
    volume_scope=set(),
):
    """res return resistance from volume until outer join_node or between join_node and outer join_node

    Args:
        volume (_type_): _description_
        join_node (_type_): _description_
        outer_join_node (_type_): _description_
    """
    if start:
        res.nodes = set()
    if (
        volume == None
        or volume == outer_join_node
        or (volume not in res.volume_scope and len(res.volume_scope) > 0)
    ):
        # print("volscope")
        return 0
    next_vol = volume.next_vol
    if verbose:
        if outer_join_node != None:
            print(
                "res (",
                volume.resistance,
                ") following: ",
                [vol.resistance for vol in next_vol],
                " reserved node :",
                outer_join_node.resistance,
            )

        else:
            print(
                "res (",
                volume.resistance,
                ") following: ",
                [vol.resistance for vol in next_vol],
                " reserved node :",
                outer_join_node,
            )

    if len(next_vol) == 1:
        if next_vol[0] != outer_join_node and next_vol[0] not in nodes_to_cut:
            return volume.resistance + res(
                next_vol[0], outer_join_node, nodes_to_cut=nodes_to_cut
            )  # serial
        else:
            if verbose:
                print("retself")
            return volume.resistance  # para end

    elif len(next_vol) == 0:
        return volume.resistance  # any end

    else:
        # nvol >1 parallel open
        if len(res.volume_scope) > 0:
            d = get_next_to_each_other(
                next_vol, volume_scope=res.volume_scope
            )  # TODO limit?
        else:
            d = get_next_to_each_other(next_vol)
        joining_nodes = set()
        for join_subset in d.values():
            joining_nodes = joining_nodes | join_subset
        not_joining = set(next_vol).difference(joining_nodes)
        ##print("next_vol, joining, not joining",[v.name for v in  next_vol],[v.name for v in   joining_nodes], [v.name for v in  not_joining])
        end_node_valuelist = solve_parallel_strands(d)
        total_resistances = []
        for node, value in end_node_valuelist:
            ##print("end node value", node.name)
            total_resistances.append(
                value + res(node, outer_join_node, nodes_to_cut=nodes_to_cut)
            )  # res until join node plus follow

        for node in not_joining:
            total_resistances.append(res(node, nodes_to_cut=d.keys()))

        return volume.resistance + calculate_parallel_resistance(total_resistances)


def OLDres(
    volume,
    outer_join_node=None,
    outer_strand_end=None,
    nodes_to_cut=set(),
    outer_layer=True,
    start=False,
    verbose=False,
):
    """res return resistance from volume until outer join_node or between join_node and outer join_node

    Args:
        volume (_type_): _description_
        join_node (_type_): _description_
        outer_join_node (_type_): _description_
    """
    if start:
        res.nodes = set()
    if volume == None or (volume not in res.volume_scope and len(res.volume_scope) > 0):
        return 0
    next_vol = volume.next_vol
    if verbose:
        if outer_join_node != None:
            print(
                "res (",
                volume.resistance,
                ") following: ",
                [vol.resistance for vol in next_vol],
                " reserved node :",
                outer_join_node.resistance,
            )

        else:
            print(
                "res (",
                volume.resistance,
                ") following: ",
                [vol.resistance for vol in next_vol],
                " reserved node :",
                outer_join_node,
            )

    if len(next_vol) == 1:
        if (
            next_vol[0] != outer_join_node
            and next_vol[0] != outer_strand_end
            and next_vol[0] not in nodes_to_cut
        ):
            return volume.resistance + res(
                next_vol[0], outer_join_node, outer_strand_end, outer_layer=outer_layer
            )  # serial
        else:
            if verbose:
                print("retself")
            return volume.resistance  # para end

    elif len(next_vol) == 0:
        return volume.resistance  # any end

    else:
        # nvol >1 parallel open
        d = get_next_to_each_other(next_vol)
        strands, merge_nodes = get_strands(d, next_vol)
        res.nodes = res.nodes | set(merge_nodes)
        if verbose:
            print(
                "strands:",
                [
                    (
                        strand.resistance
                        if type(strand) != tuple
                        else [s.resistance for s in strand[0]]
                    )
                    for strand in strands
                ],
            )

        # strands that are a set join together others end "by themselve" so are in parallel to rest
        parallel_vals = []
        parallel_val_presol = dict()
        final_node = None
        for strand in strands:
            if type(strand) == tuple:
                # one merging strand gives one val, substitutes have to be added
                outer_strand_end = get_next_strand_end(
                    merge_nodes, strand[1]
                )  # next biggest strand that ends in the "next" join_node
                if strand[1] == outer_strand_end and len(res.nodes) != 1:  # last node
                    final_node = strand[1]
                    parallel_vals.append(
                        calculate_parallel_resistance(
                            [
                                res(strandval, strand[1], outer_layer=False)
                                for strandval in strand[0]
                            ]
                        )
                    )
                else:
                    parallel_vals.append(
                        calculate_parallel_resistance(
                            [
                                res(strandval, strand[1], outer_layer=False)
                                for strandval in strand[0]
                            ]
                        )
                        + res(strand[1], outer_strand_end, outer_layer=False)
                    )

            else:
                # one strand gives one parallel val
                parallel_vals.append(res(strand))
        if verbose:
            print(
                "parallel_vals",
                parallel_vals,
                " -> ",
                calculate_parallel_resistance(parallel_vals),
            )
            print(len(res.nodes) == 1)
        if outer_layer:
            add = res(final_node, outer_layer=True)

        else:
            add = 0
            if (
                final_node != None and len(final_node.next_vol) > 0
            ):  # do not use last volumes resistance as that one will be added by outer layer if its a node itself
                add = res(
                    final_node, nodes_to_cut=res.nodes, outer_layer=False
                )  # from inner end_node to outer_endnode (of higher layer)

        return volume.resistance + calculate_parallel_resistance(parallel_vals) + add


def follow(path_nodes, reserved_nodes, depth=0):
    """
    following resistance until outer_endnode is reached where all paths converge
    """
    print("followcall", [node.resistance for node in path_nodes])
    if len(path_nodes) == 1:
        return 0
    k = [node.resistance for node in path_nodes]
    k
    reserved_nodes
    following = lambda vol: flatten(
        [vol.next_vol + [following(v) for v in vol.next_vol]]
    )
    pathes = [following(node) for node in path_nodes]
    print(
        "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
    )
    print("pathes")
    for path in pathes:
        print([p.resistance for p in path])
    print("-----")
    print("reserved:")
    print([node.resistance for node in reserved_nodes])
    print("-----")
    merge_node = get_merge_node(pathes, reserved_nodes)
    if merge_node != None:
        print(
            [node.resistance for node in path_nodes], "merge ON", merge_node.resistance
        )
        print("////////////////////////////////////////////////")
        return res(merge_node, reserved_nodes, depth=depth + 1), merge_node
    print("nomerge", merge_node)
    print("////////////////////////////////////////////////")

    return 0, merge_node


def checkYdelta_transform_situation(volumes):
    # ydelta transform is needed if 2 pathes touch each other but also continue:
    # so there are common nodes but the common onde is not the only following node for incoming volumes to that node
    # save all prevs:
    for volume in volumes:
        # Iterate through the next resistors of each resistor
        for next_volume in volume.next_vol:
            for another_next_volume in next_volume.next_vol:
                # Check if there is a Y-delta configuration
                if volume in another_next_volume.next_vol:
                    print(
                        f"Y-delta configuration found between {volume.vessel.associated_vesselname}, {next_volume.vessel.associated_vesselname}, and {another_next_volume.vessel.associated_vesselname}"
                    )


def set_res_results(res):
    for vol in res:
        if type(vol) != form_vol:
            vol.set_val_for_symbol(vol.Q_1, vol.q)
            vol.set_val_for_symbol(vol.P_1, vol.p1)
            vol.set_val_for_symbol(vol.P_2, vol.p2)


def substitute_from_system(vol, system, accumulator=None, verbose=False):
    """substitute_from_system Remove two volumes (vol and accumulator) or a list of parallel volumes from the system and replace them with a combined volume

    Args:
        vol (_type_): _description_
        system (_type_): _description_
        accumulator (_type_, optional): _description_. Defaults to None.
        verbose (bool, optional): _description_. Defaults to False.
    """
    # vol is rightside, accumulator leftside
    if type(vol) != list:  # combine serial resistors
        assert accumulator != None
        if verbose:
            print("serialrem", vol.resistance, "for", accumulator.resistance)
        system.remove(vol)
        system.remove(accumulator)

        resistance = vol.resistance + accumulator.resistance
        new_vol = resistant_volume(
            radius=1,
            length=1,
            identifier=str(accumulator.id) + "_" + str(vol.id),
            lower_type="serial",
            lower_volumes=[accumulator, vol],
            res=resistance,
        )
        new_vol.path_indices = None
        # update connections
        new_vol.next_vol = vol.next_vol
        new_vol.prevs = accumulator.prevs

        for v in new_vol.prevs:
            v.next_vol.remove(accumulator)
            v.next_vol.append(new_vol)
        for v in new_vol.next_vol:
            v.prevs.remove(vol)
            v.prevs.add(new_vol)

        system.append(new_vol)
    else:
        # parallel resistors, only parallel part
        for v in vol:
            system.remove(v)
        assert accumulator == None
        if verbose:
            print(
                "parallelrem",
                [v.resistance for v in vol],
                "for",
                calculate_parallel_resistance([v.resistance for v in vol]),
            )

        resistance = calculate_parallel_resistance([v.resistance for v in vol])
        ident = ""
        for v in vol:
            ident += str(v.id)
        new_node = resistant_volume(
            radius=1,
            length=1,
            identifier=ident,
            lower_type="parallel",
            lower_volumes=vol,
            res=resistance,
        )
        new_node.path_indices = None
        # update connections of prevs or next ones
        prev_node = None
        next_node = None
        if (
            len(list(vol)[0].prevs) == 1
        ):  # remove from previous element and replace with merged element
            assert all([len(v.prevs) == 1 for v in vol])
            prev_node = list(list(vol)[0].prevs)[0]
            for node in vol:
                prev_node.next_vol.remove(node)
            prev_node.next_vol.append(new_node)
            # set the node that was previous as previous to new node

        if (
            len(list(vol)[0].next_vol) == 1
        ):  # remove from next element and replace with merged one
            assert all([len(v.next_vol) == 1 for v in vol])

            next_node = list(vol)[0].next_vol[0]
            # next_node.prevs={new_node}
            for node in vol:
                next_node.prevs.remove(node)
            next_node.prevs.add(new_node)
            # set next node for new node
        new_node.prevs = list(vol)[0].prevs
        new_node.next_vol = list(vol)[0].next_vol
        assert prev_node != None or next_node != None
        system.append(new_node)


def same_prev(volumes, verbose=False):
    prev = list(volumes)[0].prevs
    for v in volumes:
        if prev != v.prevs:
            if verbose:
                print("not samePREV", [v.resistance for v in volumes])
            return False
    if verbose:
        print("all same", [v.resistance for v in volumes])
    return True


def same_next(volumes, verbose=False):
    prev = list(volumes)[0].next_vol
    for v in volumes:
        if set(prev) != set(v.next_vol):
            if verbose:
                print("not same", [v.resistance for v in volumes])
            return False
    if verbose:
        print("all same", [v.resistance for v in volumes])
    return True


def share_prev_vol_group(volumes):
    # find largest volumegroup that shares prev
    group = dict((vol, [vol]) for vol in volumes)
    for vol in volumes:
        for vol2 in volumes:
            if vol2.prevs == vol.prevs:
                group[vol].append(vol2)
    maxval = max([len(g) for g in group.values()])
    if maxval > 1:
        for value in group.values():
            if len(value) == maxval:
                return value


def prev2s(volumes):
    previouses = set()
    for volume in volumes:
        previouses = previouses | volume.prevs
    return previouses


def next2s(volumes):
    previouses = set()
    for volume in volumes:
        previouses = previouses | set(volume.next_vol)
    return previouses


def simplify_network_serial(volumes, verbose=False):
    """N11N transform

    Args:
        volumes (_type_): _description_

    Returns:
        _type_: _description_
    """
    system = volumes.copy()
    for vol in system:
        vol.prevs = set()
    for vol in system:
        for v2 in vol.next_vol:
            v2.prevs.add(vol)
    # assert only one volume has no prevs all the time
    other_found = False
    for vol in system:
        if len(vol.prevs) == 0 and other_found:
            assert False, "before linear reduce startvolumes got split"
        if len(vol.prevs) == 0:
            other_found = True
    # vereinfache system, alle seriellenzusammenfassen bel- X - X -bel der naechste hat nur einen davor danach 3er
    prevlen = len(system) + 1
    while (len(system)) < prevlen:
        prevlen = len(system)
        if verbose:
            print(prevlen)
        select = None
        for vol in system:
            # if len(vol.prevs)!=1 or True:#starts a series
            if len(vol.next_vol) == 1 and len(vol.next_vol[0].prevs) == 1:
                if not hasattr(vol, "preset_q") and not hasattr(
                    vol.next_vol[0], "preset_q"
                ):
                    assert (
                        vol in vol.next_vol[0].prevs
                    )  # merge with next one if next one has no other prevs and no other connetion from vol
                    if verbose:
                        print("det")
                    select = vol.next_vol[0]
                    accu = vol
                    break
        if select != None:
            substitute_from_system(select, system, accu, verbose=verbose)
    return system


def simplify_network_parallel(volumes, verbose=True):
    """1N1 transform

    Args:
        volumes (_type_): _description_

    Returns:
        _type_: _description_
    """
    system = volumes.copy()
    for vol in system:
        vol.prevs = set()
    for vol in system:
        for v2 in vol.next_vol:
            v2.prevs.add(vol)
    # assert only one volume has no prevs all the time
    other_found = False
    for vol in system:
        if len(vol.prevs) == 0 and other_found:
            assert False, "before parallel reduce startvolumes got split"
        if len(vol.prevs) == 0:
            other_found = True
    # vereinfache system, alle seriellenzusammenfassen bel- X - X -bel der naechste hat nur einen davor danach 3er
    prevlen = len(system) + 1
    while (len(system)) < prevlen:
        prevlen = len(system)
        if verbose:
            print(prevlen)
        select = None
        for vol in system:
            # check 0n1
            # check 1n1
            # check xnx n to 1 only

            # serial before so check for next and prev 1
            if (
                len(vol.next_vol) == 1 and len(vol.prevs) == 1
            ):  # doesnt take any that have none before so shouldnt even need to check that presetq holds
                parallels = {vol}
                # gather all other parallels
                for other_vol in vol.next_vol[0].prevs:
                    if other_vol.prevs == vol.prevs and len(other_vol.next_vol) == 1:
                        parallels.add(other_vol)
                if len(parallels) > 1 and not any(
                    [hasattr(vol, "preset_q") for vol in parallels]
                ):
                    if verbose:
                        print("selected parallel:", [v.resistance for v in parallels])

                    select = parallels
                    break
        if select != None:
            substitute_from_system(list(select), system, verbose=verbose)
    return system


def simplify_network_delta(volumes, verbose=True):
    """delta to y transform

    Args:
        volumes (_type_): _description_

    Returns:
        _type_: _description_
    """
    # assert only one volume has no prevs all the time
    other_found = False
    for vol in volumes:
        if len(vol.prevs) == 0 and other_found:
            assert False, "before delta reduce startvolumes got split"
        if len(vol.prevs) == 0:
            other_found = True
    # vereinfache system, alle seriellenzusammenfassen bel- X - X -bel der naechste hat nur einen davor danach 3er
    reduced = True
    found_config = False
    while reduced:
        reduced = False
        # print("next_riun",[(v.name,[c.name for c in v.next_vol]) for v in volumes])
        for potential_terminal_node in volumes:
            if (
                not hasattr(potential_terminal_node, "preset_q")
                and len(potential_terminal_node.next_vol) >= 2
            ):  # can be higher than 2 then check all 2 combinations! and update terminal node for only them
                if len(potential_terminal_node.next_vol) != 2:
                    combinations = list(
                        itertools.combinations(potential_terminal_node.next_vol, 2)
                    )  # [('a', 'b'), ('a', 'c'), ('b', 'c')]
                    # eg a b c make ab ac bc
                    for a, b in combinations:
                        delta_config = check_delta_to_y_transform(
                            potential_terminal_node, ab=(a, b)
                        )

                        if delta_config != False:
                            assert delta_config[0] == True
                            ab, ac, bc = delta_config[1]
                            terminal_a = delta_config[2]
                            assert ab in volumes
                            assert ac in volumes
                            assert bc in volumes, len(volumes)
                            volumes = apply_delta_to_y_transform(
                                volumes, ab, ac, bc, terminal_a
                            )
                            print(
                                "applied Tranform delta to y",
                                [v.resistance for v in volumes],
                            )
                            reduced = True
                            found_config = True
                            break
                else:
                    delta_config = check_delta_to_y_transform(potential_terminal_node)

                    if delta_config != False:
                        assert delta_config[0] == True
                        ab, ac, bc = delta_config[1]
                        terminal_a = delta_config[2]
                        assert ab in volumes
                        assert ac in volumes
                        assert bc in volumes, len(volumes)
                        volumes = apply_delta_to_y_transform(
                            volumes, ab, ac, bc, terminal_a
                        )
                        print(
                            "applied Tranform delta to y"
                        )  # , [v.name for v in volumes])
                        reduced = True
                        found_config = True
                        break
    return volumes, found_config


def check_delta_to_y_transform(potential_terminal_node, ab=None, verbose=True):
    # find terminal node that has 2 following nodes AC and AB
    # one of those following nodes connects to another node C that connects to the follower of AB in case of AC and of AC from AB
    def c_nodes(AB, AC):
        c_nodes = []
        for vol in AB.next_vol:
            for latervol in vol.next_vol:
                if latervol in AC.next_vol:
                    c_nodes.append(vol)
        return c_nodes

    node_c = None
    if len(potential_terminal_node.next_vol) == 2:
        potential_AB = potential_terminal_node.next_vol[0]
        potential_AC = potential_terminal_node.next_vol[1]
    if ab != None:
        potential_AB = ab[0]
        potential_AC = ab[1]
    print(
        "checking ",
        potential_terminal_node.resistance,
        "other 2:   ",
        potential_AB.resistance,
        potential_AC.resistance,
    )

    if (
        (len(potential_AC.next_vol) == 1 and len(potential_AB.next_vol) == 1)
        or len(potential_AB.next_vol) == 0
        or len(potential_AC.next_vol) == 0
    ):
        if verbose:
            print("too short next")
        return False  # cant be delta config if not one of them goes to mid and to next, other transform
    else:
        # check if one has more than 2 following: then another transform should be solved first! (to keep checking simple)
        if (
            (False)
            or potential_AB in potential_AC.next_vol
            or potential_AC in potential_AB.next_vol
        ):  # last one checks for dead one
            if verbose:  # len(potential_AC.next_vol)>2 or len(potential_AB.next_vol)>2
                print("Det delta prior to delta")
            return False
    # check if one node leads to the other or both to each other:
    if len(potential_AC.next_vol) == 2 or True:
        if verbose:
            print("see with 2 later")
        for v in potential_AC.next_vol:
            if any([latervol in potential_AB.next_vol for latervol in v.next_vol]):
                node_c = v
    if len(potential_AB.next_vol) == 2 or True:
        for v in potential_AB.next_vol:
            # next vol is same or half of it
            if any([latervol in potential_AC.next_vol for latervol in v.next_vol]):
                if node_c != None:
                    if node_c != v:
                        print(
                            "Two ways but not the same middle volume!",
                            "middle vols from ab",
                            len(c_nodes(potential_AB, potential_AC)),
                            "middle vols from ac",
                            len(c_nodes(potential_AC, potential_AB)),
                            "middle vols shared ",
                            len(
                                set(c_nodes(potential_AC, potential_AB)).intersection(
                                    set(c_nodes(potential_AB, potential_AC))
                                )
                            ),
                        )
                        # colored_volumegraph(volumes,node_size_multiplier=4,highlight_vols=[potential_AB,potential_AC,c_nodes(potential_AB,potential_AC)])
                        return False
                node_c = v

    if node_c != None:
        assert len(node_c.prevs) == len(node_c.next_vol), (
            len(node_c.prevs),
            len(node_c.next_vol),
            node_c.resistance,
            potential_terminal_node.resistance,
            potential_AB.resistance,
            potential_AC.resistance,
        )
        # if verbose:print("found node bc as ", node_c.name)
        return (
            True,
            [v for v in [potential_AB, potential_AC, node_c]],
            potential_terminal_node,
        )
    return False


def apply_delta_to_y_transform(volumes, ab, ac, bc, terminal_resistor):
    old_config = [ab, ac, bc]
    # calculate ABC
    divsum = sum(v.resistance for v in old_config)
    # rem ac c ab
    for v in old_config:
        # print("remove ", v.name)
        volumes.remove(v)
    # add A b c
    ident = ""
    for v in old_config:
        ident += str(v.id) + "_"
    r_ab = ab.resistance
    r_ac = ac.resistance
    r_bc = bc.resistance
    resistance_a = (r_ab * r_ac) / divsum
    resistance_b = (r_ab * r_bc) / divsum
    resistance_c = (r_ac * r_bc) / divsum

    a = resistant_volume(
        radius=1,
        length=1,
        identifier=ident + "a",
        lower_type="delta",
        lower_volumes=old_config,
        res=resistance_a,
    )
    a.delta_resolved = False
    a.name = "a" + ident
    b = resistant_volume(
        radius=1,
        length=1,
        identifier=ident + "b",
        lower_type="delta",
        lower_volumes=old_config,
        res=resistance_b,
    )
    b.delta_resolved = False
    b.name = "b" + ident
    c = resistant_volume(
        radius=1,
        length=1,
        identifier=ident + "c",
        lower_type="delta",
        lower_volumes=old_config,
        res=resistance_c,
    )
    c.delta_resolved = False
    c.name = "c" + ident
    # link to following volumes
    a.next_vol = [b, c]
    bnext = ab.next_vol.copy()
    if bc in bnext:
        bnext.remove(bc)
    b.next_vol = bnext  # was hinter bc und ab war ohne bc
    cnext = ac.next_vol.copy()
    if bc in cnext:
        cnext.remove(bc)
    c.next_vol = cnext  # was hinter ac und bc war ohne bc

    # link to beginning and delete link to delta config beginning
    terminal_resistor.next_vol.remove(ab)
    terminal_resistor.next_vol.remove(
        ac
    )  # remove old and add new in case more than 2 where after it
    terminal_resistor.next_vol.append(a)
    new_config = [a, b, c]

    a.same_config = new_config
    b.same_config = new_config
    c.same_config = new_config

    # add to volumesystem
    volumes += new_config
    print("RETURNED NEW CONFIG Y INSTEAD OF DELTA")
    # print("ret",[v.name for v in volumes])
    return volumes


def check_dead_resistors(volumes):
    for volume in volumes:
        if len(volume.next_vol) > 1:
            for vol in volume.next_vol:
                for v in vol.next_vol:
                    if v in volume.next_vol:
                        return (
                            True,
                            vol.name,
                        )  # is either really dead (one out one in) or in delta dead with Y transform just having 0 resistors -> then delete node and give next to previous


def calculate_simplified_network(
    volumes,
    verbose=False,
    visualize=False,
    font_size=100,
    node_size_multiplier=1,
    arrow_size=50,
    return_images=False,
    print_attr_only=False,
    print_attr=False,
    restore=True,
    restype=True,
):
    if restype:
        font_size = 10
        arrow_size = 40
        node_size_multiplier = 1
    connections_save = save_connections(volumes)

    # assumes that all are in a network, thus a vol 0 gets created if necessary
    start_vols = [
        vol for vol in volumes if no_refs(vol, volumes) and not hasattr(vol, "preset_q")
    ]  # preset for veins
    additional_start = []
    if len(start_vols) > 1 or True:
        print("addstarter")
        r0_vol = form_vol(
            res=0.00000000000000000000001, p2=None, following=start_vols, p1=None
        )
        r0_vol.name = "ADDEDSTART"
        additional_start = [r0_vol]

    end_vols = [
        vol for vol in volumes if vol.next_vol == [] and not hasattr(vol, "preset_q")
    ]
    new_end = form_vol(res=0, p2=None, following=[], p1=None)
    for vol in end_vols:
        vol.next_vol = [new_end]
    # add start and endterminals
    system = volumes + additional_start + [new_end]

    assert len(system) == len(set(system)), "No duplicate elements allowed"
    prevlen = len(system) + 1
    if verbose:
        print(
            [(vol.resistance, [v.resistance for v in vol.next_vol]) for vol in (system)]
        )
    for vol in system:
        for vol2 in vol.next_vol:
            assert (
                vol2 in system
            ), f"Start has nextvol not in sys {vol2.vessel.associated_vesselname} from {vol.vessel.associated_vesselname}"
    # set previous for all volumes
    set_prevs(system)
    images = []
    # crreate first image if wanted
    if return_images:
        images.append(
            colored_volumegraph(
                system,
                arrowsize=arrow_size,
                print_attr=print_attr,
                print_attr_only=print_attr_only,
                ret_image=True,
                node_size_multiplier=node_size_multiplier,
                font_size=font_size,
                restype=restype,
            )
        )

    while len(system) < prevlen:
        while len(system) < prevlen:
            prevlen = len(system)

            if verbose:
                print("--------------SER--------------")
            system = simplify_network_serial(system, verbose=verbose)
            if return_images:
                images.append(
                    colored_volumegraph(
                        system,
                        arrowsize=arrow_size,
                        print_attr=print_attr,
                        print_attr_only=print_attr_only,
                        ret_image=True,
                        node_size_multiplier=node_size_multiplier,
                        font_size=font_size,
                        restype=restype,
                    )
                )

            for vol in system:
                for vol2 in vol.next_vol:
                    assert vol2 in system, "serial has nextvol not in sys"
            if verbose:
                print(
                    [
                        (vol.resistance, [v.resistance for v in vol.next_vol])
                        for vol in (system)
                    ]
                )
            if verbose:
                print("----------------------------")

            if verbose:
                print("--------------PAR--------------")
            system = simplify_network_parallel(system, verbose=verbose)
            if return_images:
                images.append(
                    colored_volumegraph(
                        system,
                        arrowsize=arrow_size,
                        print_attr=print_attr,
                        print_attr_only=print_attr_only,
                        ret_image=True,
                        node_size_multiplier=node_size_multiplier,
                        font_size=font_size,
                        restype=restype,
                    )
                )

            for vol in system:
                for vol2 in vol.next_vol:
                    assert vol2 in system, "parallel has nextvol not in sys"
            if verbose:
                print(
                    [
                        (vol.resistance, [v.resistance for v in vol.next_vol])
                        for vol in (system)
                    ]
                )
            if verbose:
                print("----------------------------")

        # try delta transform if length didnt go down
        found_config = False
        if verbose:
            print("--------------DELTA--------------")
        system, found_config = simplify_network_delta(system)
        if return_images:
            images.append(
                colored_volumegraph(
                    system,
                    arrowsize=arrow_size,
                    print_attr=print_attr,
                    print_attr_only=print_attr_only,
                    ret_image=True,
                    node_size_multiplier=node_size_multiplier,
                    font_size=font_size,
                    restype=restype,
                )
            )

        for vol in system:
            for vol2 in vol.next_vol:
                assert vol2 in system, "delta has nextvol not in sys"
        if verbose:
            print(
                [
                    (vol.resistance, [v.resistance for v in vol.next_vol])
                    for vol in (system)
                ]
            )
        if verbose:
            print("----------------------------")
        prevlen = len(system)
        if found_config:
            prevlen += 1  # try reducin the new form
    for vol in system:
        for vol2 in vol.next_vol:
            assert vol2 in system, "END has nextvol not in sys"

    if visualize == True:
        colored_volumegraph(system, print_attr="resistance", arrowsize=70)
    if restore:
        restore_connections(connections_save)
    for vol in system:
        for vol2 in vol.next_vol:
            if not vol2 in system:
                print("END has nextvol not in sys after restoration")
    if return_images:
        return system, images
    return system


def simple_resistance(volume, scope=[]):
    follow = following(volume, scope)
    if volume in scope or scope == []:
        print("QADDED")
        follow += [volume]
    volumes = list(set(follow))
    print(
        "vols", len(volumes)
    )  # , [(v.q,len(v.next_vol)) for v in volumes if type(v)!=form_vol])
    net = calculate_simplified_network(volumes)
    print(
        "vols", len(volumes)
    )  # , [(v.q,len(v.next_vol)) for v in volumes if type(v)!=form_vol])

    if len(net) == 1:
        return net[0].resistance
    else:
        print("No simple resistance calculatable")


def get_all_nextshare_volumes(volume_selection, all_volumes):
    """get_all_nextshare_volumes return all volumes, that have one or more same elements in next"""
    combined_next = set()
    for v in volume_selection:
        for n in v.next_vol:
            combined_next.add(n)
    comb_volumes = set()
    for v in all_volumes:
        if any([e in combined_next for e in v.next_vol]):
            comb_volumes.add(v)
    return list(comb_volumes)


def get_all_prevshare_volumes(volume_selection, all_volumes):
    """get_all_nextshare_volumes return all volumes, that have one or more same elements in next"""
    set_prevs(all_volumes)
    combined_prevs = set()
    for v in volume_selection:
        for n in v.prevs:
            combined_prevs.add(n)
    comb_volumes = set()
    for v in all_volumes:
        if any([e in combined_prevs for e in v.prevs]):
            comb_volumes.add(v)
    return list(comb_volumes)


# connect volumes and keep graph planar:
def following(vol, scope=[]):
    if scope == []:
        return flatten([vol.next_vol + [following(v) for v in vol.next_vol]])
    else:
        return flatten(
            [
                [v for v in vol.next_vol if v in scope]
                + [following(v, scope) for v in vol.next_vol if v in scope]
            ]
        )


def previous_volumes(vol):
    return flatten([list(vol.prevs) + previous_volumes(v) for v in vol.prevs])


def get_common_node_following(volumes):
    lists = [following(vol) for vol in volumes]
    for v, lst in enumerate(lists):
        for v2, lst2 in enumerate(lists):
            if v != v2:
                vol = volumes[v]
                vol2 = volumes[v2]

                common_elements = set(lst2)
                common_elements &= set(lst)
                if len(common_elements) == 0:
                    continue  # no common elements

                # check if all volumes have that common node in their list
                common_elements = [
                    e for e in lst if e in common_elements
                ]  # keep order of prevs!
                for element in common_elements:
                    print("checking", element.resistance)
                    first_common_node = element
                    if all([first_common_node in ls for ls in lists]):
                        print("first common node ", first_common_node.resistance)
                        return first_common_node


def get_common_node_previous(volumes):
    lists = [previous_volumes(vol) for vol in volumes]
    for v, lst in enumerate(lists):
        for v2, lst2 in enumerate(lists):
            if v != v2:
                vol = volumes[v]
                vol2 = volumes[v2]

                common_elements = set(lst2)
                common_elements &= set(lst)
                if len(common_elements) == 0:
                    continue  # no common elements

                # check if all volumes have that common node in their list
                common_elements = [
                    e for e in lst if e in common_elements
                ]  # keep order of prevs!
                for element in common_elements:
                    print("checking", element.resistance)
                    first_common_node = element
                    if all([first_common_node in ls for ls in lists]):
                        print("first common node", first_common_node.resistance)
                        return first_common_node


def get_safe_connection_volumes(volumes_to_connect, volumes):
    # returns a group of volumes thats save to connect, this means that they share the first common node in following of volumes to connect if they are startnodes and
    # the first previous ellement if all are endvolumes
    if len(volumes_to_connect) == 1:
        return volumes_to_connect  # one connect is always safe
    set_prevs(volumes)
    if all([len(volume.next_vol) == 0 for volume in volumes_to_connect]):
        # connect arteries
        print("selection for arteries")
        first_common_node = get_common_node_previous(volumes_to_connect)
        assert first_common_node != None, "volumesection is a separate strand?"
        addition = []
        for vol in volumes:
            if first_common_node in previous_volumes(vol) and len(vol.next_vol) == 0:
                addition.append(vol)
        assert all([v in addition for v in volumes_to_connect])
        return addition

    elif all([len(volume.prevs) == 0 for volume in volumes_to_connect]):
        print("selection for veins")

        # connect veins
        first_common_node = get_common_node_following(volumes_to_connect)
        assert first_common_node != None, "volumesection is a separate strand?"
        addition = []
        for vol in volumes:
            if first_common_node in following(vol) and len(vol.prevs) == 0:
                addition.append(vol)
        assert all([v in addition for v in volumes_to_connect])
        return addition
    else:
        assert False, "No safe group for mixed volumetypes implemented"


def connect_arteries(artery_vol_selection, volumes):
    new_node = form_vol(
        res=0.00000000000000000000000000001, p2=None, following=[], p1=None
    )
    selection = get_safe_connection_volumes(artery_vol_selection, volumes)
    for v in selection:
        assert v in volumes
        v.next_vol = [new_node]
    volumes.append(new_node)
    return new_node


def connect_veins(vein_vol_selection, volumes):
    new_node = form_vol(
        res=0.00000000000000000000000000001, p2=None, following=[], p1=None
    )
    selection = get_safe_connection_volumes(vein_vol_selection, volumes)
    new_node.next_vol = selection
    volumes.append(new_node)
    return new_node


def symvals_to_attributes(volumes):
    for vol in volumes:
        for var in vol.variables:
            if "Q" in str(var):
                vol.q = vol.variables[var]
            if "P1" in str(var):
                vol.p1 = vol.variables[var]
            if "P2" in str(var):
                vol.p2 = vol.variables[var]
