# methods that are related to vesselvolumes, for example calculating resistances or flowrates for vessels
import itertools
import CeFloPS.simulation.settings as settings
import CeFloPS.simulation.common.functions as cf
import CeFloPS.simulation.common.unit_conv as unit_conv
from sympy import Eq, Symbol, symbols, Float, zoo, oo, nan, simplify, solve
from scipy.integrate import quad
import sympy

VISCOSITY = 0.0035

PI = 3.14159265359


# pyhsics formulas
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
    profile = lambda r: (1 - ((r**2) / (radius**2))) * v_max(
        radius, length, p1, p2
    )  # noqa: E731
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
    v_max = (1 / (4 * VISCOSITY)) * radius**2 * ((p1 - p2) / length)
    assert v_max > 0
    return v_max


def probability_profile(profile, radius):
    """Generate a probability profile for a given velocity profile and the volumes radius as the integral of its volumeflow function TODO

    Args:
        profile (_type_): _description_

    Returns:
        _type_: _description_
    """
    # integrate.quad(lambda r: q_over_r(r) / total, 0, r)[0]
    """ total = quad(lambda r: profile(r) * abs(r) * 2 * PI, 0, radius)[0]  # q over r

    pprofile = lambda x: quad(lambda r: (profile(r) * abs(r) * 2 * PI) / total, 0, x)[
        0
    ]  # noqa: E731 """

    pprofile = lambda x: x**2 / radius**2
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


# creation


class resistant_volume:
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
        if res is None:
            self.resistance = (8 * VISCOSITY * self.length) / (
                PI * self.radius**4
            )  # Pa = N/m^2
        else:
            self.resistance = res
        if identifier is not None:
            assert (
                type(identifier) is not int
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
        if other_vol is None:
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
                    variable is not None
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
                    if type(vol) is resistant_volume:
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

                if type(left_side) is resistant_volume:
                    left_side.set_val_for_symbol(left_side.P_1, p1)
                    left_side.set_val_for_symbol(
                        left_side.P_2, calc_p2(q=q, p1=p1, r=left_side.resistance)
                    )
                    left_side.set_val_for_symbol(left_side.Q_1, q)
                if type(right_side) is resistant_volume:
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
                if self.delta_resolved is False:
                    # check if other parts are already solved, if not then let other one solve q
                    solvable = True
                    for vol in self.same_config:
                        if hasattr(vol, "q") and vol.q is not None:
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
            if type(vol) is resistant_volume:
                vol.solve_lower_volumes()

    def update_eq_q(self):
        self.variables_searched = []
        # Equation for Q:
        self.eq_q = Eq(self.Q_1, (self.P_1 - self.P_2) / self.W)
        for key, value in self.variables.items():
            if value is not None:
                self.eq_q = self.eq_q.subs(key, value)
                if type(value) is Symbol:
                    self.variables_searched.append(value)
                elif type(value) is float:
                    self.variables[key] = Float(value)
            else:
                self.variables_searched.append(key)

    def get_symval(self, key):
        out = self.variables[key]
        if out is None:
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


def apply_connections(volumes):
    deg = False
    """volumes=[]
    for vessel in vessels:
        volumes+=vessel.volumes"""
    fully_connected = lambda x: len(x.to_append) == 0  # noqa: E731
    while not all([fully_connected(volume) for volume in volumes]):
        print(
            [
                [v.associated_vesselname[-30::] for v, t in volume.to_append]
                for volume in volumes
                if fully_connected(volume) is False
            ]
        )

        print(
            len(
                [
                    fully_connected(volume)
                    for volume in volumes
                    if fully_connected(volume) is True
                ]
            ),
            len(
                [
                    fully_connected(volume)
                    for volume in volumes
                    if fully_connected(volume) is False
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
                    if fully_connected(volume) is False
                ]
            )
            == 10
        ):
            deg = True
        """ for vessel in vessels:
            if "jump" not in vessel.associated_vesselname: """
        for volume in volumes:
            if not fully_connected(volume):
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


def get_volume_data(vessel, lower_pathindex, higher_index):
    # get vesseldata for volume
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
    except:  # noqa: E722
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


def get_traversable_links(new_vessel):
    """Get traversable links for a vessel, based on if its classified as artery or vein

    Args:
        new_vessel (Vessel): Vesselobject which has an amount of associated links

    Returns:
        list of Link: traversable links of vessel
    """
    if new_vessel.type == "vein":
        try:
            if new_vessel.highest_link() is not None:
                return [new_vessel.highest_link()]
            return []
        except:  # noqa: E722
            print("vein has no links or wrong direction")
            return []
    else:
        return new_vessel.next_links(0)


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


def split_volumes_for_links(vessels):
    # create split volumes for each linking volume from antoher vessel if necessary
    for vessel in vessels:
        # if "jump" not in vessel.associated_vesselname:
        for link in get_traversable_links(vessel):
            # if "jump" not in link.target_vessel.associated_vesselname:
            to_rem = None
            target = link.target_index
            tvessel = link.target_vessel
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

                    assert higher_vol is not None, (
                        t_vol.path_indices,
                        [vol2.path_indices for vol2 in tvessel.volumes],
                        target,
                        len(tvessel.path),
                        lower_vol,
                        higher_vol,
                    )
                    assert lower_vol is not None, "lowa!"
                    break
            if to_rem is not None:
                tvessel.volumes.remove(t_vol)
                tvessel.volumes.append(lower_vol)
                tvessel.volumes.append(higher_vol)
            if target == 0:
                assert any([vo.path_indices[1] == 0 for vo in tvessel.volumes])
                assert found is True or any(
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
        if "jump_vessel" not in vessel.associated_vesselname:
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
        assert v.vessel is not None
        v.next_vol = []
        v.vessel.volumes.append(v)
        v.vessel.volumes = list(set(v.vessel.volumes))


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

            assert found_source is True
            assert (
                found_target is True
                or "jump" in link.target_vessel.associated_vesselname
            ), (
                link.source_vessel.associated_vesselname,
                link.target_index,
                [vol.path_indices for vol in link.target_vessel.volumes],
                len(link.target_vessel.path),
                str(link),
            )


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


def generate_volumes(
    vessels,
):
    for vessel in vessels:
        if hasattr(vessel, "volumes"):
            for vol in vessel.volumes:
                assert vol is not None
    create_heart_volumes(vessels)
    # print(sum([len(vessel.volumes) for vessel in vessels]))
    create_volumes(vessels)
    for vessel in vessels:
        if hasattr(vessel, "volumes"):
            for vol in vessel.volumes:
                assert vol is not None
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


# ----------------------------------------------------------------
# --------------------Flowrate calculation------------------------
# ----------------------------------------------------------------


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


def flatten(items, seqtypes=(list, tuple)):
    for i, x in enumerate(items):
        while i < len(items) and isinstance(items[i], seqtypes):
            items[i : i + 1] = items[i]
    return items


def set_prevs(volumes):
    for vol in volumes:
        vol.prevs = set()
    for vol in volumes:
        for v2 in vol.next_vol:
            v2.prevs.add(vol)


def no_refs(volume, volumes):
    for v in volumes:
        if volume in v.next_vol:
            return False
    return True


class Object(object):
    pass


class form_vol(resistant_volume):
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
    from visualization.vesselvol_vis import colored_volumegraph

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


# -----------
# nodepotential
