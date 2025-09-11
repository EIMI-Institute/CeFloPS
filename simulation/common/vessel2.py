from .functions import calculate_distance_euclidian
import CeFloPS.simulation.settings as settings
import numpy as np
import sympy
from CeFloPS.data_processing.volume_fun import (
    v_max,
    velocity_profile,
    probability_profile,
)

# from .volumes import create_profiles
import scipy.integrate as integrate
import random
from numba import jit
from CeFloPS.logger_config import setup_logger
from CeFloPS.data_processing.geometric_fun import closest_point_on_circle, random_point_on_circle

import logging
logger = setup_logger(__name__, level=logging.WARNING)

class Vessel:
    """class for representing route information in a bloodvessel

    Stores diameterinformation, path, cuts, and distances to traverse the vessel / simulate a cell passing through

    Attributes:
        path: A list of all points that make up a path through the vessel
        diameter_a: A numpy.float64 that represents avg diameter in first 10 percent of points in the path
        diameter_b:  A numpy.float64 that represents avg diameter in last 10 percent of points in the path
        avg_diameter:  A numpy.float64 that represents avg diameter at points of the path
        distances: A list of pairwise distances from pathpoint to next pathpoint, starts at 0
        associated_vesselname: A String that holds the value of the associated mesh the vessel belongs to
        links_to_path: A list of links in format [pathindex, vesselobject], showing how the vessels are interconnected
        tags: A list of tags at vertexindices format: [(index, "blahtag"),(...,...),...]
        cuts: A list of cuts that the pathpoints were created from
        volume: A numpy.float64 that represents the vessels volume as given by trimesh.volume. It is assumed, that the volume was generated from a watertight mesh
        length: A float that represents the total distance from path[0] to the end of path
    """

    def __init__(
        self,
        path,
        diameter_a,
        diameter_b,
        avg_diameter,
        diameters,
        associated_vesselname,
        speed_function,
    ):
        """Inits Vessel with path, diameter on end a (path[0]), diameter on end b (path[len(path)-1]) and associated_vesselname."""
        self.speed_function = speed_function
        self.path = path
        self.fully_profiled = False
        self.diameter_a = diameter_a
        self.diameter_b = diameter_b
        self.avg_diameter = avg_diameter
        self.volumes = []
        self.diameters = diameters
        self.associated_vesselname = associated_vesselname
        self.node_tags=["B" for _ in range(len(self.path))]
        self.length = sum(
            [
                calculate_distance_euclidian(self.path[i], self.path[i + 1])
                for i in range(len(self.path) - 1)
            ]
        )
        self.links_to_path = []
        self.tags = []
        self.cuts = None
        self.type = "artery"
        self.reachable_rois = []  # [[lower_index, roi], ...]
        if settings.VESSEL_VEINLIKE in self.associated_vesselname:
            self.type = "vein"
        self.times = self.calculate_times()
        self.distances = [
            calculate_distance_euclidian(self.path[i], self.path[i + 1])
            for i in range(len(self.path) - 1)
        ]
        self.id = f"{id(self)}_{self.associated_vesselname[0:15]}_{len(self.path)}"

        self.profiles = dict()
        self.links_to_vois = []

    def check_fully_profiled(self):
        if self.has_speed_profile(0, len(self.path) - 1) and len(self.volumes) > 0:
            self.fully_profiled = True
        else:
            self.fully_profiled = False

    def unregister_functions(self):
        for volume in self.volumes:
            self.profiles[(volume.id, "v_max")] = None
            self.profiles[(volume.id, "v_r")] = None
            self.profiles[(volume.id, "p_r")] = None

    def register_volumes(self, force=False, verbose=False):
        """register_volumes Create speedprofiles for all volumes of this vessel"""
        if not hasattr(self, "pre_cached_profiles") or force:
            self.pre_cached_profiles = False
        self.volumes = sorted(list(set(self.volumes)), key=lambda x: x.path_indices)
        if not hasattr(self, "profiles"):
            self.profiles = dict()
        # convert to float if necessary
        for volume in self.volumes:
            if type(volume.get_symval(volume.Q_1)) == sympy.Symbol:
                if (
                    hasattr(volume, "q")
                    and hasattr(volume, "p1")
                    and hasattr(volume, "p2")
                ):
                    volume.set_val_for_symbol(volume.Q_1, float(volume.q))
                    volume.set_val_for_symbol(volume.P_1, float(volume.p1))
                    volume.set_val_for_symbol(volume.P_2, float(volume.p2))

            # create lambda function profiles
            self.profiles[(volume.id, "v_max")] = v_max(
                float(volume.radius),
                float(volume.length),
                float(volume.get_symval(volume.P_1)),
                float(volume.get_symval(volume.P_2)),
            )
            self.profiles[(volume.id, "v_r")] = velocity_profile(
                float(volume.radius),
                float(volume.length),
                float(volume.get_symval(volume.P_1)),
                float(volume.get_symval(volume.P_2)),
            )

            self.profiles[(volume.id, "p_r")] = probability_profile(
                self.profiles[(volume.id, "v_r")], volume.radius
            )

        # if not pre cached create functions and value arrays
        if not self.pre_cached_profiles:
            # logger.print("Forced")
            self.distances = [
                calculate_distance_euclidian(self.path[i], self.path[i + 1])
                for i in range(len(self.path) - 1)
            ]
            self.times = self.calculate_times()
            PI = 3.14159265359

            for volume in self.volumes:
                # set floatconversion if not previously done
                if (
                    volume.get_symval(volume.Q_1) != None
                    and type(volume.get_symval(volume.Q_1)) != sympy.Symbol
                ):
                    if verbose:
                        logger.print(volume.get_symval(volume.Q_1))
                    volume.set_val_for_symbol(
                        volume.Q_1, float(volume.get_symval(volume.Q_1))
                    )
                    if volume.get_symval(volume.W) != None:
                        volume.set_val_for_symbol(
                            volume.W, float(volume.get_symval(volume.W))
                        )
                    if volume.get_symval(volume.P_1) != None:
                        volume.set_val_for_symbol(
                            volume.P_1, float(volume.get_symval(volume.P_1))
                        )
                    if volume.get_symval(volume.P_2) != None:
                        volume.set_val_for_symbol(
                            volume.P_2, float(volume.get_symval(volume.P_2))
                        )

                assert (
                    type(volume.radius) == np.float64 or type(volume.radius) == float
                ), (
                    type(volume.radius),
                    volume.radius,
                    volume.vessel.associated_vesselname,
                )
                try:
                    float(volume.get_symval(volume.P_1))
                    float(volume.get_symval(volume.P_2))
                except:
                    assert False, (
                        volume.get_symval(volume.P_1),
                        volume.get_symval(volume.P_2),
                    )

                resolution = 100
                self.profiles[(volume.id, "rs")] = [
                    float(volume.radius) * i / resolution
                    for i in range(
                        1, int(resolution)
                    )  # TODO instead of 0 to +1 because infinite values are undesired for times
                ]
                x = self.profiles[(volume.id, "rs")]
                p_for_v = self.profiles[(volume.id, "p_r")]
                self.profiles[(volume.id, "rs_prob")] = [
                    p_for_v(i) for i in x
                ]  # put to reg

        else:
            # logger.print("load prof")
            for volume in self.volumes:
                assert (volume.id, "rs") in self.profiles, set(
                    [k[1] for k in self.profiles.keys()]
                )
                assert (volume.id, "rs_prob") in self.profiles

        self.pre_cached_profiles = True  # rs and rsprob
        self.check_fully_profiled()

    def get_profile(self, index):
        assert index >= 0 and index < len(self.path), "index out of range"
        selected_volume = None
        for volume in self.volumes:
            if volume.path_indices[0] <= index and volume.path_indices[1] >= index:
                selected_volume = volume
                break
        if selected_volume == None:
            # logger.print("No volume for index found", index, self.associated_vesselname[-30::])
            return None
        # logger.print(selected_volume.variables)
        """ if (selected_volume, "p_r") not in self.profiles or (selected_volume, "v_r") not in self.profiles:
            #print(self.associated_vesselname)
            return None """
        return (
            self.profiles[(selected_volume.id, "p_r")],
            self.profiles[(selected_volume.id, "v_r")],
        )

    def get_volume_by_index(self, index):
        for volume in self.volumes:
            if (
                volume.path_indices[0] <= index
                and volume.path_indices[1] > index
                or volume.path_indices[1] == len(self.path) - 1
            ):
                return volume

    def calculate_times(self, no_vol_speed=False):
        """Calculate timearray for distances between pathpoints using a speed function of this vessel

        Args:
            path (_type_): _description_
            speed_function (_type_): _description_

        Returns:
            list: time passed between points"""

        if len(self.volumes) > 0:
            if (
                all(
                    [
                        type(vol.get_symval(vol.Q_1)) != sympy.Symbol
                        for vol in self.volumes
                    ]
                )
                and no_vol_speed == False
            ):
                return [
                    self.speed_function(
                        calculate_distance_euclidian(self.path[i], self.path[i + 1]),
                        list_of_vesselobjects=[],
                        volume=self.get_volume_by_index(i),
                    )
                    for i in range(len(self.path) - 1)
                ]
        return [
            self.speed_function(
                calculate_distance_euclidian(self.path[i], self.path[i + 1]), [self]
            )
            for i in range(len(self.path) - 1)
        ]

    def relink(self, other_link):
        for link in self.links_to_path:
            if (
                link.target_vessel == other_link.source_vessel
                and link.source_index == other_link.target_index
                and link.target_index == other_link.source_index
                and other_link.target_vessel == self
            ):
                return link

    def reverse(self, vessels=[], static_links=False):
        """Reverse the path and timelist for this vessel effectively changing its traversal direction"""
        # Reverse the main vessel properties
        self.path = list(self.path)
        self.times = list(self.times)
        self.diameters = list(self.diameters)
        self.path.reverse()
        self.times.reverse()
        self.diameters.reverse()

        self.diameter_start = self.path[0]

        if not static_links:
            # Adjust indices for links directly attached to this vessel
            for link in self.links_to_path:
                """rl = link.target_vessel.relink(link)
                if rl is not None:
                    rl.target_index = len(self.path) - 1 - link.source_index"""
                link.source_index = len(self.path) - 1 - link.source_index

        # Update links for other vessels if needed
        if not vessels:
            vessels = list(set(link.target_vessel for link in self.links_to_path))

        for vessel in vessels:
            for l in vessel.links_to_path:
                if l.target_vessel == self:
                    l.target_index = len(self.path) - 1 - l.target_index

        # Deduplicate and sort links by `source_index`
        self.links_to_path = sorted(
            set(self.links_to_path), key=lambda x: x.source_index
        )

    def get_rois(self, index):
        return [roi[1] for roi in self.reachable_rois if roi[0] >= index]

    def add_link(self, link):
        """Add a link, all links are sorted in order of their source index ASC

        Args:
            link (_type_): Link to add
        """

        for olink in self.links_to_path:
            if (
                olink.target_vessel == link.target_vessel
                and olink.target_index == link.target_index
                and olink.source_index == link.source_index
            ):
                return
        self.links_to_path.append(link)
        self.links_to_path = sorted(
            list(set(self.links_to_path)), key=lambda x: x.source_index
        )

    def no_higher_links(self, index, ignore_tlinks=True):
        """Check for link existence with source_index higher than index

        Args:
            index (_type_): lower bound for link.source_index

        Returns:
            boolean: true, if no links higher than index exist, false else
        """
        return len(self.next_links(index, ignore_tlinks)) == 0

    def next_links(self, index, ignore_tlinks=True):
        """Check for links with source_index higher than index

        Args:
            index (_type_): lower bound for link.source_index

        Returns:
            list: links with higher source_index
        """
        if not ignore_tlinks:
            return [
                link
                for link in self.links_to_path + self.links_to_vois
                if link.source_index >= index
            ]
        return [link for link in self.links_to_path if link.source_index >= index]

    def highest_link(self):
        """get highest link associated with vessel

        Returns:
            Link: highest link
        """
        endlinks = [
            link
            for link in self.links_to_path
            if link.source_index == len(self.path) - 1
        ]
        if len(endlinks) > 0:
            return endlinks[len(endlinks) - 1]

    def set_speed_fun(self, speed_fun):
        """Set speed function for vessel used to calculate passed time

        Args:
            speed_fun (_type_): speed fucntion to set
        """
        self.speed_function = speed_fun

    def delete(self, vessels):
        """delete Removes self from vessel-data structure and deletes all links to self"""
        for vessel in vessels:
            to_rem = []
            for link in vessel.links_to_path:
                if link.target_vessel == self:
                    to_rem.append(link)
            for link in to_rem:
                vessel.links_to_path.remove(link)
        vessels.remove(self)

    def neighbouring_vessels(self, restrictor=None):
        """neighbouring_vessels Returns a list of vessels that are linked to from this vessel"""
        if restrictor is None:
            restrictor = lambda vessel: True
        return [
            link.target_vessel
            for link in self.links_to_path
            if restrictor(link.target_vessel)
        ]

    def parallel_part(self):
        """parallel_part returns all links to multi othervessels. These are those that have 2 connections between self and same vessel

        Returns:
            dict: Key is targetvessel, values links to vessel
        """
        parallel = dict()
        targets = [l.target_vessel for l in self.links_to_path]
        if len(targets) > len(set(targets)):
            # get doubled part
            doubled_targets = [
                target for target in targets if count(targets, target) > 1
            ]
            doubled_targets = set(doubled_targets)

            for target in doubled_targets:
                links_to_double = [
                    link for link in self.links_to_path if link.target_vessel == target
                ]

                parallel[target] = links_to_double
            for key in parallel.keys():
                if not (len(parallel[key]) < 3):
                    logger.print("More than one parallel section found for target")
        return parallel

    def strip_from_path(self, lower_index, upper_index, vessels):
        assert not (
            lower_index == 0 and upper_index == len(self.path) - 1
        ), self.associated_vesselname
        offset = upper_index - lower_index
        assert offset > -1
        assert lower_index == 0 or upper_index == len(self.path) - 1
        # remove from path and remove links from part
        if lower_index != 0:
            self.path = self.path[0 : lower_index + 1]
            self.diameters = self.diameters[0 : lower_index + 1]
            self.times = self.times[0:lower_index]
            logger.print("lower HIGH")
        else:
            self.path = self.path[upper_index::]
            self.diameters = self.diameters[upper_index::]
            self.times = self.times[upper_index::]
            logger.print("lower LOW")
        rem = []
        self.diameter_start = self.path[0]
        # exclude outer index that connects to rest
        if lower_index == 0:
            upper_index = upper_index - 1
        else:
            lower_index = lower_index + 1
        for link in self.links_to_path:
            if link.source_index <= upper_index and link.source_index >= lower_index:
                rem.append(link)
        for l in rem:
            self.links_to_path.remove(l)
        # remove links to part
        for vessel in vessels:
            to_rem = []
            for link in vessel.links_to_path:
                if (
                    link.target_vessel == self
                    and link.target_index <= upper_index
                    and link.target_index >= lower_index
                ):
                    to_rem.append(link)
            for link in to_rem:
                vessel.links_to_path.remove(link)

        # set indices in source and target to correct new values

        logger.print("offset", offset)
        if lower_index == 0:
            for i, link in enumerate(self.links_to_path):
                self.links_to_path[i].source_index -= offset
            for vessel in vessels:
                for i, link in enumerate(vessel.links_to_path):
                    if link.target_vessel == self:
                        vessel.links_to_path[i].target_index -= offset

    def has_speed_profile(self, lower, higher):
        if hasattr(self, "fully_profiled"):
            if self.fully_profiled:
                return True
        try:
            not_exists_associated_profile = any(
                [self.get_profile(i) == None for i in range(lower, higher)]
            )
        except:
            return False  # in case there are no profiles
        if not_exists_associated_profile:
            return False
        return True

    def get_rs(self, volume_id, number, prev_r=-1):
        try:
            x = self.profiles[(volume_id, "rs")]
        except:
            logger.print("fully profiled", self.fully_profiled)
            logger.print("vols", len(self.volumes))
            x = self.profiles[(volume_id, "rs")]

        if (volume_id, "rs_prob") not in self.profiles:
            p_for_v = self.profiles[(volume_id, "p_r")]
            self.profiles[(volume_id, "rs_prob")] = [
                p_for_v(i) for i in x
            ]  # put to reg

        prev_bias = settings.R_BIAS
        rs = []
        for _ in range(number):
            r = random.choices(
                x, cum_weights=(self.profiles[(volume_id, "rs_prob")]), k=1
            )[0]

            # logger.print(r )
            if prev_r != -1:
                r = r * (1 - prev_bias) + prev_r * prev_bias

            rs.append(r)
        # assert len(rs) > 0, (number, higher, low)
        return rs, prev_r

    def get_times(
        self, lower, higher, no_vol_speed=False, fastest_speed=False, db=False
    ):
        """get_times Method that returns the time it takes to traverse the vesselpath from lower to higher based on speed and volume profiles of its volumeparts

        Args:
            lower (_type_): lower pointindex
            higher (_type_): higher pointindex

        Returns:
            _type_: Array with times
        """
        assert lower <= higher
        if lower == higher:
            return []
        assert higher < len(self.path)

        if (
            self.has_speed_profile(lower, higher)
            and no_vol_speed == False
            and len(self.volumes) > 0
        ):  # excludes point at higher
            times = []
            # get vollumes for pathparts
            # vols = []
            prev_bound_lower = lower
            prev_r = -1
            for volume in self.volumes:
                if volume.path_indices[1] > lower and volume.path_indices[0] < higher:
                    # vols.append(v)

                    # for volume in vols:
                    low = volume.path_indices[0]
                    high = volume.path_indices[1]
                    if volume.path_indices[0] <= lower:
                        low = lower
                    if volume.path_indices[1] > higher:
                        high = higher

                    if prev_r > volume.radius:
                        prev_r = -1  # reset r when outer r
                    rs, prev_r = self.get_rs(volume.id, high - low, prev_r)
                    distances = self.distances[low:high]
                    # assert len(distances) == len(rs)
                    if fastest_speed:
                        logger.print("ft")
                        times += [
                            distance / (self.profiles[(volume.id, "v_max")] * 1000)
                            for j, distance in enumerate(distances)
                        ]
                    else:
                        times += [
                            distance / (self.profiles[(volume.id, "v_r")](rs[j]) * 1000)
                            for j, distance in enumerate(distances)
                        ]

            return times
        else:
            if db:
                logger.print("timesret")
            return self.times[lower:higher]

    def get_avg_t(self, lower, higher):
        assert lower <= higher
        if lower == higher:
            return []
        assert higher < len(self.path)
        if self.has_speed_profile(lower, higher):  # excludes point at higher
            times = []
            # get vollumes for pathparts
            # vols = []
            prev_bound_lower = lower
            prev_r = -1
            for volume in self.volumes:
                if volume.path_indices[1] > lower and volume.path_indices[0] < higher:
                    # vols.append(v)

                    # for volume in vols:
                    low = volume.path_indices[0]
                    high = volume.path_indices[1]
                    if volume.path_indices[0] <= lower:
                        low = lower
                    if volume.path_indices[1] > higher:
                        high = higher
                    speed_m = volume.get_symval(volume.Q_1) / volume.A  # in m/s
                    speed = speed_m * 1000

                    distances = self.distances[low:high]
                    # assert len(distances) == len(rs)
                    time_add = [
                        distance / speed for j, distance in enumerate(distances)
                    ]
                    if None in time_add:
                        time_add = self.times[low:high]
                    times += time_add
                    return times
        else:
            logger.print("DEBUG: asked for volume not present!")
            return self.times[lower:higher]

    def get_cell_radius(self, index):
        """get_cell_radius returns the radius for a given index in vessel


        Args:
            index (_type_): _description_

        Returns:
            tuple: (vessel,integer of radius)
        """
        if len(self.profiles) == 0:
            # logger.print("NO VOLUME FOR INDEX and RADIUS", self.associated_vesselname)
            return (self, 0, None)
        # get volume for radius
        vol = self.get_volume_by_index(index)
        # sample a radius given the calculated cdf
        r = random.choices(
            [i for i in range(len(self.profiles[(vol.id, "rs")]))],
            cum_weights=(self.profiles[(vol.id, "rs_prob")]),
            k=1,
        )[0]
        return (self, r, vol)

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

    def determine_next_target_index(self, index, p_fun, ignore_tlinks=False,debug=False):
        """determine_next_target_index

        Args:
            index (_type_): _description_
            p_fun (_type_): _description_
            ignore_tlinks (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if debug: print("DB search",self.associated_vesselname,index == len(self.path), index, len(self.path),self.links_at(index-1))
        if not index == len(self.path): debug =False


        logger.print(f"[DETERMINE NEXT NODE] from {self.associated_vesselname[-32:],index} , reachable_vessels_general {[(v.source_vessel.associated_vesselname[-32:],v.source_index) for v in self.links_to_path]}")
        if index == len(self.path) and not any([l.target_vessel.orig==self.orig for l in self.links_to_path]): #and self.max_iteration_number==self.iteration_number:  # reached end of vessel semantically
            if debug:print("[DETERMINE AS HIGHEST]")
            #print("VEss HAS HIGHLINK with previous connect",self.highest_link() is not None
             #   and self.highest_link().source_index == index - 1)
            # see if a vessel or voi connects

            if (
                self.highest_link() is not None
                and self.highest_link().source_index == index - 1
            ):
                if self.type=="vein":
                    # take endlink if possible
                    next_index, next_object = (
                        self.highest_link().target_index,
                        self.highest_link().target_vessel,
                    )

                else:
                    links = self.links_at(index - 1, self.type == "vein") # veinmode ignores tlink
                    _, chances = p_fun(links)
                    if debug:print("taking lings2",links,chances)

                    chosen_link = random.choices(links, weights=chances, k=1)[0]
                    if debug:print("INSIDE VESS",type(chosen_link))

                    if hasattr(chosen_link, "target_vessel"):  # type(chosen_link) is Link:
                        next_index, next_object = (
                            chosen_link.target_index,
                            chosen_link.target_vessel,
                        )
                    if hasattr(
                        chosen_link, "target_tissue"
                    ):  # if type(chosen_link) is TissueLink:
                        next_index, next_object = (
                            chosen_link.target_index,
                            chosen_link.target_tissue,
                        )
                #print("Returning then: nidx,nobj",next_index,next_object,"obj differs",next_object!=self)
            else:
                # check if voi connects
                reachable_rois = self.get_rois(len(self.path) - 1)

                if len(reachable_rois) > 0:
                    # choose a roi
                    chosen_roi = random.choice(
                        reachable_rois
                    )  # TODO choose according to attraction or roivolume / rel connectiondens
                    # get closest index to reachable roi
                    return chosen_roi
                else:
                    # set next vein beginning next
                    return 0
        elif index == len(self.path):  # reached end of vessel but just partially for the volume
            #print("[DETERMINE AS HIGH WITH SAMENAME NEXT]")
            # see if a vessel or voi connects
            links = self.links_at(index - 1, self.type == "vein") # veinmode ignores tlink
            _, chances = p_fun(links)
            if debug:print("taking lings2",links,chances)

            chosen_link = random.choices(links, weights=chances, k=1)[0]
            if debug:print("INSIDE VESS",type(chosen_link))

            if hasattr(chosen_link, "target_vessel"):  # type(chosen_link) is Link:
                next_index, next_object = (
                    chosen_link.target_index,
                    chosen_link.target_vessel,
                )
            if hasattr(
                chosen_link, "target_tissue"
            ):  # if type(chosen_link) is TissueLink:
                next_index, next_object = (
                    chosen_link.target_index,
                    chosen_link.target_tissue,
                )
        else:
            # in vessel, check if next refpoint is next point or point from link!
            # get all links
            # WAIT if needed
            #print("[DETERMINE AS INNER]")
            links = self.get_links_trav(index, ignore_tlinks, self.type == "vein")
            #print("Links",links)
            assert links !=None, links
            assert p_fun!=None, p_fun
            _, chances = p_fun(links)
            chosen_link = random.choices(links, weights=chances, k=1)[0]
            # logger.print("INSIDE VESS",type(chosen_link))

            if hasattr(chosen_link, "target_vessel"):  # type(chosen_link) is Link:
                next_index, next_object = (
                    chosen_link.target_index,
                    chosen_link.target_vessel,
                )
            if hasattr(
                chosen_link, "target_tissue"
            ):  # if type(chosen_link) is TissueLink:
                next_index, next_object = (
                    chosen_link.target_index,
                    chosen_link.target_tissue,
                )
        return next_object, next_index

    def get_links_trav(self, index, ignore_tlinks=True, veinmode=False):
        # index is next possible in path
        if veinmode:
            return [Link(self, index - 1, self, index)]
        return [Link(self, index - 1, self, index)] + self.links_at(
            index - 1, ignore_tlinks
        )

    def links_at(self, index, ignore_tlinks=True):
        if not ignore_tlinks:
            return [
                link
                for link in self.links_to_path + self.links_to_vois
                if link.source_index == index
            ]
        return [link for link in self.links_to_path if link.source_index == index]

    def reduce(self, index_start, index_end_excluded):
        """reduce Reduces vessels path to the specified indices, useful for splitting a vessel into 2 parts without generating 2 new vessels

        Args:
            index_start (_type_): _description_
            index_end_excluded (_type_): _description_
        """
        self.path = self.path[index_start:index_end_excluded]
        self.diameters = self.diameters[index_start:index_end_excluded]
        self.times = self.calculate_times()
        self.length = sum(
            [
                calculate_distance_euclidian(self.path[i], self.path[i + 1])
                for i in range(len(self.path) - 1)
            ]
        )
        for link in self.links_to_path:
            link.source_index = link.source_index - index_start
            for l2 in link.target_vessel.links_to_path:
                if l2.target_vessel == self:
                    l2.target_index = link.source_index

    def __str__(self):
        return f"{self.associated_vesselname}, l:{len(self.path),len(self.diameters)},s:{[l.source_index for l in self.links_to_path]}"


def interpol_speed(type, diameter):
    if type == "vein":
        x_speed = [0.008, 0.02, 5, 30]
        f_speed = [1, 2, 100, 380]
    else:
        x_speed = [0.008, 0.05, 4, 25]  # durchmesser 25 eig aorta
        f_speed = [1, 50, 450, 480]
    return np.interp(diameter, x_speed, f_speed)


def count(array, num):
    count = 0
    for a in array:
        if a == num:
            count += 1
    return count


class Link:
    """Object that stores linkinformation"""

    def __init__(
        self,
        source_vessel,
        source_index,
        target_vessel,
        target_index,
        tag=None,
        chance=0,
    ):
        self.target_index = target_index
        self.source_index = source_index
        self.target_vessel = target_vessel
        self.source_vessel = source_vessel
        self.tag = tag
        self.chance = chance
        self.no_reverse = False

    def get_time(self):
        return self.source_vessel.speed_function(
            calculate_distance_euclidian(
                self.source_vessel.path[self.source_index],
                self.target_vessel.path[self.target_index],
            ),
            [self.source_vessel, self.target_vessel],
        )

    def __str__(self):
        return f"{self.source_vessel.id},{self.source_index},{self.target_vessel.id},{self.target_index},{self.tag},{self.chance}"


class TissueLink:
    """Object that stores linkinformation between Vessel and VOI"""

    def __init__(
        self,
        source_vessel,
        source_index,
        target_tissue,
        target_index,
        tag=None,
        chance=0.05,
    ):
        self.target_index = target_index
        self.source_index = source_index
        self.target_tissue = target_tissue
        self.source_vessel = source_vessel
        assert hasattr(self.target_tissue, "name")
        self.chance = chance
        self.no_reverse = False

    def get_time(self):
        return self.source_vessel.speed_function(
            calculate_distance_euclidian(
                self.source_vessel.path[self.source_index],
                self.target_tissue.geometry[self.target_index],
            ),
            [
                self.source_vessel
            ],  # use source vessel for speed, TODO add arteriol speed?
        )

    def __str__(self):
        return f"{self.source_vessel.id},{self.source_index},{self.target_tissue.name},{self.target_index},{self.tag},{self.chance}"

    def __hash__(self):
        return hash(
            f"{self.source_vessel},{self.target_tissue},{self.source_index},{self.target_index}"
        )


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
