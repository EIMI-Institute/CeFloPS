import random
import math


class Distributor:
    """Class for Distributor Object that can be used to generate cellpositions for rois"""

    def __init__(self, rois, arr, restriction_tags=None, count=0, vein_start=False):
        """__init__ Initialisation method

        Args:
            rois (list): _description_
            arr (list or string): Type of generation, either an array with count per roi or a string, even or random
            restriction_tags (list of strings, optional): list of stringtags to restrict included rois for possible positions. Defaults to None.
            count (int, optional): Number of cellspositions to generate, not needed when specifying arr. Defaults to 0.
            vein_start (bool, optional): if blood is restriction tags, only use veins for position creation. Defaults to False.
        """
        self.vein_start = False
        self.vein_start = vein_start and restriction_tags == ["blood"]
        # arr is the arrangement as array or as string
        self.rois = rois
        self.arr = arr  # cellcount per roi
        if arr == "random":
            assert count > 0
            self.distribution = self.distribute_randomly(rois, count, restriction_tags)
        elif arr == "even":
            assert count > 0
            self.distribution = self.distribute_evenly(
                rois, count, restriction_tags=restriction_tags
            )
        else:
            assert len(rois) == len(
                arr
            ), "cell distribution either omits rois or has too many"

    def distribute_randomly(self, rois, count, restriction_tags):
        """distribute_randomly Generate a distribution (restricted on restriction tag named rois if given) and fill a random amount of cells per roi at a random possible position within the roi

        Args:
            rois (list): Roiobjects, vois or vesselsystem
            count (int): Number of cellspositions to generate
            restriction_tags (list of strings, optional): list of stringtags to restrict included rois for possible positions. Defaults to None.

        Returns:
            list of tuples: positions for cells
        """
        results = []

        allowed_rois = rois
        if restriction_tags is not None:
            allowed_rois = [
                roi
                for roi in rois
                if any([rtag in roi.name for rtag in restriction_tags])
            ]
        remaining = count
        for roi in allowed_rois:
            c_count = random.randint(0, remaining)
            if roi == allowed_rois[-1]:
                c_count = remaining
            # print("getting position,", c)
            if roi.type == 1:
                possible_points = []
                for vessel in roi.geometry:
                    if self.vein_start:
                        if vessel.type == "artery":
                            continue  # skip arteries
                    for i in range(len(vessel.path)):
                        possible_points.append((vessel, i))
            elif roi.type == 2:
                possible_points = roi.geometry.get_points()
            elif roi.type == 3:
                possible_points = roi.geometry.get_points()

            for i in range(c_count):
                results.append((random.choice(possible_points), roi))

            remaining -= c_count
        return results

    def distribute_evenly(self, rois, count, restriction_tags=None):
        """distribute_evenly Distribute even amount of cells per roi randomly among rois

        Args:
            rois (list): Roiobjects, vois or vesselsystem
            count (int): Number of cellspositions to generate
            restriction_tags (list of strings, optional): list of stringtags to restrict included rois for possible positions. Defaults to None.

        Returns:
            list of tuples: positions for cells
        """
        results = []
        c_in_comp = [math.floor(count / len(rois)) for c in rois]
        remaining = count - sum(c_in_comp)
        c_in_comp[0] += remaining

        if restriction_tags is not None:
            allowed_rois = [
                roi
                for roi in rois
                if any([rtag in roi.name for rtag in restriction_tags])
            ]

        for i, roi in enumerate(allowed_rois):
            # print("getting position,", c)
            c_count = c_in_comp[i]
            if roi.type == 1:
                possible_points = []
                for vessel in roi.geometry:
                    if self.vein_start:
                        if vessel.type == "artery":
                            continue  # skip arteries
                    for i in range(len(vessel.path)):
                        possible_points.append((vessel, i))
            elif roi.type == 2:
                possible_points = roi.geometry.get_points()
            elif roi.type == 3:
                possible_points = roi.geometry.get_points()

            for i in range(c_count):
                results.append((random.choice(possible_points), roi))

            remaining -= c_count
        return results

    def get_distribution(self, larm=False, rleg=False):
        """get_distribution Return the distribution from the distributor

        Args:
            larm (bool, optional): Set bool to start in larmvein. Defaults to False.

        Returns:
            list of tuples: positions for cells
        """
        if larm or rleg:
            # leg by default, overriden by arm
            distinct_volume_ident = "2383700382736_vl.blood_vessel_332"#"2383689294352_vl.blood_vessel_595"#
            #"2381568985680_vl.blood_vessel_405"
            offset = 200#350#200
            larmvein=None
            if larm:

                offset = 2
                distinct_volume_ident = (
                    "2381639734288_vl.blood_vessel_14" 
                )
            blood_roi = [roi for roi in self.rois if "blood" in roi.name][0]
            vessels = blood_roi.geometry
            for vessel in vessels:
                if distinct_volume_ident in vessel.id:
                    larmvein = vessel
                    vessel_injection_index = len(vessel.path) - offset
                """ if hasattr(vessel, "volume"):
                    if vessel.volume == distinct_volume_ident and vessel.type == "vein":
                        larmvein = vessel """
            if larmvein==None:
                larmvein=blood_roi.geometry[0]
                offset=10
                vessel_injection_index = len(vessel.path) - offset
            results = []
            for roi_index, roi in enumerate(self.rois):
                c = self.arr[roi_index]
                for i in range(c):
                    results.append(((larmvein, vessel_injection_index), roi))
            return results
        else:
            if self.arr != "even" and self.arr != "random":
                return self.create_roi_positions()
            elif self.arr == "even" or self.arr == "random":
                return self.distribution
            else:
                print("Not implemented")

    def create_roi_positions(self):
        """create_roi_positions create roi positions given distribution from arr and randomly choosing from possible positions

        Returns:
            list of tuples: positions for cells
        """
        results = []
        for roi_index, roi in enumerate(self.rois):
            c = self.arr[roi_index]
            # print("getting position,", c)
            if roi.type == 1:
                possible_points = []
                for vessel in roi.geometry:
                    for i in range(len(vessel.path)):
                        possible_points.append((vessel, i))
            elif roi.type == 2:
                possible_points = roi.geometry.get_points()
            elif roi.type == 3:
                possible_points = roi.geometry.get_points()

            for i in range(c):
                results.append((random.choice(possible_points), roi))
            # get random point for each assigned cell
            # positions.extend([random.randint(0, len(possible_points)), rois[comparitment_index])for i in range(0, self.distribution[roi_index])])
        return results
