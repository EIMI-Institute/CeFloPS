"""roi.py

"""

import trimesh
import scipy
import numpy
import math
from .compartment_model import CompartmentModel
import CeFloPS.simulation.settings as settings
import pickle
from .unit_conv import mm3_ml
from .shared_geometry import SharedGeometry

# from .simulation import Simulation
import random


# lambda is (math.log(2) / half_life)
def lambda_decay_constant(half_life):
    return math.log(2) / half_life


def n(n_0, t, half_life):
    return n_0 * math.exp(-(math.log(2) / half_life) * t)


def n_becq(n_0, becquerel, t, half_life):
    return (math.log(2) / half_life) * n_0 * math.exp(-(math.log(2) / half_life) * t)


def activity(n_0, t, half_life):  # Becquerel
    return (math.log(2) / half_life) * n(n_0, t, half_life)


def n_from_b(n_0, half_life):
    return activity(n_0, 0, half_life) / (math.log(2) / half_life)


def get_concentration(compartment):
    activity(
        len([cell for cell in compartment.cells if cell.status != "decayed"]),
        half_life=6600,
    ) / compartment.roi.get_volume()


def fdg_activity(tracercount, t=0):
    return activity(tracercount, t, 6600)


def fdg_count_from_activity(activity, t=0):
    return activity / (math.log(2) / 6600)


def load_as_roi(path):
    mesh = trimesh.load(path)
    cubes = mesh.voxelized(0.2)
    ndarray = numpy.asarray(cubes.points)
    ndarray.size  # 0.54 mb, mit 0..1 statt 0.2: 3.5


class Roi:
    """Represents a region of interest: tissue of organs etc."""

    def __init__(self, name, path, roi_type, compartment_model_ks, blood_roi):
        print(path)
        self.name = name
        self.concentration_current = 0
        self.share_current = 0
        self.share_target = 0
        self.target_fraction = 1
        self.type = roi_type
        self.volume = 0
        self.blood_roi = blood_roi
        self.veins = []
        self.cell_status = None
        self.compartment_model_ks = compartment_model_ks
        self.commpartment_model = None
        # load geometry or create teststump
        if path != "test":
            """if roi_type == 2:
                mesh = trimesh.load(path)
                self.center = mesh.centroid
                m = mesh.voxelized(settings.ROI_VOXEL_PITCH)
                m.hollow()

                self.volume = mm3_ml(m.volume)
                self.geometry = SharedGeometry(m)

                modelcreator = CompartmentModel()
                self.compartment_model = modelcreator.create(
                    2,
                    compartment_model_ks,
                    self,
                    self.blood_roi,
                )
                self.k1 = compartment_model_ks[0]
                # self.blood_roi.register_connected_roi_comp(self)
            elif roi_type == 3:
                mesh = trimesh.load(path)
                self.center = mesh.centroid
                m = mesh.voxelized(settings.ROI_VOXEL_PITCH)
                m.fill()
                self.volume = mm3_ml(m.volume)
                self.geometry = SharedGeometry(m)

                modelcreator = CompartmentModel()
                self.compartment_model = modelcreator.create(
                    2,
                    compartment_model_ks,
                    self,
                    self.blood_roi,
                )
                self.k1 = compartment_model_ks[0]
                # self.blood_roi.register_connected_roi_comp(self)"""  # not loaded, tissue roi parallel instead
            if roi_type == 1:
                from CeFloPS.simulation.common.shared_geometry import SharedGeometryConfig
                reset_geom_mode=False
                if SharedGeometryConfig.disk_load_mode==False:
                    reset_geom_mode=True
                    # Enable disk mode for main process loading
                    SharedGeometryConfig.set_mode(True)
                with open(r"" + path, "rb") as input_file:
                    loaded_vessels, additional_vois = pickle.load(input_file)
                    self.geometry = loaded_vessels

                    for vessel in self.geometry:#[0]:
                        vessel.register_volumes()

                    self.volume = sum(
                        [
                            mm3_ml(vessel.volume)
                            for vessel in self.geometry
                            if hasattr(vessel, "volume")
                        ])
                    for x in additional_vois:
                            x.reload(self)#self.geometry[0])
                    self.additional_vois=additional_vois
                if reset_geom_mode:
                    SharedGeometryConfig.set_mode(False)
            else:
                assert False
                self.geometry = None
                self.volume = 800
                self.k1 = random.random()

        self.concentration_share_current = 0
        self.concentration_share_target = 0

    def recreate_compartments(self):
        modelcreator = CompartmentModel()
        self.compartment_model = modelcreator.create(
            2,
            self.compartment_model_ks,
            self,
            self.blood_roi,
        )
        self.k1 = self.compartment_model_ks[0]

    def set_target_fraction(self, fraction):
        self.target_fraction = fraction

    def get_volume(self):
        return self.volume

    def add_cell(self):
        pass

    def update_concentration(self):
        pass

    def update_cell_share(self, simulation):
        pass

    def update_concentration_share(self, simulation):
        pass

    def get_concentration_share(self, simulation):
        pass

    def get_cell_share(self, simulation):
        pass

    def get_concentration(self):
        pass

    def get_attraction(self, simulation):
        pass
