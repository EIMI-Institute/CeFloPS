from .roi import Roi
import CeFloPS.simulation.common.functions as f


class Blood_roi(Roi):
    #ROI wrapper object for the vessel system
    def __init__(
        self,
        name,
        path,
        roi_type,
        no_vol_speed=False,
        blood_roi=None,
        additional_vois_already_in_mem=False,
    ):
        super().__init__(name, path, roi_type, None, None) 
        for vessel in self.geometry:
            vessel.times = vessel.calculate_times(no_vol_speed=no_vol_speed)
        self.volume_ml = self.volume
        self.cells = 0 
        self.registered_rois = []

    def update_concentration(self):
        self.concentration_current = self.cells / self.volume 

    def update_cell_share(self, simulation):
        self.share_current = self.cells / CeFloPS.simulation.cell_count

    def update_concentration_share(self, simulation):
        self.update_concentration()
        index = [
            index
            for index, name in enumerate(simulation.sim_concentration)
            if name == self.name
        ][0]
        self.concentration_share_current = f.normalize(
            [value for value in CeFloPS.simulation.sim_concentration.values()]
        )[index]
        self.concentration_share_target = CeFloPS.simulation.concentration[self.name]

    def get_concentration_share(self, simulation):
        self.update_concentration_share(simulation)

        return self.concentration_share_current

    def get_cell_share(self, simulation):
        self.update_cell_share(simulation)
        return self.share_current

    def get_concentration(self):
        self.update_concentration()

        return self.concentration_current

    def get_attraction(self, simulation):
        # sum([vein.get_start_flow() for vein in roi.veins])

        # assert False, "no more calls to this"
        self.update_concentration_share(simulation)
        return self.concentration_share_target - self.get_concentration_share(
            simulation
        )

    # Methods for compartment behaviour

    def register_connected_roi_comp(self, roi):
        self.registered_rois.append((roi.k1, roi))

    def get_cells(self):
        return self.cells

    def add_cell(self):
        self.cells += 1

    def sub_cell(self):
        self.cells -= 1
