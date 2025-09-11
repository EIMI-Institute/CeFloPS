import random
import matplotlib.pyplot as plt


class Sampler:
    def __init__(self, interval, name=None):
        self.entries = dict()
        self.sampling_interval = interval
        self.name_generic = False
        if name == None:
            self.name_generic = True
            name = id(self)
        self.name = name

    def add_entry(self, time):
        key = time // self.sampling_interval
        key += 1
        if key not in self.entries:
            self.entries[key] = 0
        self.entries[key] += 1

    def rem_entry(self, time):
        key = time // self.sampling_interval
        key += 1
        if key not in self.entries:
            self.entries[key] = -1
            print("Negative Value warning")
        self.entries[key] -= 1

    def get_plot_data(self):
        return self.entries.keys(), self.entries.values()

    def get_plot_figure(self, name_add="", any_name=False):
        maxkey = 1
        for key in self.entries.keys():
            if key > maxkey:
                maxkey = key
        for i in range(1, maxkey):
            if i not in self.entries.keys():
                self.entries[i] = 0
        self.entries = dict(sorted(self.entries.items()))
        fig = plt.figure(figsize=(12, 8))
        # adding multiple Axes objects
        ax = plt.subplot()
        title = "Sampler Interval: ", self.sampling_interval
        if not self.name_generic or any_name:
            title = "Sampler Interval: ", self.sampling_interval, self.name
        ax.set_title(str(title) + str(name_add))
        ax.set_xlabel("Time " + str(self.sampling_interval) + " s")
        ax.set_ylabel("Samples")

        ax.plot(
            [0] + list(self.entries.keys()),
            [self.entries[1]] + list(self.entries.values()),
            drawstyle="steps-pre",
        )
        plt.show()
        # del


# create possible rois and traversal between those asociated with a ROI
class Compartment:
    def __init__(
        self,
        k_outs,
        cell_speed,
        cell_status,
        k_in=0,
        roi=None,
        C_outs=None,  # none if directly supplied from blood in cappilary system
        cells=0,
        descriptor="Compartment for tracer",
        type_id="NaN",
    ):
        self.k_in_sampler = Sampler(60, "cells_in")  # 60 seconds sampler
        self.k_out_sampler = Sampler(60, "cells_out")  # 60 seconds sampler
        self.cell_count_sampler = Sampler(60, "cell_count")  # 60 seconds sampler
        self.type = type_id
        # todo make diferent sample kinds
        self.k_in = k_in
        self.C_outs = C_outs
        assert len(k_outs) == len(C_outs)
        self.cells = cells
        # self.previous_stati = {}
        # self.previous_speeds = {}
        self.roi = roi
        self.cell_speed = cell_speed
        self.cell_status = cell_status
        self.descriptor = descriptor
        self.tracked_connected_compartments = []
        self.k_outs = k_outs

    def get_cell_count(self):
        return self.cells

    def get_concentration(self):
        return self.cells / self.roi.volume

    def get_full_cell_count(self):
        return self.cells + sum(
            [comp.get_cell_count() for comp in self.tracked_connected_compartments]
        )

    def get_compartment_cellcounts(self):
        others = [comp.get_cell_count() for comp in self.tracked_connected_compartments]
        return [self.cells] + [
            comp.get_cell_count() for comp in self.tracked_connected_compartments
        ]

    def get_compartment(self, status):
        for c in [self] + self.tracked_connected_compartments:
            if c.cell_status == status:
                return c

    """def affect_cell(self, cell):
        if cell.status not in previous_stati.keys():
            self.previous_stati[cell.status] = [cell]
        else:
            self.previous_stati[cell.status].append[cell]
        if cell.speed not in previous_speeds.keys():
            self.previous_speeds[cell.speed] = [cell]
        else:
            self.previous_speeds[cell.speed].append[cell]

        cell.speed = self.cell_speed
        cell.status = self.cell_status"""

    """def unaffect_cell(self):
        self.previous_stati[cell.status] = [cell]
        if len(previous_speeds) == 1:
            cell.speed = [key for key in previous_speeds.keys()][0]
        else:
            # look for cell
            speeds = [key for key, v in previous_speeds.items() if v == cell]
            assert len(speeds) == 1
            cell.speed = speeds[0]
        if len(previous_stati) == 1:
            cell.status = [key for key in previous_stati.keys()][0]
        else:
            # look for cell
            stati = [key for key, v in previous_stati.items() if v == cell]
            assert len(stati) == 1
            cell.status = stati[0]"""

    def add_compartment(self, other_compartment, k_out):
        """Adds a compartment to the connected and tracked compartments. Those all should be connected ONLY to this Compartment

        Args:
            other_compartment (Compartment): compartment to be added to tracking list
        """
        assert self.roi != None, "ROI has to be assigned for C1"
        other_compartment.roi = self.roi
        self.C_outs.append(other_compartment)
        self.k_outs.append(k_out)
        self.tracked_connected_compartments.append(other_compartment)

    def add_cell(self, time=None):
        self.cells += 1
        if time != None:
            self.k_in_sampler.add_entry(time)
            self.cell_count_sampler.add_entry(time)
        # self.samples[time //sampling_interval] += 1
        # self.affect_cell(cell)

    def sub_cell(self, time=None):
        self.cells -= 1
        if time != None:
            self.k_out_sampler.add_entry(time)
            self.cell_count_sampler.rem_entry(time)
        # self.samples[time //sampling_interval] -= 1
        # cells.remove(cell)
        # Note that unaffecting a cell isnt needed as a cell will be affected by another compartment overriding affected variables

    def __str__(self):
        return str(
            (
                self.roi.name,
                self.descriptor,
                self.cells,
                self.k_outs,
                self.C_outs,
                [str(c) for c in self.tracked_connected_compartments],
            )
        )
