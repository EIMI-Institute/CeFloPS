import matplotlib.pyplot as plt
import numpy as np


class CompartmentPlotter:
    def __init__(self, roi):
        self.roi = roi
        self.C1 = []
        self.C2 = []
        self.times = []

    def get_sample(self, time):
        values = self.roi.compartment_model.get_compartment_cellcounts()
        assert len(values) == 2
        self.C1.append(values[0])
        self.C2.append(values[1])
        self.times.append(time)

    def combine(self, another_plotter):
        assert len(another_plotter.C1) == len(self.C1)
        assert len(another_plotter.C1) == len(self.C1)
        assert len(another_plotter.times) == len(self.times)

        self.C1 = np.asarray(self.C1) + np.asarray(another_plotter.C1)
        self.C2 = np.asarray(self.C2) + np.asarray(another_plotter.C2)

    def save_plot(self, path):
        plt.clf()
        plt.plot(self.times, self.C1, label="C1 cells")
        plt.plot(self.times, self.C2, label="C2 cells")
        plt.legend()
        plt.savefig(path)

    def get_plot(self):
        plt.plot(self.times, self.C1, label="C1 cells")
        plt.plot(self.times, self.C2, label="C2 cells")
        plt.legend()
        return plt.gcf()
