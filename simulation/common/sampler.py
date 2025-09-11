import matplotlib.pyplot as plt

import math
import numpy as np


def round_down_to_step(value, step):
    """Rundet die gegebene Zahl `value` auf den nächstkleineren Wert, der ein Vielfaches von `step` ist."""
    return np.round(math.floor(value / step) * step, 3)


class Sampler:
    def __init__(self, interval, name=None):
        self.entries = dict()
        self.sampling_interval = 1  # TODO enable lower resolution
        self.name_generic = False
        if name == None:
            self.name_generic = True
            name = str(id(self))
        self.name = name

    def add_entry(self, time):
        # key = int(time // self.sampling_interval)
        key = int(round_down_to_step(time, self.sampling_interval))
        # key += 1
        if key not in self.entries:
            self.entries[key] = 0
        self.entries[key] += 1

    def rem_entry(self, time):
        if "_in" in self.name:
            assert False, time
        # key = int(time // self.sampling_interval)
        key = int(round_down_to_step(time, self.sampling_interval))
        # key += 1

        if key not in self.entries:
            self.entries[key] = 0
            # print("Negative Value warning") TODO this is not necessary, sometimes time is in the future? -> vol sampler triple update
        self.entries[key] -= 1

    def get_plot_data(self):
        return self.entries.keys(), self.entries.values()

    def get_plot_figure(self, any_name=False, total=False):
        fig = plt.figure(figsize=(12, 8))
        # adding multiple Axes objects
        ax = plt.subplot()
        title = "Sampler Interval: ", self.sampling_interval
        if not self.name_generic or any_name:
            title = "Sampler Interval: ", self.sampling_interval, self.name
        ax.set_title(title)
        ax.set_xlabel("Time " + str(self.sampling_interval) + " s")
        ax.set_ylabel("Samples")
        self.entries = dict(sorted(self.entries.items()))
        if total:
            final_entries = dict()
            start_value = 0
            final_entries[0] = start_value
            for time, change in self.entries.items():
                start_value += change
                final_entries[time] = start_value
            for i in range(list(final_entries)[0], list(final_entries)[-1]):
                if i not in final_entries:
                    final_entries[i] = final_entries[i - 1]
            final_entries = dict(sorted(final_entries.items()))
            ax.plot(
                list(final_entries.keys()),
                list(final_entries.values()),
                drawstyle="steps-pre",
            )
            # print(final_entries.items())
            return fig
        else:
            ax.plot(
                [i for i in range(list(self.entries)[0], list(self.entries)[-1])],
                [
                    self.entries[i] if i in self.entries else 0
                    for i in range(list(self.entries)[0], list(self.entries)[-1])
                ],
                drawstyle="steps-pre",
            )
            return fig

    def save_plot(self, any_name=False, total=False, path=""):
        fig = self.get_plot_figure(any_name=any_name, total=total)
        fig.savefig(path + self.name + ".png")


class SamplerCollection:
    def __init__(self, identifier_array, stepsize=1, name="SamplingCollection"):
        self.name = name
        assert len(identifier_array) > 0
        for ident in identifier_array:
            assert type(ident) == str
            setattr(self, ident, Sampler(stepsize, ident))
        self.subsampler = [getattr(self, ident) for ident in identifier_array]

    def retrieve(self, ident):
        return getattr(self, ident).entries
