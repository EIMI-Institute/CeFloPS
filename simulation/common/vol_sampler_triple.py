from .sampler import Sampler
import matplotlib.pyplot as plt
import CeFloPS.simulation.settings as settings
import numpy as np
import os
import lzma


class volume_sampler_triple:
    def __init__(
        self,
        interval,
        entry_object,
        entry_index,
        exit_object,
        exit_index,
        name=None,
        store_all_positions=False,
        dummy=False,
    ):
        self.in_s = Sampler(interval, name + "_in")
        self.out_s = Sampler(interval, name + "_out")
        self.total_s = Sampler(interval, name + "_total")
        self.position_tracker = dict()
        self.entry_object = entry_object
        self.entry_index = entry_index
        self.exit_object = exit_object
        self.exit_index = exit_index
        self.name = name
        self.store_all_positions = store_all_positions

        self.outs_t = []
        self.ins_t = []
        # manually look that there are no links between!
        if not dummy:
            assert self.entry_index < self.exit_index
            assert self.exit_index < len(self.exit_object.path)
            assert self.entry_index > -1
            if self.entry_object is self.exit_object:
                assert self.entry_index != self.exit_index
            else:
                assert (
                    False
                ), "Support for sampling over multiple vessels not yet implemented"

    def update(self, ref_object, ref_index, time):
        # points tuple(np.round(exit_point, 5))

        if ref_object is self.entry_object and ref_index is self.entry_index:
            self.in_s.add_entry(time)
            self.total_s.add_entry(time)
            self.ins_t.append(time)
        if ref_object is self.exit_object and ref_index is self.exit_index:
            self.out_s.rem_entry(time)
            self.total_s.rem_entry(time)
            self.outs_t.append(time)
        """ if self.store_all_positions:
            prev_time = -1  # does this work in second iteration?
            for i, t in enumerate(point_times):
                # save positions
                # print("time", t)
                time = int(t // 1)  # gets multiple at one second if not caught

                position = tuple(cellpoints[i])

                if prev_time != time:  # no doublke entries for one second
                    if time not in self.position_tracker:
                        self.position_tracker[time] = dict()
                    if position not in self.position_tracker[time]:
                        self.position_tracker[time][position] = 0

                    self.position_tracker[time][position] += 1
                    prev_time = time

        for i in range(len(cellpoints)):
            if tuple(np.round(cellpoints[i], 5)) == self.entry_point:
                self.in_s.add_entry(point_times[i])
                self.total_s.add_entry(point_times[i])
            if tuple(np.round(cellpoints[i], 5)) == self.exit_point:
                self.out_s.rem_entry(point_times[i])
                self.total_s.rem_entry(point_times[i]) """

    def get_plot_figure(self, any_name=False, total=False):
        print(self.name)
        import matplotlib.pyplot as plt
        from scipy.ndimage import uniform_filter1d

        fig, ax = plt.subplots(figsize=(30, 16), nrows=2, ncols=1)
        combined_values = np.union1d(self.ins_t, self.outs_t)
        unique_ins, counts_ins = np.unique(self.ins_t, return_counts=True)
        accumulated_counts_ins = np.cumsum(counts_ins)
        ins_interp = np.interp(combined_values, unique_ins, accumulated_counts_ins)

        unique_outs, counts_outs = np.unique(self.outs_t, return_counts=True)
        accumulated_counts_outs = np.cumsum(counts_outs)
        outs_interp = np.interp(combined_values, unique_outs, accumulated_counts_outs)

        ax[0].step(
            combined_values,
            ins_interp,
            where="mid",
            color="green",
            marker="o",
            label="in cumulative",
        )
        ax[0].step(
            combined_values,
            outs_interp,
            where="mid",
            color="red",
            marker="^",
            label="out cumulative",
        )

        ins_interp = np.interp(combined_values, unique_ins, accumulated_counts_ins)
        outs_interp = np.interp(combined_values, unique_outs, accumulated_counts_outs)
        difference = ins_interp - outs_interp
        N = 50
        y = uniform_filter1d(difference, size=N)
        ax[0].step(
            combined_values,
            y,
            where="mid",
            linestyle="--",
            color="blue",
            label="total as difference",
        )

        first_derivative = np.diff(difference, prepend=0)
        ax[0].step(
            combined_values,
            first_derivative,
            where="mid",
            label="First derivative of total",
            color="black",
        )

        ax[0].set_xlabel("time in s")
        ax[0].set_ylabel("event count")
        ax[0].legend()
        ax[1].step(
            combined_values,
            y,
            where="mid",
            linestyle="--",
            color="blue",
            label="total as difference",
        )
        x_half_value = combined_values[len(combined_values) // 2]
        mask = combined_values >= x_half_value
        mean_last_50_percent_xspan = np.mean(difference[mask])
        ax[1].axhline(
            y=mean_last_50_percent_xspan,
            color="orange",
            linestyle="-",
            label="Mean of last 50%",
        )

        ax[1].scatter(
            combined_values,
            first_derivative,
            label="First derivative of total",
            color="black",
            alpha=0.25,
        )
        ax[1].set_xlabel("time in s")
        ax[1].set_ylabel("event count")
        ax[1].legend()
        # plt.xticks(unique_values)
        return fig

        fig, ax = plt.subplots(figsize=(40, 4), nrows=1, ncols=3)

        upper = settings.TIME_LIMIT_S + 100
        sampler = [self.in_s, self.out_s, self.total_s]
        for i, a in enumerate(ax):
            s = sampler[i]
            title = "Sampler Interval: ", s.sampling_interval
            if not s.name_generic or any_name:
                title = "Sampler Interval: ", s.sampling_interval, s.name
            a.set_title(title)
            a.set_xlabel("Time " + str(s.sampling_interval) + " s")
            a.set_ylabel("Samples")

            s.entries = dict(sorted(s.entries.items()))
            if len(s.entries) == 0:
                s.entries[0] = 0
            if i == 2:  # total
                final_entries = dict()
                start_value = 0
                final_entries[0] = start_value
                for time, change in s.entries.items():
                    start_value += change
                    final_entries[time] = start_value
                for i in range(list(final_entries)[0], list(final_entries)[-1]):
                    if i not in final_entries:
                        final_entries[i] = final_entries[i - 1]
                final_entries = dict(sorted(final_entries.items()))
                a.plot(
                    [i for i in range(0, upper)],  # list(s.entries)[-1])],
                    [
                        final_entries[i] if i in final_entries else 0
                        for i in range(0, upper)
                    ],
                    drawstyle="steps-pre",
                )
            else:
                a.plot(
                    [i for i in range(0, upper)],  # list(s.entries)[-1])],
                    [s.entries[i] if i in s.entries else 0 for i in range(0, upper)],
                    drawstyle="steps-pre",
                )
        return fig

    def save_plot(self, any_name=False, path="", total=False):
        if not os.path.isdir(path):
            raise ValueError("The specified path is not a valid directory.")

        file_name = f"{self.name}.png"
        full_path = os.path.join(path, file_name)

        if len(full_path) > 255:
            raise ValueError(
                "The constructed file path exceeds the maximum length allowed."
            )

        if len(self.ins_t) > 0:
            fig = self.get_plot_figure(any_name=any_name, total=total)
            fig.savefig(full_path)
            plt.close(fig)
        # also save the compressed data:
        with open(path + "in_events_" + self.name, "wb") as file:
            file.write(lzma.compress(repr(self.ins_t).encode()))
        with open(path + "out_events_" + self.name, "wb") as file:
            file.write(lzma.compress(repr(self.outs_t).encode()))

    def get_sub_entries(self):
        sampler = [self.in_s, self.out_s, self.total_s]
        return [s.entries for s in sampler]

    def get_pos_entries(self):
        return self.position_tracker
