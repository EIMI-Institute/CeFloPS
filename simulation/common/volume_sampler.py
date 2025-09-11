from .sampler import Sampler


class volume_sampler(Sampler):
    def __init__(self, interval, entry_point, exit_point, name=None):
        super().__init__(interval, name)
        self.entry_point = entry_point
        self.exit_point = exit_point
        # manually look that there are no links between!

    def update(self, cellpoints, point_times):
        for i in range(len(cellpoints)):
            if tuple(cellpoints[i]) == tuple(self.entry_point):
                self.add_entry(point_times[i])

            if tuple(cellpoints[i]) == tuple(self.exit_point):
                self.rem_entry(point_times[i])
