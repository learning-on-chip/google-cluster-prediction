import numpy as np


class Manager:
    def __init__(self, config):
        self.backup_schedule = Schedule(config.backup_schedule)
        self.test_schedule = Schedule(config.test_schedule)

    def should_backup(self, state):
        return self.backup_schedule.should(state.iteration)

    def should_test(self, state):
        return self.test_schedule.should(state.iteration)


class Schedule:
    def __init__(self, schedule):
        self.schedule = np.cumsum(schedule)

    def should(self, iteration):
        iteration = iteration % self.schedule[-1] + 1
        phase = np.nonzero(self.schedule >= iteration)[0][0]
        return phase % 2 == 1
