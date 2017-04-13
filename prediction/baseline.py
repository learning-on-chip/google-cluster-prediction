from .learner import Learner
import numpy as np


class Baseline(Learner):
    def __init__(self, *arguments):
        super(Baseline, self).__init__(*arguments)

    def run_test(self):
        return self.teacher.assess(self.input.test, self._assess)

    def run_validation(self):
        return self.teacher.assess(self.input.validation, self._assess)

    def _assess(self, sample, future_length):
        sample_length, dimension_count = sample.shape
        y_hat = np.empty([sample_length, future_length, dimension_count])
        for i in range(sample_length):
            for j in range(future_length):
                y_hat[i, j, :] = sample[i, :]
        return y_hat
