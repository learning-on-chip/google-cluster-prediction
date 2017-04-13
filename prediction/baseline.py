from .learner import Learner
import numpy as np


class Baseline(Learner):
    def __init__(self, *arguments):
        super(Baseline, self).__init__(*arguments)

    def _assess(self, sample, future_length):
        sample_length, dimension_count = sample.shape
        y_hat = np.empty([sample_length, future_length, dimension_count])
        for i in range(sample_length):
            for j in range(future_length):
                y_hat[i, j, :] = sample[i, :]
        return y_hat

    def _run_assessment(self, target):
        return self.teacher.assess(getattr(self.input, target), self._assess)
