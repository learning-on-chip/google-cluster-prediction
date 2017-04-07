from .learner import Learner
import numpy as np


class Baseline:
    def __init__(self, *arguments):
        learner = Learner(*arguments)
        self.input = learner.input
        self.teacher = learner.teacher
        self.summarer = learner.summarer

    def run_test(self):
        return self.teacher.assess(self.input.test, self._run_assessment)

    def run_validation(self):
        return self.teacher.assess(self.input.validation, self._run_assessment)

    def _run_assessment(self, sample, future_length):
        sample_length, dimension_count = sample.shape
        y_hat = np.empty([sample_length, future_length, dimension_count])
        for i in range(sample_length):
            for j in range(future_length):
                y_hat[i, j, :] = sample[i, :]
        return y_hat
