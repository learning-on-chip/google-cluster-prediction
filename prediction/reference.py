from . import support
from .teacher import Teacher
import numpy as np
import tensorflow as tf


class Reference:
    def __init__(self, input, config):
        self.input = input
        self.teacher = Teacher(config.teacher)
        self.summarer = tf.summary.FileWriter(config.output.path)

    def run_comparison(self, target):
        errors = getattr(self, '_run_' + target)()
        support.summarize_static(self.summarer, errors, 'comparison_' + target)

    def _run_testing(self):
        return self.teacher.test(self.input.testing, self._test)

    def _run_validation(self):
        return self.teacher.validate(self.input.validation, self._validate)

    def _test(self, sample, future_length):
        sample_length, dimension_count = sample.shape
        y_hat = np.empty([sample_length, future_length, dimension_count])
        for i in range(sample_length):
            for j in range(future_length):
                y_hat[i, j, :] = sample[i, :]
        return y_hat

    def _validate(self, sample):
        return Teacher.assess_numeric(support.shift(sample, -1), sample)
