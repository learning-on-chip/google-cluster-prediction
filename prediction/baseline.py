import numpy as np
import tensorflow as tf


class Baseline:
    def __init__(self, input, teacher, summarer):
        self.input = input
        self.teacher = teacher
        self.summarer = summarer

    def run(self):
        errors = self.teacher.assess(self.input.validation, self._run)
        for name in errors:
            tag = 'baseline_{}'.format(name)
            for i in range(len(errors[name])):
                value = tf.Summary.Value(tag=tag, simple_value=errors[name][i])
                self.summarer.add_summary(tf.Summary(value=[value]), i + 1)
        self.summarer.flush()
        return errors

    def _run(self, sample, future_length):
        y_hat = np.empty(
            [sample.shape[0], future_length, self.input.dimension_count])
        for i in range(sample.shape[0]):
            for j in range(future_length):
                y_hat[i, j, :] = sample[i, :]
        return y_hat
