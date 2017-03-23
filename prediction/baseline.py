import numpy as np
import tensorflow as tf


class Baseline:
    def __init__(self, input, teacher, summarer):
        self.input = input
        self.teacher = teacher
        self.summarer = summarer

    def run(self):
        loss = self.teacher.test(self.input.test, self._run)
        for name in loss:
            tag = 'baseline_{}'.format(name)
            for i in range(len(loss[name])):
                value = tf.Summary.Value(tag=tag, simple_value=loss[name][i])
                self.summarer.add_summary(tf.Summary(value=[value]), i + 1)
        self.summarer.flush()
        return loss

    def _run(self, sample, test_length):
        y_hat = np.empty(
            [sample.shape[0], test_length, self.input.dimension_count])
        for i in range(sample.shape[0]):
            for j in range(test_length):
                y_hat[i, j, :] = sample[i, :]
        return y_hat
