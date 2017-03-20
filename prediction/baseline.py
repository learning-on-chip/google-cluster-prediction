import numpy as np
import tensorflow as tf


class Baseline:
    def __init__(self, input, teacher, summary_writer):
        self.input = input
        self.teacher = teacher
        self.summary_writer = summary_writer

    def run(self):
        loss = self.teacher.test(self.input.test, self._run)
        for i in range(len(loss)):
            value = tf.Summary.Value(tag='baseline_loss', simple_value=loss[i])
            self.summary_writer.add_summary(
                tf.Summary(value=[value]), i + 1)
        return loss

    def _run(self, sample, test_length):
        y_hat = np.empty(
            [sample.shape[0], test_length, self.input.dimension_count])
        for i in range(sample.shape[0]):
            for j in range(test_length):
                y_hat[i, j, :] = sample[i, :]
        return y_hat
