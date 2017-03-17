import numpy as np
import tensorflow as tf


class Teacher:
    def __init__(self, model, config):
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.squared_difference(model.y, model.y_hat))
        gradient = tf.gradients(self.loss, model.parameters)
        gradient, _ = tf.clip_by_global_norm(gradient, config.gradient_clip)
        name = '{}Optimizer'.format(config.optimizer.name)
        optimizer = getattr(tf.train, name)(**config.optimizer.options)
        self.step = optimizer.apply_gradients(zip(gradient, model.parameters))
        self.tester = config.tester

    def test(self, input, test):
        return Teacher._test(input, self.tester.length, test)

    def _test(input, length, test):
        sums = np.zeros([length])
        counts = np.zeros([length], dtype=np.int)
        def _callback(sample, y_hat, offset):
            tail = min(sample.shape[0] - offset, y_hat.shape[0])
            delta = y_hat[:tail, :] - sample[offset:(offset + tail), :]
            sums[:tail] += np.sum(delta**2, axis=0)
            counts[:tail] += 1
        for sample in range(input.sample_count):
            test(input.get(sample), length, _callback)
        return sums / counts
