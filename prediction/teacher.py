import numpy as np
import tensorflow as tf


class Teacher:
    def run_test(input, length, evaluate):
        sums = np.zeros([length])
        counts = np.zeros([length], dtype=np.int)
        def _callback(sample, y_hat, offset):
            length = min(sample.shape[0] - offset, y_hat.shape[0])
            delta = y_hat[:length, :] - sample[offset:(offset + length), :]
            sums[:length] += np.sum(delta**2, axis=0)
            counts[:length] += 1
        for sample in range(input.sample_count):
            evaluate(input.get(sample), _callback)
        return sums / counts

    def __init__(self, model, config):
        with tf.variable_scope('loss'):
            self.loss = Teacher._loss(model.y, model.y_hat)
        gradient = tf.gradients(self.loss, model.parameters)
        gradient, _ = tf.clip_by_global_norm(gradient, config.gradient_clip)
        name = '{}Optimizer'.format(config.optimizer.name)
        optimizer = getattr(tf.train, name)(**config.optimizer.options)
        self.step = optimizer.apply_gradients(zip(gradient, model.parameters))

    def test(self, *arguments):
        return Teacher.run_test(*arguments)

    def _loss(y, y_hat):
        return tf.reduce_mean(tf.squared_difference(y, y_hat))
