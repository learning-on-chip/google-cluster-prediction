import numpy as np
import tensorflow as tf


class Teacher:
    def __init__(self, model, config):
        self.tester = config.tester
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.squared_difference(model.y, model.y_hat))
        gradient = tf.gradients(self.loss, model.parameters)
        gradient, _ = tf.clip_by_global_norm(gradient, config.gradient_clip)
        name = '{}Optimizer'.format(config.optimizer.name)
        optimizer = getattr(tf.train, name)(**config.optimizer.options)
        self.step = optimizer.apply_gradients(zip(gradient, model.parameters))

    def test(self, input, predict):
        return Teacher._test(input, self.tester.length, predict)

    def _test(input, test_length, predict):
        sums = np.zeros([test_length])
        counts = np.zeros([test_length], dtype=np.int)
        for sample in range(input.sample_count):
            sample = input.get(sample)
            sample_length = sample.shape[0]
            y_hat = predict(sample, test_length)
            for i in range(sample_length):
                future_length = min(sample_length - (i + 1), test_length)
                delta = y_hat[i, :future_length, :] - \
                        sample[(i + 1):(i + 1 + future_length), :]
                sums[:future_length] += np.sum(delta**2, axis=0)
                counts[:future_length] += 1
        return sums / counts
