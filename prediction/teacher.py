from . import support
import numpy as np
import tensorflow as tf


class Examiner:
    def __init__(self, learner, config):
        self.future_length = config.future_length
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.squared_difference(learner.y, learner.y_hat))

    def test(self, input, compute, report_each=10000):
        progress = support.Progress(subject=self, description='testing',
                                    report_each=report_each)
        sum = np.zeros([self.future_length])
        for sample in input.iterate():
            sample_length, dimension_count = sample.shape
            y_hat = compute(sample, self.future_length)
            for i in range(sample_length):
                length = min(sample_length - (i + 1), self.future_length)
                y_hat[i, :length, :] -= sample[(i + 1):(i + 1 + length), :]
                sum += np.sum(y_hat[i, :, :]**2, axis=-1)
            progress.advance(sample_length * dimension_count)
        progress.finish()
        return {
            'MSE': sum / progress.count,
        }

    def validate(self, input, compute, report_each=10000):
        progress = support.Progress(subject=self, description='validation',
                                    report_each=report_each)
        sum = 0
        for sample in input.iterate():
            sum += compute(sample)
            progress.advance()
        progress.finish()
        return {
            'MSE': [sum / progress.count],
        }


class Trainer(Examiner):
    def __init__(self, learner, config):
        super(self.__class__, self).__init__(learner, config)
        if len(learner.parameters) == 0:
            self.optimize = tf.no_op()
            return
        gradient = tf.gradients(self.loss, learner.parameters)
        gradient, _ = tf.clip_by_global_norm(gradient, config.gradient_clip)
        name = '{}Optimizer'.format(config.optimizer.name)
        optimizer = getattr(tf.train, name)(**config.optimizer.options)
        self.optimize = optimizer.apply_gradients(
            zip(gradient, learner.parameters))

    def train(self, input, compute):
        return {
            'MSE': [compute(input.next())],
        }
