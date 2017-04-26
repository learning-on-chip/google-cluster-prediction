from . import support
import numpy as np
import tensorflow as tf


class Tester:
    def __init__(self, learner, config):
        self.learner = learner
        self.future_length = config.future_length

    def run(self, input, session, report_each=10000):
        progress = support.Progress(subject=self, description='testing',
                                    total_count=input.sample_count,
                                    report_each=report_each)
        sum, count = np.zeros([self.future_length]), 0
        for _ in input.iterate(session):
            y, y_hat = self.learner.test(session, self.future_length)
            _, sample_length, dimension_count = y.shape
            count += sample_length * dimension_count
            for i in range(sample_length):
                length = min(sample_length - (i + 1), self.future_length)
                y_hat[:length, i, :] -= y[0, (i + 1):(i + 1 + length), :]
                sum += np.sum(y_hat[:, i, :]**2, axis=-1)
            progress.advance()
        progress.finish()
        return {
            'MSE': sum / count,
        }


class Trainer:
    def __init__(self, learner, config):
        self.learner = learner
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.squared_difference(learner.y, learner.y_hat))
        if len(learner.parameters) == 0:
            self.optimize = tf.no_op()
            return
        gradient = tf.gradients(self.loss, learner.parameters)
        gradient, _ = tf.clip_by_global_norm(gradient, config.gradient_clip)
        name = '{}Optimizer'.format(config.optimizer.name)
        optimizer = getattr(tf.train, name)(**config.optimizer.options)
        self.optimize = optimizer.apply_gradients(
            zip(gradient, learner.parameters))

    def run(self, input, session, step_count):
        error = None
        for _ in input.iterate(session, step_count):
            error = self.learner.train(session, self.optimize, self.loss)
        return {
            'MSE': np.array([error]),
        }


class Validator:
    def __init__(self, learner, _):
        self.learner = learner
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.squared_difference(learner.y, learner.y_hat))

    def run(self, input, session, report_each=10000):
        progress = support.Progress(subject=self, description='validation',
                                    total_count=input.sample_count,
                                    report_each=report_each)
        sum = 0
        for _ in input.iterate(session):
            sum += self.learner.validate(session, self.loss)
            progress.advance()
        progress.finish()
        return {
            'MSE': np.array([sum / input.sample_count]),
        }
