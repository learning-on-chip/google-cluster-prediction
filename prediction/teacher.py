from . import support
import numpy as np
import tensorflow as tf


class Teacher:
    def assess_numeric(y, y_hat):
        return np.mean(np.ravel(y - y_hat)**2)

    def assess_symbolic(y, y_hat):
        return tf.reduce_mean(tf.squared_difference(y, y_hat))

    def __init__(self, config):
        self.future_length = config.future_length

    def test(self, input, compute):
        progress = support.Progress(subject=self, description='testing')
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

    def validate(self, input, compute):
        progress = support.Progress(subject=self, description='validation')
        sum = 0
        for sample in input.iterate():
            sum += compute(sample)
            progress.advance()
        progress.finish()
        return {
            'MSE': [sum / progress.count],
        }


class Examiner(Teacher):
    def __init__(self, learner, config):
        super(Examiner, self).__init__(config)
        with tf.variable_scope('loss'):
            self.loss = Teacher.assess_symbolic(learner.y, learner.y_hat)


class Trainer(Examiner):
    def __init__(self, learner, config):
        super(Trainer, self).__init__(learner, config)
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
