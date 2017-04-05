from . import support
import numpy as np
import tensorflow as tf


class Teacher:
    _EPSILON = np.finfo(np.float32).eps

    def __init__(self, model, config):
        self.future_length = config.future_length
        with tf.variable_scope('training_loss'):
            self.training_loss = tf.reduce_mean(
                tf.squared_difference(model.y, model.y_hat))
        gradient = tf.gradients(self.training_loss, model.parameters)
        gradient, _ = tf.clip_by_global_norm(gradient, config.gradient_clip)
        name = '{}Optimizer'.format(config.optimizer.name)
        optimizer = getattr(tf.train, name)(**config.optimizer.options)
        self.training_step = optimizer.apply_gradients(
            zip(gradient, model.parameters))

    def assess(self, input, predict):
        return Teacher._assess(input, self.future_length, predict)

    def _assess(input, future_length, predict):
        rmse_sum = np.zeros([future_length])
        nrmse_sum = np.zeros([future_length])
        flat_count = 0
        sum = np.zeros([future_length])
        count = np.zeros([future_length], dtype=np.int)
        for sample in range(input.sample_count):
            sample = input.get(sample)
            norm = np.amax(sample) - np.amin(sample)
            if norm < Teacher._EPSILON:
                flat_count += 1
                continue
            sample_length, dimension_count = sample.shape
            y_hat = predict(sample, future_length)
            sum.fill(0)
            count.fill(0)
            for i in range(sample_length):
                length = min(sample_length - (i + 1), future_length)
                deviation = sample[(i + 1):(i + 1 + length), :] - \
                            y_hat[i, :length, :]
                sum[:length] += np.sum(deviation**2, axis=-1)
                count[:length] += dimension_count
            rmse = np.sqrt(sum / count)
            rmse_sum += rmse
            nrmse_sum += rmse / norm
        if flat_count > 0:
            support.log(
                Teacher, 'Flat samples: {}',
                support.format_percentage(flat_count, input.sample_count))
        return {
            'MRMSE': rmse_sum / (input.sample_count - flat_count),
            'MNRMSE': nrmse_sum / (input.sample_count - flat_count),
        }
