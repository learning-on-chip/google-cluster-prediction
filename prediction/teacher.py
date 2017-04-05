import numpy as np
import tensorflow as tf


class Teacher:
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
        nrmse_sample_count = 0
        squared = np.zeros([future_length])
        count = np.zeros([future_length], dtype=np.int)
        for sample in range(input.sample_count):
            sample = input.get(sample)
            sample_length = sample.shape[0]
            y_hat = predict(sample, future_length)
            squared.fill(0)
            count.fill(0)
            for i in range(sample_length):
                length = np.min([sample_length - (i + 1), future_length])
                deviation = y_hat[i, :length, :] - \
                            sample[(i + 1):(i + 1 + length), :]
                squared[:length] += np.sum(deviation**2, axis=-1)
                count[:length] += 1
            rmse = np.sqrt(squared / count)
            rmse_sum += rmse
            norm = np.amax(sample) - np.amin(sample)
            if norm > 0:
                nrmse_sample_count += 1
                nrmse_sum += rmse / norm
        return {
            'MRMSE': rmse_sum / input.sample_count,
            'MNRMSE': nrmse_sum / nrmse_sample_count,
        }
