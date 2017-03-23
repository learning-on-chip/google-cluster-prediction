import numpy as np
import tensorflow as tf


class Teacher:
    def __init__(self, model, config):
        self.tester = config.tester
        with tf.variable_scope('train_loss'):
            self.train_loss = tf.reduce_mean(
                tf.squared_difference(model.y, model.y_hat))
        gradient = tf.gradients(self.train_loss, model.parameters)
        gradient, _ = tf.clip_by_global_norm(gradient, config.gradient_clip)
        name = '{}Optimizer'.format(config.optimizer.name)
        optimizer = getattr(tf.train, name)(**config.optimizer.options)
        self.train_step = optimizer.apply_gradients(
            zip(gradient, model.parameters))

    def test(self, input, predict):
        return Teacher._test(input, self.tester.length, predict)

    def _test(input, test_length, predict):
        rmse_sum = np.zeros([test_length])
        nrmse_sum = np.zeros([test_length])
        squared = np.zeros([test_length])
        count = np.zeros([test_length], dtype=np.int)
        for sample in range(input.sample_count):
            sample = input.get(sample)
            sample_length = sample.shape[0]
            y_hat = predict(sample, test_length)
            squared.fill(0)
            count.fill(0)
            for i in range(sample_length):
                future_length = np.min([sample_length - (i + 1), test_length])
                deviation = y_hat[i, :future_length, :] - \
                            sample[(i + 1):(i + 1 + future_length), :]
                squared[:future_length] += np.sum(deviation**2, axis=0)
                count[:future_length] += 1
            rmse = np.sqrt(squared / count)
            rmse_sum += rmse
            nrmse_sum += rmse / (np.amax(sample) - np.amin(sample))
        return {
            'MRMSE': rmse_sum / input.sample_count,
            'MNRMSE': nrmse_sum / input.sample_count,
        }
