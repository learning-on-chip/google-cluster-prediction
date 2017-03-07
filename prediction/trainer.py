import tensorflow as tf


class Trainer:
    def __init__(self, model, config):
        with tf.variable_scope('loss'):
            self.loss = Trainer._loss(model.y, model.y_hat)
        gradient = tf.gradients(self.loss, model.parameters)
        gradient, _ = tf.clip_by_global_norm(gradient, config.gradient_clip)
        name = '{}Optimizer'.format(config.optimizer.name)
        optimizer = getattr(tf.train, name)(**config.optimizer.options)
        self.step = optimizer.apply_gradients(zip(gradient, model.parameters))

    def _loss(y, y_hat):
        return tf.reduce_mean(tf.squared_difference(y, y_hat))
