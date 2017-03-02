import tensorflow as tf


class Trainer:
    def __init__(self, model, config):
        self.epoch_count = config.epoch_count
        self.test_length = config.test_length
        with tf.variable_scope('loss'):
            self.loss = Trainer._loss(model.y, model.y_hat)
        gradient = tf.gradients(self.loss, model.parameters)
        gradient, _ = tf.clip_by_global_norm(gradient, config.gradient_clip)
        name = '{}Optimizer'.format(config.optimizer.name)
        optimizer = getattr(tf.train, name)(**config.optimizer.options)
        self.step = optimizer.apply_gradients(zip(gradient, model.parameters))
        self.state = tf.Variable([0, 0, 0], name='state', dtype=tf.int64,
                                 trainable=False)
        self.state_update = tf.placeholder(tf.int64, shape=3,
                                           name='state_update')
        self.update_state = self.state.assign(self.state_update)

    def get_state(self, session):
        return State.deserialize(session.run(self.state))

    def set_state(self, session, state):
        session.run(self.update_state, {
            self.state_update: state.serialize(),
        })

    def _loss(y, y_hat):
        return tf.reduce_mean(tf.squared_difference(y, y_hat))


class State:
    def deserialize(state):
        return State(*state)

    def __init__(self, time, epoch, sample):
        self.time = time
        self.epoch = epoch
        self.sample = sample

    def increment_epoch(self):
        self.epoch += 1
        self.sample = 0

    def increment_time(self):
        self.time += 1
        self.sample += 1

    def serialize(self):
        return [self.time, self.epoch, self.sample]
