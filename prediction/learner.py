from . import support
from tensorflow.contrib import rnn as crnn
from tensorflow.python.ops import rnn
import numpy as np
import tensorflow as tf


class Learner:
    def __init__(self, x, y, config):
        self.x, self.y = x, y
        with tf.variable_scope('batch_size'):
            self.batch_size = tf.shape(x)[0]
        with tf.variable_scope('network'):
            self.start, self.finish, h = Learner._network(x, config)
        with tf.variable_scope('unroll_count'):
            self.unroll_count = tf.shape(h)[1]
        with tf.variable_scope('regression'):
            w, b = Learner._regression(
                self.batch_size, self.unroll_count, config)
        with tf.variable_scope('y_hat'):
            self.y_hat = tf.matmul(h, w) + b
        self.parameters = tf.trainable_variables()
        support.log(self, 'Parameters: {}', self.parameter_count)

    @property
    def parameter_count(self):
        return np.sum([int(np.prod(p.get_shape())) for p in self.parameters])

    def _finish(state, config):
        parts = []
        for i in range(config.layer_count):
            parts.append(state[i].c)
            parts.append(state[i].h)
        return tf.stack(parts, name='finish')

    def _initialize(config):
        name = 'random_{}_initializer'.format(config.initializer.name)
        return getattr(tf, name)(**config.initializer.options)

    def _network(x, config):
        name = '{}Cell'.format(config.cell.name)
        cell = getattr(crnn, name)(config.unit_count,
                                   initializer=Learner._initialize(config),
                                   **config.cell.options)
        cell = crnn.DropoutWrapper(cell, **config.dropout.options)
        cell = crnn.MultiRNNCell([cell] * config.layer_count)
        start, state = Learner._start(config)
        h, state = rnn.dynamic_rnn(cell, x, initial_state=state)
        finish = Learner._finish(state, config)
        return start, finish, h

    def _regression(batch_size, unroll_count, config):
        w = tf.get_variable(
            'w', [1, config.unit_count, config.dimension_count],
            initializer=Learner._initialize(config))
        b = tf.get_variable('b', [1, 1, config.dimension_count])
        w = tf.tile(w, [batch_size, 1, 1])
        b = tf.tile(b, [batch_size, unroll_count, 1])
        return w, b

    def _start(config):
        shape = [2 * config.layer_count, 1, config.unit_count]
        start = tf.placeholder(tf.float32, shape, name='start')
        parts = tf.unstack(start)
        state = []
        for i in range(config.layer_count):
            c, h = parts[2 * i], parts[2 * i + 1]
            state.append(crnn.LSTMStateTuple(c, h))
        return start, tuple(state)
