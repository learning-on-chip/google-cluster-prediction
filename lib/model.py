from tensorflow.contrib import rnn as crnn
from tensorflow.python.ops import rnn
import tensorflow as tf


class Model:
    def __init__(self, config):
        shape = [None, None, config.dimension_count]
        self.x = tf.placeholder(tf.float32, shape, name='x')
        self.y = tf.placeholder(tf.float32, shape, name='y')
        with tf.variable_scope('batch_size'):
            self.batch_size = tf.shape(self.x)[0]
        with tf.variable_scope('network'):
            self.start, self.finish, h = Model._network(self.x, config)
        with tf.variable_scope('unroll_count'):
            self.unroll_count = tf.shape(h)[1]
        with tf.variable_scope('regression'):
            w, b = Model._regress(self.y, self.batch_size, self.unroll_count,
                                  config)
        with tf.variable_scope('y_hat'):
            self.y_hat = tf.matmul(h, w) + b
        with tf.variable_scope('loss'):
            self.loss = Model._assess(self.y, self.y_hat)

    def _assess(y, y_hat):
        return tf.reduce_mean(tf.squared_difference(y, y_hat))

    def _finish(state, config):
        parts = []
        for i in range(config.layer_count):
            parts.append(state[i].c)
            parts.append(state[i].h)
        return tf.stack(parts, name='finish')

    def _initialize(config):
        name = 'random_{}_initializer'.format(config.initializer.type)
        return getattr(tf, name)(**config.initializer.options)

    def _network(x, config):
        name = '{}Cell'.format(config.cell.type)
        cell = getattr(crnn, name)(config.unit_count,
                                   initializer=Model._initialize(config),
                                   **config.cell.options)
        cell = crnn.DropoutWrapper(cell, **config.dropout.options)
        cell = crnn.MultiRNNCell([cell] * config.layer_count)
        start, state = Model._start(config)
        h, state = rnn.dynamic_rnn(cell, x, initial_state=state)
        finish = Model._finish(state, config)
        return start, finish, h

    def _regress(y, batch_size, unroll_count, config):
        w = tf.get_variable(
            'w', [1, config.unit_count, config.dimension_count],
            initializer=Model._initialize(config))
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
