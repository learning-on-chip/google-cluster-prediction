from . import support
from tensorflow.contrib import rnn as crnn
from tensorflow.python.ops import rnn
import numpy as np
import tensorflow as tf


class Candidate:
    def __init__(self, x, y, config):
        self.x, self.y = x, y
        with tf.variable_scope('batch_size'):
            self.batch_size = tf.shape(x)[0]
        with tf.variable_scope('unroll_count'):
            self.unroll_count = tf.shape(x)[1]
        with tf.variable_scope('network'):
            h, _, _ = Candidate._network(x, config)
        with tf.variable_scope('regression'):
            w, b = Candidate._regression(
                self.batch_size, self.unroll_count, config)
        with tf.variable_scope('y_hat'):
            self.y_hat = tf.matmul(h, w) + b
        self.parameters = tf.trainable_variables()
        support.log(self, 'Parameters: {}', self.parameter_count)

    @property
    def parameter_count(self):
        return np.sum([int(np.prod(p.get_shape())) for p in self.parameters])

    def test(self, session, future_length):
        assert(future_length == 1)
        return session.run([self.y, self.y_hat])

    def train(self, session, optimize, loss):
        return session.run([optimize, loss])[1]

    def validate(self, session, loss):
        return session.run(loss)

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
                                   initializer=Candidate._initialize(config),
                                   **config.cell.options)
        cell = crnn.DropoutWrapper(cell, **config.dropout.options)
        cell = crnn.MultiRNNCell([cell] * config.layer_count)
        start, state = Candidate._start(config)
        h, state = rnn.dynamic_rnn(cell, x, initial_state=state)
        finish = Candidate._finish(state, config)
        return h, start, finish

    def _regression(batch_size, unroll_count, config):
        w = tf.get_variable(
            'w', [1, config.unit_count, config.dimension_count],
            initializer=Candidate._initialize(config))
        b = tf.get_variable('b', [1, 1, config.dimension_count])
        w = tf.tile(w, [batch_size, 1, 1])
        b = tf.tile(b, [batch_size, unroll_count, 1])
        return w, b

    def _start(config):
        shape = [2 * config.layer_count, 1, config.unit_count]
        start = tf.placeholder_with_default(np.zeros(shape, np.float32),
                                            name='start', shape=shape)
        parts = tf.unstack(start)
        state = []
        for i in range(config.layer_count):
            c, h = parts[2 * i], parts[2 * i + 1]
            state.append(crnn.LSTMStateTuple(c, h))
        return start, tuple(state)


class Reference:
    def __init__(self, x, y, _):
        self.x, self.y, self.y_hat = x, y, x
        self.parameters = []

    def test(self, session, future_length):
        assert(future_length == 1)
        return session.run([self.y, self.y_hat])

    def validate(self, session, loss):
        return session.run(loss)


def Learner(config):
    if len(config) > 0:
        return tf.make_template(
            'learner', lambda x, y: Candidate(x, y, config))
    else:
        return tf.make_template(
            'learner', lambda x, y: Reference(x, y, config))
