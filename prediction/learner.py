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
            self.start, self.finish, h = Candidate._network(x, config)
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

    def test(self, session, sample, future_length):
        fetch = {
            'y_hat': self.y_hat,
            'finish': self.finish,
        }
        sample_length, dimension_count = sample.shape
        y_hat = np.empty([sample_length, future_length, dimension_count])
        for i in range(sample_length):
            feed = {
                self.start: np.zeros(self.start.get_shape(), np.float32),
                self.x: np.reshape(sample[:(i + 1), :], [1, i + 1, -1]),
            }
            for j in range(future_length):
                result = session.run(fetch, feed)
                y_hat[i, j, :] = result['y_hat'][0, -1, :]
                feed[self.start] = result['finish']
                feed[self.x] = y_hat[i:(i + 1), j:(j + 1), :]
        return y_hat

    def train(self, session, optimize, loss, sample):
        shape = [1, sample.shape[0], -1]
        feed = {
            self.start: np.zeros(self.start.get_shape(), np.float32),
            self.x: np.reshape(sample, shape),
            self.y: np.reshape(support.shift(sample, -1), shape),
        }
        fetch = {
            'optimize': optimize,
            'loss': loss,
        }
        return session.run(fetch, feed)['loss']

    def validate(self, session, loss, sample):
        shape = [1, sample.shape[0], -1]
        feed = {
            self.start: np.zeros(self.start.get_shape(), np.float32),
            self.x: np.reshape(sample, shape),
            self.y: np.reshape(support.shift(sample, -1), shape),
        }
        fetch = {
            'loss': loss,
        }
        return session.run(fetch, feed)['loss']

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
        return start, finish, h

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
        start = tf.placeholder(tf.float32, shape, name='start')
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

    def test(self, _, sample, future_length):
        sample_length, dimension_count = sample.shape
        y_hat = np.empty([sample_length, future_length, dimension_count])
        for i in range(sample_length):
            for j in range(future_length):
                y_hat[i, j, :] = sample[i, :]
        return y_hat

    def validate(self, session, loss, sample):
        shape = [1, sample.shape[0], -1]
        feed = {
            self.x: np.reshape(sample, shape),
            self.y: np.reshape(support.shift(sample, -1), shape),
        }
        fetch = {
            'loss': loss,
        }
        return session.run(fetch, feed)['loss']


def Learner(config):
    if len(config) > 0:
        return tf.make_template('learner',
                                lambda x, y: Candidate(x, y, config))
    else:
        return tf.make_template('learner',
                                lambda x, y: Reference(x, y, config))
