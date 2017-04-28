from . import support
from tensorflow.contrib import rnn
import numpy as np
import tensorflow as tf


class Candidate:
    def __init__(self, x, y, config):
        self.x, self.y = x, y
        batch_size = x.get_shape().as_list()[0]
        with tf.variable_scope('unroll_count'):
            self.unroll_count = tf.shape(x)[1]
        with tf.variable_scope('network'):
            h, self.start, self.finish = Candidate._network(x, batch_size,
                                                            config)
        with tf.variable_scope('regression'):
            w, b = Candidate._regression(batch_size, self.unroll_count, config)
        with tf.variable_scope('y_hat'):
            self.y_hat = tf.matmul(h, w) + b
        self.parameters = tf.trainable_variables()
        support.log(self, 'Parameters: {}', self.parameter_count)

    @property
    def parameter_count(self):
        return np.sum([int(np.prod(p.get_shape())) for p in self.parameters])

    def test(self, session, input, future_length):
        x = session.run(input.x)
        if future_length == 1:
            y_hat = session.run(self.y_hat, {self.x: x})
        else:
            _, sample_length, dimension_count = x.shape
            y_hat = np.zeros([future_length, sample_length, dimension_count])
            for i in range(sample_length):
                feed = {
                    self.x: x[:, :(i + 1), :],
                }
                for j in range(future_length):
                    y_hat_k, finish = session.run([self.y_hat, self.finish], feed)
                    y_hat[j, i, :] = y_hat_k[0, -1, :]
                    feed[self.start] = finish
                    feed[self.x] = y_hat_k[:1, -1:, :]
        return _time_travel(x, future_length), y_hat

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

    def _network(x, batch_size, config):
        klass = getattr(rnn, '{}Cell'.format(config.cell.name))
        cells = []
        for _ in range(config.layer_count):
            cell = klass(config.unit_count,
                         initializer=Candidate._initialize(config),
                         reuse=tf.get_variable_scope().reuse,
                         **config.cell.options)
            cell = rnn.DropoutWrapper(cell, **config.dropout.options)
            cells.append(cell)
        cell = rnn.MultiRNNCell(cells)
        start, state = Candidate._start(batch_size, config)
        h, state = tf.nn.dynamic_rnn(cell, x, initial_state=state)
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

    def _start(batch_size, config):
        shape = [2 * config.layer_count, batch_size, config.unit_count]
        start = tf.placeholder_with_default(np.zeros(shape, np.float32),
                                            name='start', shape=shape)
        parts = tf.unstack(start)
        state = []
        for i in range(config.layer_count):
            c, h = parts[2 * i], parts[2 * i + 1]
            state.append(rnn.LSTMStateTuple(c, h))
        return start, tuple(state)


class Reference:
    def __init__(self, x, y, _):
        self.x, self.y, self.y_hat = x, y, x
        self.parameters = []

    def test(self, session, input, future_length):
        x = session.run(input.x)
        _, sample_length, dimension_count = x.shape
        y_hat = np.zeros([future_length, sample_length, dimension_count])
        for i in range(sample_length):
            for j in range(future_length):
                y_hat[j, i, :] = x[0, i, :]
        return _time_travel(x, future_length), y_hat

    def validate(self, session, loss):
        return session.run(loss)


def Learner(config):
    if len(config) > 0:
        return tf.make_template(
            'learner', lambda x, y: Candidate(x, y, config))
    else:
        return tf.make_template(
            'learner', lambda x, y: Reference(x, y, config))

def _time_travel(x, future_length):
    _, sample_length, dimension_count = x.shape
    y = np.empty([future_length, sample_length, dimension_count])
    for i in range(sample_length):
        for j in range(future_length):
            k = i + j + 1
            y[j, i, :] = x[0, k, :] if k < sample_length else 0
    return y
