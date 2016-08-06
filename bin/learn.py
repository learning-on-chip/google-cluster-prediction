#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.dirname(__file__))

import matplotlib.pyplot as pp
import numpy as np
import support
import tensorflow as tf

def assess(f):
    layer_count = 1
    unit_count = 100
    unroll_count = 10
    learning_rate = 1e-4
    train_count = int(1e5)
    monitor_period = int(1e4)
    imagine_count = int(1e3)

    stream_fn = stream(f)
    model_fn = model(layer_count, unit_count)

    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, [1, None, 1], name='x')
        y = tf.placeholder(tf.float32, [1, 1, 1], name='y')
        (y_hat, loss), (start, finish) = model_fn(x, y)

        with tf.variable_scope('optimization'):
            trainees = tf.trainable_variables()
            gradient = tf.gradients(loss, trainees)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train = optimizer.apply_gradients(zip(gradient, trainees))

        initialize = tf.initialize_variables(tf.all_variables(),
                                             name='initialize')

    with tf.Session(graph=graph) as session:
        tf.train.SummaryWriter('log', graph)
        session.run(initialize)
        taken_count = 0

        learn_count = np.sum([int(np.prod(t.get_shape())) for t in trainees])
        print('Learning {} parameters...'.format(learn_count))
        print('%12s %12s %12s' % ('Iterations', 'Samples', 'Loss'))
        fetches = {'finish': finish, 'train': train, 'loss': loss}
        feeds = {start: np.zeros(start.get_shape(), dtype=np.float32)}
        for i in range(train_count // unroll_count):
            feeds[x], feeds[y], taken_count = stream_fn(unroll_count, taken_count)
            results = session.run(fetches, feeds)
            feeds[start] = results['finish']
            if taken_count % monitor_period != 0: continue
            print('%12d %12d %12.2e' % (i + 1, taken_count, results['loss']))

        Y_observed = np.zeros([imagine_count, 1])
        Y_imagined = np.zeros([imagine_count, 1])
        fetches = {'finish': finish, 'y_hat': y_hat}
        feeds = {start: feeds[start], x: feeds[y]}
        for i in range(imagine_count):
            _, y_observed, taken_count = stream_fn(1, taken_count)
            Y_observed[i] = y_observed[0]
            results = session.run(fetches, feeds)
            feeds[start] = results['finish']
            Y_imagined[i] = feeds[x][0] = results['y_hat'][0]

        compare(Y_observed, Y_imagined)

    pp.show()

def stream(f):
    def compute(needed_count, taken_count):
        data = f(np.arange(taken_count, taken_count + needed_count + 1))
        x = np.reshape(data[:needed_count], [1, needed_count, 1])
        y = np.reshape(data[needed_count], [1, 1, 1])
        return x, y, taken_count + needed_count

    return compute

def compare(y, y_hat):
    support.figure(height=6)
    pp.plot(y)
    pp.plot(y_hat)
    pp.legend(['Observed', 'Imagined'])

def model(layer_count, unit_count):
    def compute(x, y):
        with tf.variable_scope('network') as scope:
            initializer = tf.random_uniform_initializer(-0.05, 0.05)
            cell = tf.nn.rnn_cell.LSTMCell(unit_count, forget_bias=0.0,
                                           initializer=initializer,
                                           state_is_tuple=True)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * layer_count,
                                               state_is_tuple=True)
            start, state = initialize()
            h, state = tf.nn.dynamic_rnn(cell, x, initial_state=state,
                                         parallel_iterations=1)
            finish = finalize(state)
            h = tf.reverse(h, [False, True, False])
            h = tf.slice(h, [0, 0, 0], [1, 1, unit_count])
            h = tf.reshape(h, [1, unit_count])
        return regress(h, y), (start, finish)

    def finalize(state):
        parts = []
        for i in range(layer_count):
            parts.append(state[i].c)
            parts.append(state[i].h)
        return tf.pack(parts, name='finish')

    def initialize():
        start = tf.placeholder(tf.float32, [2 * layer_count, 1, unit_count],
                               name='start')
        parts = tf.unpack(start)
        state = []
        for i in range(layer_count):
            c, h = parts[2 * i], parts[2*i + 1]
            state.append(tf.nn.rnn_cell.LSTMStateTuple(c, h))
        return start, state

    def regress(x, y):
        with tf.variable_scope('regression') as scope:
            initializer = tf.truncated_normal([unit_count, 1], stddev=0.5)
            w = tf.get_variable('w', initializer=initializer)
            b = tf.get_variable('b', [1])
            y_hat = tf.squeeze(tf.matmul(x, w) + b, squeeze_dims=[1])
            loss = tf.reduce_sum(tf.square(tf.sub(y_hat, y)))
        return y_hat, loss

    return compute

def target(x):
    return np.sin(0.1 * x)

assess(target)
