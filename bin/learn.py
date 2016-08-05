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

    learning_rate = 1e-4
    train_count = int(1e5)
    monitor_period = int(1e4)
    imagine_count = int(1e3)

    stream_fn = stream(f)
    model_fn = model(layer_count, unit_count)

    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, [1, 1, 1], name='x')
        y = tf.placeholder(tf.float32, [1, 1, 1], name='y')
        (y_hat, l), updates = model_fn(x, y)

        with tf.variable_scope('optimization'):
            trainees = tf.trainable_variables()
            gradient = tf.gradients(l, trainees)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train = optimizer.apply_gradients(zip(gradient, trainees))

        initialize = tf.initialize_variables(tf.all_variables(),
                                             name='initialize')

    with tf.Session(graph=graph) as session:
        tf.train.SummaryWriter('log', graph)
        session.run(initialize)
        taken_count = 0

        parameter_count = np.sum([int(np.prod(trainee.get_shape())) for trainee in trainees])
        print('Learning {} parameters...'.format(parameter_count))
        print('%12s %12s %12s' % ('Iterations', 'Samples', 'Loss'))
        for i in range(train_count):
            x_observed, y_observed, taken_count = stream_fn(taken_count)
            results = session.run([l, train] + updates, {x: x_observed, y: y_observed})
            if taken_count % monitor_period != 0: continue
            print('%12d %12d %12.2e' % (i + 1, taken_count, results[0]))

        Y_observed = np.zeros([imagine_count, 1])
        Y_imagined = np.zeros([imagine_count, 1])
        x_imagined = y_observed
        for i in range(imagine_count):
            _, y_observed, taken_count = stream_fn(taken_count)
            Y_observed[i] = y_observed[0]
            results = session.run([y_hat] + updates, {x: x_imagined})
            Y_imagined[i] = results[0][0]
            x_imagined[0] = results[0][0]

        compare(Y_observed, Y_imagined)

    pp.show()

def stream(f):
    def compute(taken_count):
        data = f(np.arange(taken_count, taken_count + 1 + 1))
        x = np.reshape(data[0], [1, 1, 1])
        y = np.reshape(data[1], [1, 1, 1])
        return x, y, taken_count + 1

    return compute

def compare(y, y_hat):
    support.figure(height=6)
    pp.plot(y)
    pp.plot(y_hat)
    pp.legend(['Observed', 'Imagined'])

def model(layer_count, unit_count):
    def compute(x, y):
        with tf.variable_scope('network') as scope:
            initializer = tf.random_uniform_initializer(-0.5, 0.5)
            cell = tf.nn.rnn_cell.LSTMCell(unit_count, forget_bias=0.0,
                                           initializer=initializer,
                                           state_is_tuple=True)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * layer_count,
                                               state_is_tuple=True)
            start = initiate()
            h, finish = cell(tf.squeeze(x, squeeze_dims=[1]), start)
            updates = shortcut(start, finish)
        return regress(h, y), updates

    def initiate():
        state = []
        for i in range(layer_count):
            c = tf.get_variable('c{}0'.format(i + 1),
                                initializer=tf.zeros([1, unit_count]))
            h = tf.get_variable('h{}0'.format(i + 1),
                                initializer=tf.zeros([1, unit_count]))
            state.append(tf.nn.rnn_cell.LSTMStateTuple(c, h))
        return state

    def regress(x, y):
        with tf.variable_scope('regression') as scope:
            w = tf.get_variable('w', initializer=tf.truncated_normal([unit_count, 1]))
            b = tf.get_variable('b', [1])
            y_hat = tf.squeeze(tf.matmul(x, w) + b, squeeze_dims=[1])
            loss = tf.reduce_sum(tf.square(tf.sub(y_hat, y)))
        return y_hat, loss

    def shortcut(start, finish):
        updates = []
        for i in range(layer_count):
            updates.append(tf.assign(start[i].c, finish[i].c,
                                     name="c{}0-update".format(i + 1)))
            updates.append(tf.assign(start[i].h, finish[i].h,
                                     name="h{}0-update".format(i + 1)))
        return updates

    return compute

def target(x):
    return np.sin(0.1 * x)

assess(target)
