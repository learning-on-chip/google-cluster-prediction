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
        x = tf.placeholder(tf.float32, [None, None, 1], name='x')
        y = tf.placeholder(tf.float32, [None, 1, 1], name='y')
        y_hat, l = model_fn(x, y)

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
        for i in range(train_count // unroll_count):
            x_observed, y_observed, taken_count = stream_fn(unroll_count, taken_count)
            l_observed, _ = session.run([l, train], {x: x_observed, y: y_observed})
            if taken_count % monitor_period != 0: continue
            print('%12d %12d %12.2e' % (i + 1, taken_count, l_observed))

        Y_observed = np.zeros([imagine_count, 1])
        Y_imagined = np.zeros([imagine_count, 1])
        x_imagined = x_observed
        x_imagined[0, :(unroll_count - 1), 0] = x_imagined[0, 1:, 0]
        x_imagined[0, -1, 0] = y_observed[0]
        for i in range(imagine_count):
            _, y_observed, taken_count = stream_fn(1, taken_count)
            Y_observed[i] = y_observed[0]
            y_imagined = session.run(y_hat, {x: x_imagined})
            Y_imagined[i] = y_imagined[0]
            x_imagined[0, :(unroll_count - 1), 0] = x_imagined[0, 1:, 0]
            x_imagined[0, -1, 0] = y_imagined[0]

        compare(Y_observed, Y_imagined)

    pp.show()

def stream(f):
    def compute(needed_count, taken_count):
        data = f(np.arange(taken_count, taken_count + needed_count + 1))
        x = np.reshape(data[:needed_count], [1, needed_count, 1])
        y = np.reshape(data[-1], [1, 1, 1])
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
            cell = tf.nn.rnn_cell.BasicLSTMCell(unit_count, forget_bias=0.0,
                                                state_is_tuple=True)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * layer_count,
                                               state_is_tuple=True)
            outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
            outputs = tf.reverse(outputs, [False, True, False])
            outputs = tf.slice(outputs, [0, 0, 0], [1, 1, unit_count])
            outputs = tf.reshape(outputs, [1, unit_count])
        return regress(outputs, y)

    def regress(x, y):
        with tf.variable_scope('regression') as scope:
            w = tf.get_variable('w', initializer=tf.truncated_normal([unit_count, 1]))
            b = tf.get_variable('b', [1])
            y_hat = tf.squeeze(tf.matmul(x, w) + b, squeeze_dims=[1])
            loss = tf.reduce_sum(tf.square(tf.sub(y_hat, y)))
        return y_hat, loss

    return compute

def target(x):
    return np.sin(0.1 * x)

assess(target)
