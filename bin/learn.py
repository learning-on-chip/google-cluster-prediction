#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.dirname(__file__))

import matplotlib.pyplot as pp
import numpy as np
import support
import tensorflow as tf

def assess(f):
    layer_count = 1
    layer_size = 200
    unroll_count = 10

    train_count = 10000
    report_period = 1000
    imagine_count = 1000
    learning_rate = 0.0001

    model_fn = model(layer_count, layer_size, unroll_count)
    stream_fn = stream(f, unroll_count)

    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, [None, unroll_count, 1])
        y = tf.placeholder(tf.float32, [None, 1, 1])
        y_hat, l = model_fn(x, y)

        trainees = tf.trainable_variables()
        gradient = tf.gradients(l, trainees)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.apply_gradients(zip(gradient, trainees))

        initialize = tf.initialize_all_variables()

    with tf.Session(graph=graph) as session:
        session.run(initialize)
        cursor = 0

        print('%10s %10s' % ('Samples', 'Loss'))
        for i in range(train_count):
            x_observed, y_observed, cursor = stream_fn(cursor)
            l_current, _ = session.run([l, train], {x: x_observed, y: y_observed})
            if (i + 1) % report_period != 0: continue
            print('%10d %10.2e' % (i + 1, l_current))

        x_observed[0, :(unroll_count - 1), 0] = x_observed[0, 1:, 0]
        x_observed[0, -1, 0] = y_observed[0]
        Y_imagined = np.zeros([imagine_count, 1])
        for i in range(imagine_count):
            y_imagined = session.run(y_hat, {x: x_observed})
            Y_imagined[i] = y_imagined[0]
            x_observed[0, :(unroll_count - 1), 0] = x_observed[0, 1:, 0]
            x_observed[0, -1, 0] = y_imagined[0]

        Y_observed = np.zeros([imagine_count, 1])
        for i in range(imagine_count // unroll_count):
            l, k = i * unroll_count, (i + 1) * unroll_count
            x_observed, y_predicted, cursor = stream_fn(cursor)
            Y_observed[l:(k - 1)] = np.reshape(x_observed[0, 1:, 0], [unroll_count - 1, 1])
            Y_observed[k - 1] = y_predicted[0]

        compare(Y_observed, Y_imagined, 'Synthesis')

    pp.show()

def stream(f, unroll_count):
    def compute(cursor):
        data = f(np.arange(cursor, cursor + unroll_count + 1))
        x = np.reshape(data[:unroll_count], [1, unroll_count, 1])
        y = np.reshape(data[-1], [1, 1, 1])
        return x, y, cursor + unroll_count

    return compute

def compare(y, y_hat, name):
    support.figure(height=6)
    pp.plot(y)
    pp.plot(y_hat)
    pp.legend(['Reality', name])

def model(layer_count, layer_size, unroll_count):
    def compute(x, y):
        with tf.variable_scope("model") as scope:
            x = [tf.squeeze(x, squeeze_dims=[1]) for x in tf.split(1, unroll_count, x)]
            c = tf.nn.rnn_cell.LSTMCell(layer_size, forget_bias=0.0, state_is_tuple=True)
            c = tf.nn.rnn_cell.MultiRNNCell([c] * layer_count, state_is_tuple=True)
            s = c.zero_state(tf.shape(x[0])[0], tf.float32)
            for i in range(unroll_count):
                h, s = c(x[i], s)
                scope.reuse_variables()
        return regress(h, y)

    def regress(x, y):
        w = tf.Variable(tf.truncated_normal([layer_size, 1]))
        b = tf.Variable(tf.zeros([1]))
        y_hat = tf.squeeze(tf.matmul(x, w) + b, squeeze_dims=[1])
        return y_hat, tf.reduce_sum(tf.square(tf.sub(y_hat, y)))

    return compute

def target(x):
    return np.sin(0.1 * x)

assess(target)
