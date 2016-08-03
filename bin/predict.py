#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.dirname(__file__))

import matplotlib.pyplot as pp
import numpy as np
import support
import tensorflow as tf

def assess(f):
    time_step = 0.1

    input_size = 5
    layer_count = 1
    layer_size = 20

    train_batch_count = 1000
    predict_batch_count = 100
    imagine_batch_count = 10
    report_each = 100
    batch_size = 10
    start_learning_rate = 0.03
    learning_rate_decay = 0.999

    decay_fn = decay(start_learning_rate, learning_rate_decay)
    model_fn = model(layer_count, layer_size, input_size)
    batch_fn = batch(f, time_step, input_size, 1, batch_size)

    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, [None, input_size, 1])
        y = tf.placeholder(tf.float32, [None, 1, 1])
        r = tf.Variable(0.0, trainable=False)

        y_hat, l = model_fn(x, y)

        trainees = tf.trainable_variables()
        gradient = tf.gradients(l, trainees)

        optimizer = tf.train.AdamOptimizer(r)
        train = optimizer.apply_gradients(zip(gradient, trainees))

        initialize = tf.initialize_all_variables()

    with tf.Session(graph=graph) as session:
        initialize.run()
        cursor = 0

        print('%10s %10s %10s' % ('Step', 'Rate', 'Loss'))
        for i in range(train_batch_count):
            x_observed, y_observed, cursor = batch_fn(cursor)
            r_current = decay_fn(i)
            l_current, _ = session.run([l, train], {
                r: r_current,
                x: x_observed,
                y: y_observed,
            })
            if (i + 1) % report_each == 0:
                print('%10s %10.2e %10.2f' % (i + 1, r_current, l_current))

        Y_observed = np.zeros([predict_batch_count * batch_size, 1])
        Y_predicted = np.zeros([predict_batch_count * batch_size, 1])
        for i in range(predict_batch_count):
            j, k = i * batch_size, (i + 1) * batch_size
            x_observed, y_observed, cursor = batch_fn(cursor)
            Y_observed[j:k] = np.reshape(y_observed, [batch_size, 1])
            y_predicted = session.run(y_hat, {x: x_observed})
            Y_predicted[j:k] = np.reshape(y_predicted, [batch_size, 1])
        compare(Y_observed, Y_predicted)

        x_observed = np.reshape(Y_observed[-input_size:], [1, input_size, 1])
        Y_observed = np.zeros([imagine_batch_count * batch_size, 1])
        Y_imagined = np.zeros([imagine_batch_count * batch_size, 1])
        for i in range(imagine_batch_count):
            j, k = i * batch_size, (i + 1) * batch_size
            _, y_observed, cursor = batch_fn(cursor)
            Y_observed[j:k] = np.reshape(y_observed, [batch_size, 1])
            for l in range(batch_size):
                y_predicted = session.run(y_hat, {x: x_observed})
                Y_imagined[j + l] = y_predicted[-1]
                x_observed[0, 0:(input_size - 1), 0] = x_observed[0, 1:, 0]
                x_observed[0, -1, 0 ] = y_predicted[-1]
        compare(Y_observed, Y_imagined, name='Imagined')

    pp.show()

def batch(f, time_step, input_size, output_size, batch_size):
    def compute(cursor):
        indices = np.arange(cursor, cursor + input_size + batch_size - 1 + output_size)
        data = f(time_step * indices)
        x = np.zeros([batch_size, input_size, 1], dtype=np.float32)
        y = np.zeros([batch_size, output_size, 1], dtype=np.float32)
        for i in range(batch_size):
            j = i + input_size
            k = j + output_size
            x[i, :, 0] = data[i:j]
            y[i, :, 0] = data[j:k]
        return x, y, cursor + batch_size

    return compute

def compare(y, y_hat, name='Predicted'):
    support.figure(height=6)
    pp.plot(y)
    pp.plot(y_hat)
    pp.legend(['Observed', name])

def decay(start, rate):
    def compute(i):
        return start * (rate ** i)

    return compute

def model(layer_count, layer_size, input_size):
    def compute(x, y):
        cell = tf.nn.rnn_cell.BasicLSTMCell(layer_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * layer_count, state_is_tuple=True)
        x = [tf.squeeze(x, squeeze_dims=[1]) for x in tf.split(1, input_size, x)]
        h, _ = tf.nn.rnn(cell, x, dtype=tf.float32)
        return regress(h[-1], y)

    def regress(x, y):
        w, b = tf.Variable(tf.zeros([x.get_shape()[1], 1])), tf.Variable(0.0)
        y_hat = tf.squeeze(tf.matmul(x, w) + b, squeeze_dims=[1])
        loss = tf.reduce_sum(tf.square(tf.sub(y_hat, y)))
        return y_hat, loss

    return compute

def target(x):
    return np.sin(x)

assess(target)
