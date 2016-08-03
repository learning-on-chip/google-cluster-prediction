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
    report_each = 100
    batch_size = 10
    learning_rate = 0.03

    model_fn = model(layer_count, layer_size, input_size)
    batch_fn = batch(f, time_step, input_size, 1, batch_size)

    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, [batch_size, input_size, 1])
        y = tf.placeholder(tf.float32, [batch_size, 1, 1])

        y_hat, loss = model_fn(x, y)

        trainees = tf.trainable_variables()
        gradient = tf.gradients(loss, trainees)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.apply_gradients(zip(gradient, trainees))

        initialize = tf.initialize_all_variables()

    with tf.Session(graph=graph) as session:
        initialize.run()
        cursor = 0

        print('%10s %10s' % ('Step', 'Loss'))
        for i in range(train_batch_count):
            x_observed, y_observed, cursor = batch_fn(cursor)
            error, _ = session.run([loss, train], {x: x_observed, y: y_observed})
            if (i + 1) % report_each == 0:
                print('%10s %10.2f' % (i + 1, error))

        Y_observed = np.zeros([predict_batch_count * batch_size, 1])
        Y_predicted = np.zeros([predict_batch_count * batch_size, 1])
        for i in range(predict_batch_count):
            x_observed, y_observed, cursor = batch_fn(cursor)
            y_predicted = session.run(y_hat, {x: x_observed})
            j = i * batch_size
            k = j + batch_size
            Y_observed[j:k] = np.reshape(y_observed, [batch_size, 1])
            Y_predicted[j:k] = np.reshape(y_predicted, [batch_size, 1])

    support.figure(height=6)
    pp.plot(Y_observed)
    pp.plot(Y_predicted)
    pp.legend(['Observed', 'Predicted'])
    pp.show()

def model(layer_count, layer_size, input_size):
    def create(x, y):
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

    return create

def batch(f, time_step, input_size, output_size, batch_size):
    def create(cursor):
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

    return create

def target(x):
    return np.sin(x)

assess(target)
