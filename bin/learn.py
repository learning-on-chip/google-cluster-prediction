#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.dirname(__file__))

import matplotlib.pyplot as pp
import numpy as np
import support
import tensorflow as tf

def learn(f, train_each, report_each, predict_each, predict_count, total_count,
          repeat_count, assess):

    assert(report_each % train_each == 0)
    assert(predict_each % train_each == 0)

    n = total_count // predict_each
    while n > 0 and n*predict_each + predict_count > total_count: n -= 1
    total_count = n*predict_each + predict_count
    if n == 0: return

    layer_count = 1
    unit_count = 100
    learning_rate = 1e-4
    gradient_norm = 1.0

    model = configure(layer_count, unit_count)
    graph = tf.get_default_graph()
    tf.train.SummaryWriter('log', graph)

    x = tf.placeholder(tf.float32, [1, None, 1], name='x')
    y = tf.placeholder(tf.float32, [1, 1, 1], name='y')
    (y_hat, loss), (start, finish) = model(x, y)

    with tf.variable_scope('optimization'):
        trainees = tf.trainable_variables()
        gradient = tf.gradients(loss, trainees)
        gradient, _ = tf.clip_by_global_norm(gradient, gradient_norm)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.apply_gradients(zip(gradient, trainees))

    initialize = tf.initialize_variables(tf.all_variables(),
                                         name='initialize')

    session = tf.Session(graph=graph)
    session.run(initialize)

    learn_count = np.sum([int(np.prod(t.get_shape())) for t in trainees])
    print('Learning {} parameters...'.format(learn_count))
    for _ in range(repeat_count):
        train_fetches = {'finish': finish, 'train': train, 'loss': loss}
        train_feeds = {
            start: np.zeros(start.get_shape(), dtype=np.float32),
            x: np.zeros([1, train_each, 1], dtype=np.float32),
            y: np.zeros([1, 1, 1], dtype=np.float32),
        }
        predict_fetches = {'finish': finish, 'y_hat': y_hat}
        predict_feeds = {start: None, x: None}

        Y = np.zeros([predict_count, 1])
        Y_hat = np.zeros([predict_count, 1])

        print('%12s %12s %12s' % ('Samples', 'Trainings', 'Loss'))
        for i, j in zip(range(total_count - 1), range(1, total_count)):
            train_feeds[x] = np.roll(train_feeds[x], -1, axis=1)
            train_feeds[x][0, -1, 0] = f(i)
            train_feeds[y][0] = f(j)

            if j % train_each == 0:
                train_results = session.run(train_fetches, train_feeds)
                train_feeds[start] = train_results['finish']

            if j % report_each == 0:
                print('%12d %12d %12.2e' % (j, j // train_each,
                                            train_results['loss']))

            if j % predict_each == 0:
                predict_feeds[start] = train_feeds[start]
                predict_feeds[x] = train_feeds[y]
                for k in range(predict_count):
                    predict_results = session.run(predict_fetches,
                                                  predict_feeds)
                    predict_feeds[start] = predict_results['finish']
                    Y[k], Y_hat[k] = f(j + k), predict_results['y_hat'][0]
                    predict_feeds[x][0] = Y_hat[k]
                assess(Y, Y_hat)

def configure(layer_count, unit_count):
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
            i = tf.shape(h) - np.array([1, 1, unit_count])
            h = tf.reshape(tf.slice(h, i, [1, 1, unit_count]), [1, unit_count])
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

support.figure()
pp.pause(1)

def assess(y, y_hat):
    pp.clf()
    pp.plot(y)
    pp.plot(y_hat)
    pp.legend(['Observed', 'Predicted'])
    pp.pause(1)

if False:
    learn(lambda i: np.sin(0.1 * i),
          train_each=10,
          report_each=int(1e4),
          predict_each=int(1e4),
          predict_count=int(1e3),
          total_count=int(1e5 + 1e3),
          repeat_count=1,
          assess=assess)
else:
    data = support.select_interarrivals(app=None, user=37)
    data = support.normalize(data)
    learn(lambda i: data[i],
          train_each=20,
          report_each=int(1e4),
          predict_each=int(1e4),
          predict_count=10,
          total_count=len(data),
          repeat_count=10,
          assess=assess)

pp.show()
