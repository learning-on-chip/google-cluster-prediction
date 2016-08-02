#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.dirname(__file__))

import logging, support
import matplotlib.pyplot as pp
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn

def assess(data, window=5):
    x, y = support.prepare(data, window, validation=0.1, test=0.1)

    model_fn = model(window, [{'window': window}])
    regressor = learn.TensorFlowEstimator(model_fn=model_fn,
                                          n_classes=0,
                                          verbose=2,
                                          steps=1000,
                                          optimizer='Adagrad',
                                          learning_rate=0.03,
                                          batch_size=10)

    monitor = learn.monitors.ValidationMonitor(x['validation'], y['validation'],
                                               every_n_steps=100)

    regressor.fit(x['train'], y['train'], monitors=[monitor])

    predicted = regressor.predict(x['test'])
    predicted = np.reshape(predicted, y['test'].shape)

    error = np.mean((y['test'] - predicted) ** 2)
    print("Error: %f" % error)

    pp.plot(y['test'])
    pp.plot(predicted)
    pp.legend(['Observed', 'Predicted'])
    pp.show()

def model(window, rnn, dnn=None):
    def create(x, y):
        x = learn.ops.split_squeeze(1, window, x)
        stack = []
        for layer in rnn:
            stack.append(tf.nn.rnn_cell.BasicLSTMCell(layer['window'], state_is_tuple=True))
        stack = tf.nn.rnn_cell.MultiRNNCell(stack, state_is_tuple=True)
        output, _ = tf.nn.rnn(stack, x, dtype=tf.float32)
        output = output[-1]
        if dnn is not None:
            output = tf.ops.dnn(output, dnn['layers'],
                                activation=dnn.get('activation'),
                                dropout=dnn.get('dropout'))
        return learn.models.linear_regression(output, y)
    return create

logging.basicConfig(level=logging.INFO)
data = np.sin(np.linspace(0, 100, 10000))
assess(data)
