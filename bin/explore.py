#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.dirname(__file__))

import matplotlib.pyplot as pp
import numpy as np
import random, support

support.figure()
app_count = support.count_apps()
user_count = support.count_users()
while True:
    while True:
        app = None
        user = random.randrange(user_count)
        data = support.select_data(app=app, user=user)[:, 0]
        length = len(data)
        if length >= 10: break
    pp.clf()
    pp.subplot(2, 1, 1)
    data = 1e-6 * data
    zeros, ones = np.zeros([length, 1]), np.ones([length, 1])
    y = np.array(list(zip(zeros, ones, zeros))).flatten()
    x = np.array(list(zip(data, data, data))).flatten()
    pp.plot(x, y)
    pp.xlabel('Time')
    pp.xlim([x[0], x[-1]])
    pp.title('Arrivals (app {}, user {}, {} samples)'.format(app, user, length))
    pp.subplot(2, 1, 2)
    y = np.diff(data)
    x = np.arange(0, len(y))
    pp.plot(x, y)
    pp.xlim([x[0], x[-1]])
    pp.ylim([0, max(y)])
    pp.title('Interarrivals')
    pp.ylabel('Time')
    pp.gcf().subplots_adjust(hspace=0.5)
    pp.pause(1)
    input()