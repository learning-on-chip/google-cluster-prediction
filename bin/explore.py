#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

import matplotlib.pyplot as pp
import numpy as np
import random, support

def display(data, app, user):
    pp.clf()
    pp.subplot(2, 1, 1)
    length = len(data)
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
    pp.pause(1e-3)

support.figure()

if len(sys.argv) > 1:
    app = None
    user = int(sys.argv[1])
    data = 1e-6 * support.select_jobs(app=app, user=user)[:, 0]
    display(data, app=app, user=user)
    input()
else:
    app_count = support.count_apps()
    user_count = support.count_users()
    while True:
        while True:
            app = None
            user = random.randrange(user_count)
            data = support.select_jobs(app=app, user=user)[:, 0]
            if len(data) >= 10: break
        display(1e-6 * data, app=app, user=user)
        if input('More? ') == 'no': break
