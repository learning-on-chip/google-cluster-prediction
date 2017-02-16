#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

import matplotlib.pyplot as pp
import numpy as np
import random

import job_events
import support

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

def main_random(data_path):
    support.figure()
    app_count = job_events.count_apps(data_path)
    user_count = job_events.count_users(data_path)
    while True:
        while True:
            app = None
            user = random.randrange(user_count)
            data = job_events.select(data_path, app=app, user=user)[:, 0]
            if len(data) >= 10: break
        display(1e-6 * data, app=app, user=user)
        if input('More? ') == 'no': break

def main_specific(data_path, **arguments):
    support.figure()
    data = 1e-6 * job_events.select(data_path, **arguments)[:, 0]
    display(data, **arguments)
    input()

if __name__ == '__main__':
    assert(len(sys.argv) >= 2)
    if len(sys.argv) == 2:
        main_random(sys.argv[1])
    else:
        main_specific(sys.argv[1], app=None, user=int(sys.argv[2]))
