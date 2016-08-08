#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.dirname(__file__))

import random, support
import matplotlib.pyplot as pp

support.figure()
app_count = support.count_apps()
user_count = support.count_users()
while True:
    while True:
        app = None
        user = random.randrange(user_count)
        data = support.select_interarrivals(app=app, user=user)
        if len(data) >= 10: break
    pp.clf()
    pp.plot(data)
    pp.title('App {}, User {}, Samples {}'.format(app, user, len(data)))
    pp.pause(1)
    input()
