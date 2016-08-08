#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.dirname(__file__))

import math, support
import matplotlib.pyplot as pp
import numpy as np

def display(data):
    support.figure()
    pp.plot(data)
    pp.title('Interarrivals')

def histogram(data):
    support.figure()
    pp.hist(np.log10(data), bins=1000, log=True)
    pp.title('Histogram of interarrivals')
    pp.xlabel('log(time)')
    pp.ylabel('log(count)')

print('Apps:')
print('  Count: %d' % support.count_apps())
print('Users:')
print('  Count: %d' % support.count_users())

data = support.select_interarrivals(app=None, user=None)
mean, variance = np.mean(data), np.var(data)
print('Interarrivals:')
print('  Count: %d' % len(data))
print('  Mean: %.4f ± %.4f' % (mean, math.sqrt(variance)))
print('  Minimum: %e' % np.min(data))
print('  Maximum: %e' % np.max(data))

histogram(data)
display(data)

pp.show()
