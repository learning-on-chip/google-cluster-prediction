#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.dirname(__file__))

import math, support
import matplotlib.pyplot as pp
import numpy as np

print('Apps:')
print('  Count: %d' % support.count_apps())
print('Users:')
print('  Count: %d' % support.count_users())

data = support.count_user_jobs()
count = len(data)
support.figure()
pp.subplot(1, 2, 1)
pp.semilogy(data)
pp.title('Jobs per user')
pp.ylabel('log(Count)')
pp.xlabel('User')
pp.subplot(1, 2, 2)
data = np.cumsum(np.sort(data)[::-1]) / np.sum(data)
pp.plot(np.arange(1, count + 1) / count, data)
pp.title('Cumulative contribution')
pp.xlabel('User (sorted)')
pp.ylabel('Contribution')

data = support.select_data(app=None, user=None)[:, 0]
mean, variance = np.mean(data), np.var(data)

print('Interarrivals:')
print('  Count: %d' % len(data))
print('  Mean: %.4f ± %.4f' % (mean, math.sqrt(variance)))
print('  Minimum: %e' % np.min(data))
print('  Maximum: %e' % np.max(data))

support.figure()
count = min(len(data), 10000)
pp.plot(data[:count])
pp.title('Interarrivals, {} of {} samples'.format(count, len(data)))

support.figure()
pp.hist(np.log10(data), bins=1000, log=True)
pp.title('Histogram of interarrivals')
pp.xlabel('log(Time)')
pp.ylabel('log(Count)')

pp.show()
