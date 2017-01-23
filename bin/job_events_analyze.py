#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

import job_events, math, support
import matplotlib.pyplot as pp
import numpy as np

plot = len(sys.argv) > 1

print('Apps:')
print('  Count: %d' % job_events.count_apps())
print('Users:')
print('  Count: %d' % job_events.count_users())

data = job_events.count_user_jobs()
count = data.shape[0]

if plot:
    support.figure()
    pp.subplot(1, 2, 1)
    pp.semilogy(data[:, 1])
    pp.title('Jobs per user')
    pp.ylabel('log(Count)')
    pp.xlabel('User')

data = data[np.argsort(data[:, 1])[::-1], :]
portion = np.cumsum(data[:, 1]) / np.sum(data[:, 1])
print('  %5s %10s %10s %10s %10s %10s %10s' % ('User', 'Jobs', 'Portion', 'Min',
                                               'Mean', 'Max', 'Spurious'))
for i in range(count):
    trace = np.diff(job_events.select_jobs(user=data[i, 0])[:, 0])
    minute = np.sum(trace == 1) / len(trace)
    trace = 1e-6 * trace
    print('  %5d %10d %10.2f %10.2e %10.2e %10.2e %9.2f%%' % (data[i, 0],
                                                              data[i, 1],
                                                              portion[i],
                                                              np.min(trace),
                                                              np.mean(trace),
                                                              np.max(trace),
                                                              100 * minute))
    if portion[i] > 0.9: break

if plot:
    pp.subplot(1, 2, 2)
    pp.plot(np.arange(1, count + 1) / count, portion)
    pp.title('Cumulative contribution')
    pp.xlabel('User (sorted)')
    pp.ylabel('Contribution')

data = 1e-6 * np.diff(job_events.select_jobs(app=None, user=None)[:, 0])
mean, variance = np.mean(data), np.var(data)

print('Interarrivals:')
print('  Count: %d' % len(data))
print('  Mean: %.4f ± %.4f' % (mean, math.sqrt(variance)))
print('  Minimum: %e' % np.min(data))
print('  Maximum: %e' % np.max(data))

if plot:
    support.figure()
    count = min(len(data), 10000)
    pp.plot(data[:count])
    pp.title('Interarrivals, {} of {} samples'.format(count, len(data)))
    pp.ylabel('Time')

    support.figure()
    pp.hist(np.log10(data), bins=1000, log=True)
    pp.title('Histogram of interarrivals')
    pp.xlabel('log(Time)')
    pp.ylabel('log(Count)')

    pp.show()
