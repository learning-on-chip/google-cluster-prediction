#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

import math
import matplotlib.pyplot as pp
import numpy as np

import job_events, support

def main(data_path, plot):
    print('Apps:')
    print('  Count: %d' % job_events.count_apps(data_path))
    print('Users:')
    print('  Count: %d' % job_events.count_users(data_path))

    data = job_events.count_user_jobs(data_path)
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
    print('  %5s %10s %10s %10s %10s %10s %10s' % (
        'User', 'Jobs', 'Portion', 'Min', 'Mean', 'Max', 'Spurious'))
    for i in range(count):
        trace = job_events.select_jobs(data_path, user=data[i, 0])
        trace = np.diff(trace[:, 0])
        minute = np.sum(trace == 1) / len(trace)
        trace = 1e-6 * trace
        print('  %5d %10d %10.2f %10.2e %10.2e %10.2e %9.2f%%' % (
            data[i, 0], data[i, 1], portion[i], np.min(trace),
            np.mean(trace), np.max(trace), 100 * minute))
        if portion[i] > 0.9: break

    if plot:
        pp.subplot(1, 2, 2)
        pp.plot(np.arange(1, count + 1) / count, portion)
        pp.title('Cumulative contribution')
        pp.xlabel('User (sorted)')
        pp.ylabel('Contribution')

    data = job_events.select_jobs(data_path, app=None, user=None)
    data = 1e-6 * np.diff(data[:, 0])
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

if __name__ == '__main__':
    assert(len(sys.argv) >= 2)
    main(sys.argv[1], len(sys.argv) > 2)
