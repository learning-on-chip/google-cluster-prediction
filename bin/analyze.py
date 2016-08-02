#!/usr/bin/env python3

import math, sqlite3
import matplotlib.pyplot as pp
import numpy as np

def display(data, limit=10000):
    limit = len(data) if limit is None else limit
    figure()
    pp.semilogy(data[:limit])
    pp.title('Interarrivals, %d of %d samples' % (limit, len(data)))

def figure():
    pp.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

def histogram(data):
    figure()
    pp.hist(np.log10(data), bins=1000, log=True)
    pp.title('Histogram of interarrivals')
    pp.xlabel('log(time)')
    pp.ylabel('log(count)')

def read(path='tests/fixtures/google.sqlite3'):
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    cursor.execute('SELECT time FROM arrivals ORDER BY time')
    data = np.diff(np.array([row[0] for row in cursor]))
    connection.close()
    return data

def summarize(data):
    mean, variance = np.mean(data), np.var(data)
    print('Samples: %d' % len(data))
    print('Mean: %.4f ± %.4f' % (mean, math.sqrt(variance)))
    print('Minimum: %e' % np.min(data))
    print('Maximum: %e' % np.max(data))

data = read()
summarize(data)
histogram(data)
display(data[data > 1e-5])

pp.show()
