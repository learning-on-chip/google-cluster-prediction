#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from prediction import Config
from prediction import support
import matplotlib.pyplot as pp
import numpy as np
import socket

def main(config):
    support.loggalize()
    support.log('Address: {}', config.address)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(config.address)
    client = client.makefile(mode='r')
    plots = _prepare(config.dimension_count)
    y_limit = [-1, 1]
    while True:
        row = client.readline().split(',')
        count0 = int(row[0])
        count1 = int(row[1])
        count2 = int(row[2])
        row = [float(number) for number in row[3:]]
        x1 = np.arange(0, count1)
        x2 = np.arange(0, count2)
        y1 = np.reshape(np.array(row[:count1]), [-1, config.dimension_count])
        y2 = np.reshape(np.array(row[count1:]), [-1, config.dimension_count])
        if count1 > 0:
            y_limit[0] = min(y_limit[0], np.min(y1))
            y_limit[1] = max(y_limit[1], np.max(y1))
        if count2 > 0:
            y_limit[0] = min(y_limit[0], np.min(y2))
            y_limit[1] = max(y_limit[1], np.max(y2))
        for i in range(config.dimension_count):
            plots[3 * i + 0].set_xdata(x1)
            plots[3 * i + 0].set_ydata(y1[:, i])
            plots[3 * i + 1].set_xdata(x2)
            plots[3 * i + 1].set_ydata(y2[:, i])
            plots[3 * i + 2].set_xdata([0, count1 - 1])
            pp.subplot(config.dimension_count, 1, i + 1)
            pp.xlim([0, count0 - 1])
            pp.ylim(y_limit)
        pp.pause(1e-1)

def _prepare(dimension_count):
    pp.figure(figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
    plots = []
    for i in range(dimension_count):
        pp.subplot(dimension_count, 1, i + 1)
        plots.append(pp.plot([0, 1], [0, 0], 'b')[0])
        plots.append(pp.plot([0, 1], [0, 0], 'g')[0])
        plots.append(pp.plot([0, 1], [0, 0], 'r')[0])
        pp.legend(['Observed', 'Predicted'])
    pp.pause(1e-3)
    return plots

if __name__ == '__main__':
    assert(len(sys.argv) == 2)
    chunks = sys.argv[1].split(':')
    assert(len(chunks) == 2)
    config = Config({
        'dimension_count': 1,
        'address': (chunks[0], int(chunks[1])),
    })
    main(config)
