#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

import glob
import matplotlib.pyplot as pp

import support

def main(index_path, min_length=0, max_length=50, report_each=1000000):
    def report():
        status = 'Processed: {}, selected: {} ({:.2f}%)'.format(
            trace_count, len(samples), 100 * len(samples) / trace_count)
        support.log(status)
        pp.clf()
        pp.title(status)
        pp.hist(samples, bins=(max_length - min_length))
        pp.pause(1e-3)
    support.log('Index: {}', index_path)
    support.figure()
    samples = []
    trace_count = 0
    with open(index_path, 'r') as file:
        for record in file:
            trace_count += 1
            length = int(record.split(',')[-1])
            if length >= min_length and length <= max_length:
                samples.append(length)
            if trace_count % report_each == 0:
                report()
        report()
    pp.show()

if __name__ == '__main__':
    assert(len(sys.argv) == 2)
    support.loggalize()
    main(sys.argv[1])
