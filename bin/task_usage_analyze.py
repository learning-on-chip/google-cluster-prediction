#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

import glob, json
import matplotlib.pyplot as pp

import support

def main(index_path, min_length=0, max_length=50, report_each=1000000):
    support.figure()
    print('Loading the index from "{}"...'.format(index_path))
    with open(index_path, 'r') as file:
        index = json.load(file)['index']
    total = len(index)
    print('Processing {} traces...'.format(total))
    data = []
    processed = 0
    for trace in index:
        processed += 1
        if trace['length'] >= min_length and trace['length'] <= max_length:
            data.append(trace['length'])
        if processed % report_each == 0 or processed == total:
            pp.clf()
            pp.title("Processed {} ({:.2f}%), selected {} ({:.2f}%)".format(
                processed, 100 * processed / total, len(data),
                100 * len(data) / processed))
            pp.hist(data, bins=(max_length - min_length))
            pp.pause(1e-3)
    pp.show()

if __name__ == '__main__':
    assert(len(sys.argv) == 2)
    main(sys.argv[1])
