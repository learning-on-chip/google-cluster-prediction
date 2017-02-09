#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

import glob, json
import matplotlib.pyplot as pp
import numpy as np

import support

def main(index_path, min_length=0, max_length=200, report_each=1000000):
    print('Loading the index from "{}"...'.format(index_path))
    with open(index_path, 'r') as file:
        index = json.load(file)['index']
    total = len(index)
    print('Processing {} traces...'.format(total))
    data = []
    count = 0
    support.figure()
    for record in index:
        length = record['length']
        if length < min_length or length > max_length:
            continue
        data.append(length)
        count += 1
        if count % report_each == 0 or count == total:
            pp.clf()
            mean, max = int(np.mean(data)), np.max(data)
            pp.title("Processed {} ({:.2f}%), mean {}, max {}".format(
                count, 100 * count / total, mean, max))
            pp.hist(data, bins=200)
            pp.pause(1e-3)
    pp.show()

if __name__ == '__main__':
    assert(len(sys.argv) == 2)
    main(sys.argv[1])
