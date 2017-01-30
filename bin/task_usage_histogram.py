#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

import glob
import matplotlib.pyplot as pp
import numpy as np

import support, task_usage

def main(data_path):
    count = 0
    support.figure()
    data = np.array([], dtype=np.int)
    path_pattern = "{}/**/*.sqlite3".format(data_path)
    for part_path in glob.glob(path_pattern):
        part = task_usage.count_job_task_samples(part_path)
        data = np.append(data, part[:, 2])
        count += 1
        if count % 1000 == 0:
            pp.clf()
            mean, max = int(np.mean(data)), np.max(data)
            pp.title("Processed {}, mean {}, max {}".format(count, mean, max))
            pp.hist(data[data < 200], bins=200)
            pp.pause(1e-3)
    pp.show()

if __name__ == '__main__':
    assert(len(sys.argv) == 2)
    main(sys.argv[1])
