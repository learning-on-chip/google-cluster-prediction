#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

import glob
import matplotlib.pyplot as pp
import numpy as np

from task_usage import Database
import support

def main(data_path):
    count = 0
    support.figure()
    data = np.array([], dtype=np.int)
    path_pattern = "{}/**/*.sqlite3".format(data_path)
    for part_path in glob.glob(path_pattern):
        part = Database(part_path).count_job_task_samples()
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
