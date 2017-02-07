#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

import glob
import numpy as np

import task_usage

def main(data_path):
    count = 0
    for path in sorted(glob.glob('{}/**/*.sqlite3'.format(data_path))):
        data = task_usage.count_job_task_samples(path)
        print('[{}] Jobs: {:2}, tasks: {:6}, samples: {:10}'.format(
            path, len(np.unique(data[:, 0])), data.shape[0],
            np.sum(data[:, 2])))
        count += 1
    print('Count: {}'.format(count))

if __name__ == '__main__':
    assert(len(sys.argv) == 2)
    main(sys.argv[1])
