#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

import glob, json
import numpy as np

import task_usage

def main(data_path, index_path, report_each=10000):
    print('Looking for data in "{}"...'.format(data_path))
    paths = sorted(glob.glob('{}/**/*.sqlite3'.format(data_path)))
    total = len(paths)
    print('Processing {} databases...'.format(total))
    index = []
    processed = 0
    for path in paths:
        processed += 1
        data = task_usage.count_job_task_samples(path)
        for i in range(data.shape[0]):
            index.append({
                'path': path,
                'job': int(data[i, 0]),
                'task': int(data[i, 1]),
                'length': int(data[i, 2]),
            })
        if processed % report_each == 0 or processed == total:
            print('Processed: {} ({:.2f}%)'.format(
                processed, 100 * processed / total))
    print('Saving into "{}"...'.format(index_path))
    with open(index_path, 'w') as file:
        json.dump({'index': index}, file, indent=4)

if __name__ == '__main__':
    assert(len(sys.argv) == 3)
    main(sys.argv[1], sys.argv[2])
