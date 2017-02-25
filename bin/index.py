#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

import glob
import support
import task_usage

def main(data_path, index_path, report_each=10000):
    support.log('Data: {}', data_path)
    paths = glob.glob('{}/**/*.sqlite3'.format(data_path))
    database_count = len(paths)
    support.log('Databases: {}', database_count)
    done_count, trace_count = 0, 0
    with open(index_path, 'w') as file:
        for path in sorted(paths):
            done_count += 1
            data = task_usage.count_job_task_samples(path)
            trace_count += data.shape[0]
            for i in range(data.shape[0]):
                record = [path, data[i, 0], data[i, 1], data[i, 2]]
                file.write(','.join([str(item) for item in record]) + '\n')
            if done_count % report_each == 0 or done_count == database_count:
                support.log('Processed: {} ({:.2f}%), traces: {}',
                            done_count, 100 * done_count / database_count,
                            trace_count)

if __name__ == '__main__':
    assert(len(sys.argv) == 3)
    support.loggalize()
    main(sys.argv[1], sys.argv[2])
