#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from prediction import database
from prediction import support
import glob

def main(input_path, index_path, report_each=10000):
    support.loggalize()
    support.log('Input path: {}', input_path)
    paths = glob.glob('{}/**/*.sqlite3'.format(input_path))
    database_count = len(paths)
    support.log('Databases: {}', database_count)
    done_count, trace_count = 0, 0
    with open(index_path, 'w') as file:
        for path in sorted(paths):
            done_count += 1
            data = database.count_job_task_samples(path)
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
    main(sys.argv[1], sys.argv[2])
