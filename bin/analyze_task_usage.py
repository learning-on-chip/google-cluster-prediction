#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

import glob

def main(data_path):
    count = 0
    for part_path in glob.glob('{}/**/*.sqlite3'.format(data_path)):
        print(part_path)
        count += 1
    print('Total: {}'.format(count))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception('expected an argument')
    main(sys.argv[1])
