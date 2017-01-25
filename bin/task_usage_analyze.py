#!/usr/bin/env python3

import glob

def main(data_path):
    count = len(glob.glob('{}/**/*.sqlite3'.format(data_path)))
    print('Count: {}'.format(count))

if __name__ == '__main__':
    assert(len(sys.argv) == 2)
    main(sys.argv[1])
