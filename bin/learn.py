#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

from config import Config
from data import Data
from system import System
import argparse
import json
import numpy as np
import support

def main(config):
    support.loggalize()
    np.random.seed(config.seed)
    System(config).run(Data.find(config.data))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--input')
    parser.add_argument('--output')
    arguments = parser.parse_args()
    config = Config(json.loads(open(arguments.config).read()))
    if arguments.input is not None:
        config.data.input_path = arguments.input
    if arguments.output is not None:
        config.output_path = arguments.output
    if config.get('output_path') is None:
        config.output_path = os.path.join('output', support.format_timestamp())
    main(config)
