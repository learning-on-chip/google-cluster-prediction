#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

from config import Config
from learner import Learner
from manager import Manager
from target import Target
import argparse
import json
import numpy as np
import support

def main(config):
    support.loggalize()
    np.random.seed(config.seed)
    target = Target.create(config.target)
    config.model.dimension_count = target.dimension_count
    learner = Learner(config)
    manager = Manager(config)
    learner.run(target, manager, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--input')
    parser.add_argument('--output')
    arguments = parser.parse_args()
    config = Config(json.loads(open(arguments.config).read()))
    if arguments.input is not None:
        config.target.input_path = arguments.input
    if arguments.output is not None:
        config.output_path = arguments.output
    if 'output_path' not in config or config.output_path is None:
        config.output_path = os.path.join('output', support.format_timestamp())
    main(config)
