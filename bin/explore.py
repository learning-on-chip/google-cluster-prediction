#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

from config import Config
from explorer import Explorer
import argparse
import json
import numpy as np
import support

def main(config):
    support.loggalize()
    np.random.seed(config.seed)
    Explorer(config).run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--input')
    parser.add_argument('--output')
    arguments = parser.parse_args()
    config = Config(json.loads(open(arguments.config).read()))
    if isinstance(config.learner, str):
        config.learner = Config(json.loads(open(config.learner).read()))
    for key in ['input', 'output']:
        if config.get(key) is None:
            config[key] = config.learner.get(key)
    if arguments.input is not None:
        config.input.path = arguments.input
    if arguments.output is not None:
        config.output.path = arguments.output
    if config.output.get('path') is None:
        config.output.path = os.path.join('output', support.format_timestamp())
    main(config)
