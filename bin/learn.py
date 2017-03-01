#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

from config import Config
from learner import Learner
from manager import Manager
from target import SineWave, TaskUsage
import argparse
import numpy as np
import support

def main(config):
    support.loggalize()
    np.random.seed(config.seed)
    if 'index_path' in config:
        target = TaskUsage(config)
    else:
        target = SineWave(config)
    config.update({
        'dimension_count': target.dimension_count,
    })
    learner = Learner(config)
    manager = Manager(config)
    learner.run(target, manager, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument(
        '--output', default=os.path.join('output', support.format_timestamp()))
    parser.add_argument('--seed', default=0)
    arguments = parser.parse_args()
    config = Config({
        'seed': arguments.seed,
        # Model
        'layer_count': 1,
        'unit_count': 200,
        'cell': {
            'type': 'LSTM',
            'options': {
                'cell_clip': 1.0,
                'forget_bias': 1.0,
                'use_peepholes': True,
            },
        },
        'initializer': {
            'type': 'uniform',
            'options': {
                'minval': -0.01,
                'maxval': 0.01,
            },
        },
        'dropout': {
            'options': {
                'input_keep_prob': 1.0,
                'output_keep_prob': 1.0,
            },
        },
        # Train
        'batch_size': 1,
        'train_fraction': 0.7,
        'gradient_clip': 1.0,
        'learning_rate': 1e-4,
        'epoch_count': 100,
        # Test
        'test_schedule': [1000, 1],
        'test_length': 10,
        # Backup
        'backup_schedule': [10000, 1],
        'backup_path': os.path.join(arguments.output, 'backup'),
        # Show
        'show_schedule': [1000, 1],
        'show_address': ('0.0.0.0', 4242),
        # Summay
        'summary_path': arguments.output,
    })
    if arguments.input is not None:
        config.update({
            # Target
            'index_path': arguments.input,
            'standard_count': 1000,
            'max_sample_count': 1000000,
            'min_sample_length': 5,
            'max_sample_length': 50,
        })
    main(config)
