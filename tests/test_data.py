#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

from data import Data
import numpy as np
import unittest


class Part:
    def __init__(self, count):
        self.sample_count = count
        self.samples = np.arange(count)


class State:
    def __init__(self, epoch):
        self.epoch = epoch


class TestCase(unittest.TestCase):
    def test_on_epoch(self):
        data = Data(Part(7), Part(3))
        for epoch in [0, 1, 2, 3]:
            data.on_epoch(State(epoch))
            np.random.seed(epoch)
            samples = np.arange(7)[np.random.permutation(7)]
            self.assertTrue((data.train.samples == samples).all())


if __name__ == '__main__':
    unittest.main()
