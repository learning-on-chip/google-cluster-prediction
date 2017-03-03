#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

from trainer import State
import data
import numpy as np
import unittest


class Data(data.Data):
    class Part(data.Data.Part):
        def __init__(self, sample_count):
            super(Data.Part, self).__init__(np.arange(sample_count))

        def _get(self, sample):
            return self.samples[sample]

    def __init__(self, train_sample_count, test_sample_count):
        super(Data, self).__init__(Data.Part(train_sample_count),
                                   Data.Part(test_sample_count))


class TestCase(unittest.TestCase):
    def test_on_epoch(self):
        data = Data(7, 3)
        for epoch in [0, 1, 2, 3]:
            data.on_epoch(State(None, epoch, None))
            np.random.seed(epoch)
            expected_samples = np.arange(7)[np.random.permutation(7)]
            observed_samples = [data.train.get(i) for i in range(7)]
            self.assertTrue((expected_samples == observed_samples).all())


if __name__ == '__main__':
    unittest.main()
