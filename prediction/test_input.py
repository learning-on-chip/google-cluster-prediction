from .input import BaseInput
import numpy as np
import unittest


class DummyInput(BaseInput):
    class Part(BaseInput.Part):
        def __init__(self, sample_count):
            super(DummyInput.Part, self).__init__(np.arange(sample_count))

        def _get(self, sample):
            return self.samples[sample]

    def __init__(self, train_sample_count, test_sample_count):
        super(DummyInput, self).__init__(DummyInput.Part(train_sample_count),
                                         DummyInput.Part(test_sample_count))


class DummyState:
    def __init__(self, epoch):
        self.epoch = epoch


class InputTestCase(unittest.TestCase):
    def test_on_epoch(self):
        input = DummyInput(7, 3)
        for epoch in [0, 1, 2, 3]:
            input.on_epoch(DummyState(epoch))
            np.random.seed(epoch)
            expected_samples = np.arange(7)[np.random.permutation(7)]
            observed_samples = [input.train.get(i) for i in range(7)]
            self.assertTrue((expected_samples == observed_samples).all())
