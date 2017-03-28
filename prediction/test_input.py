from .input import BaseInput
from .random import Random
import numpy as np
import unittest


class DummyInput(BaseInput):
    class Part(BaseInput.Part):
        def __init__(self, sample_count):
            super(DummyInput.Part, self).__init__(np.arange(sample_count))

        def _get(self, sample):
            return self.samples[sample]

    def __init__(self, training_count, validation_count, test_count):
        super(DummyInput, self).__init__(DummyInput.Part(training_count),
                                         DummyInput.Part(validation_count),
                                         DummyInput.Part(test_count))


class DummyState:
    def __init__(self, epoch):
        self.epoch = epoch


class InputTestCase(unittest.TestCase):
    def test_on_epoch(self):
        input = DummyInput(7, 0, 0)
        for epoch in [0, 1, 2, 3]:
            input.on_epoch(DummyState(epoch))
            Random.get().seed(epoch)
            expected_samples = np.arange(7)[Random.get().permutation(7)]
            observed_samples = [input.training.get(i) for i in range(7)]
            self.assertTrue((expected_samples == observed_samples).all())
