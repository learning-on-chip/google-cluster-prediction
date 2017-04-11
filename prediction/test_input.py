from .input import Standard
import numpy as np
import unittest


class StandardTestCase(unittest.TestCase):
    def test_workflow(self, epsilon=1e-10):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        expected_mean, expected_deviation = np.mean(data), np.std(data, ddof=1)
        standard = Standard()
        standard.consume(data[:4])
        standard.consume(data[4:])
        mean, deviation = standard.compute()
        self.assertTrue(np.abs(expected_mean - mean) < epsilon)
        self.assertTrue(np.abs(expected_deviation - deviation) < epsilon)
