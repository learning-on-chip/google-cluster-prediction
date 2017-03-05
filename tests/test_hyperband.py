#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

from hyperband import Hyperband
import numpy as np
import unittest


class TestCase(unittest.TestCase):
    def test_run(self):
        observed_rs = []
        observed_cs = []
        def _get(n):
            return np.random.randint(10, 100, (n,))
        def _test(r, c):
            observed_rs.append(r)
            observed_cs.append(c)
            return r * c
        np.random.seed(0)
        hyperband = Hyperband()
        hyperband.run(_get, _test)
        expected_rs = [1, 3, 9, 27, 81,
                       3, 9, 27, 81,
                       9, 27, 81,
                       27, 81,
                       81]
        expected_cs = [[54, 57, 74, 77, 77, 19, 93, 31, 46, 97, 80, 98, 98, 22,
                        68, 75, 49, 97, 56, 98, 91, 47, 35, 87, 82, 19, 30, 90,
                        79, 89, 57, 74, 92, 98, 59, 39, 29, 29, 24, 49, 42, 75,
                        19, 67, 42, 41, 84, 33, 45, 85, 65, 38, 44, 10, 10, 46,
                        63, 15, 48, 27, 89, 14, 52, 68, 41, 11, 75, 51, 67, 45,
                        21, 56, 92, 10, 24, 63, 22, 52, 94, 85, 78],
                       [10, 10, 10, 11, 14, 15, 19, 19, 19, 21, 22, 22, 24, 24,
                        27, 29, 29, 30, 31, 33, 35, 38, 39, 41, 41, 42, 42],
                       [10, 10, 10, 11, 14, 15, 19, 19, 19],
                       [10, 10, 10],
                       [10],
                       [16, 78, 57, 13, 86, 62, 88, 25, 30, 68, 33, 89, 23, 95,
                        58, 59, 79, 51, 45, 74, 79, 10, 60, 46, 44, 58, 13, 52,
                        87, 31, 83, 10, 20, 53],
                       [10, 10, 13, 13, 16, 20, 23, 25, 30, 31, 33],
                       [10, 10, 13],
                       [10],
                       [68, 33, 69, 12, 72, 45, 77, 92, 56, 30, 91, 60, 37, 24,
                        51],
                       [12, 24, 30, 33, 37],
                       [12],
                       [68, 75, 46, 20, 96, 53, 21, 12],
                       [12, 20],
                       [61, 90, 42, 64, 10]]
        self.assertEqual(expected_rs, [int(r) for r in observed_rs])
        for expected_cs, observed_cs in zip(expected_cs, observed_cs):
            self.assertTrue((expected_cs == observed_cs).all())


if __name__ == '__main__':
    unittest.main()
