from .tuner import Hyperband
import unittest


class HyperbandTestCase(unittest.TestCase):
    def test_run(self):
        observed_n = []
        observed_r = []
        observed_c = []
        def _get(n):
            observed_n.append(n)
            return list(range(n))
        def _test(r, c):
            observed_r.append(int(r))
            observed_c.append(len(c))
            return [r * c for c in c]
        tuner = Hyperband()
        tuner.run(_get, _test)
        expected_n = [81, 34, 15, 8, 5]
        expected_r = [1, 3, 9, 27, 81, 3, 9, 27, 81, 9, 27, 81, 27, 81, 81]
        expected_c = [81, 27, 9, 3, 1, 27, 9, 3, 1, 9, 3, 1, 6, 2, 5]
        self.assertEqual(expected_n, observed_n)
        self.assertEqual(expected_r, observed_r)
        self.assertEqual(expected_c, observed_c)
