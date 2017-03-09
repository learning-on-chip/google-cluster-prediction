from .tuner import Hyperband
import unittest


class HyperbandTestCase(unittest.TestCase):
    def test_run(self):
        observed_ns = []
        observed_rs = []
        observed_cs = []
        def _get(n):
            observed_ns.append(n)
            return list(range(n))
        def _test(r, c):
            observed_rs.append(int(r))
            observed_cs.append(len(c))
            return [r * c for c in c]
        tuner = Hyperband()
        tuner.run(_get, _test)
        expected_ns = [81, 34, 15, 8, 5]
        expected_rs = [1, 3, 9, 27, 81, 3, 9, 27, 81, 9, 27, 81, 27, 81, 81]
        expected_cs = [81, 27, 9, 3, 1, 27, 9, 3, 1, 9, 3, 1, 6, 2, 5]
        self.assertEqual(expected_ns, observed_ns)
        self.assertEqual(expected_rs, observed_rs)
        self.assertEqual(expected_rs, observed_rs)
