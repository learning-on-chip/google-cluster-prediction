from .config import Config
import numpy as np
import unittest


class ConfigTestCase(unittest.TestCase):
    def test_copy(self):
        config1 = Config({
            'foo': np.zeros(10),
            'bar': {
                'baz': np.zeros(10),
            },
        })
        config2 = config1.copy()
        config2.foo[0] = 42
        config2.bar.baz[0] = 69
        self.assertEqual(config1.foo[0], 0)
        self.assertEqual(config2.foo[0], 42)
        self.assertEqual(config1.bar.baz[0], 0)
        self.assertEqual(config2.bar.baz[0], 69)
