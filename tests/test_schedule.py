#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

from manager import Schedule
import unittest


class TestCase(unittest.TestCase):
    def test_should(self):
        schedule = Schedule([100, 10])
        for i in range(0, 100):
            self.assertFalse(schedule.should(i))
        for i in range(100, 110):
            self.assertTrue(schedule.should(i))
        for i in range(110, 210):
            self.assertFalse(schedule.should(i))
        for i in range(210, 220):
            self.assertTrue(schedule.should(i))


if __name__ == '__main__':
    unittest.main()
