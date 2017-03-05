from .manager import Schedule
import unittest


class ScheduleTestCase(unittest.TestCase):
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
