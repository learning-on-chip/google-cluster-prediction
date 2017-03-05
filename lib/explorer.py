from data import Data
from hyperband import Hyperband
import numpy as np


class Explorer(Hyperband):
    def __init__(self, config):
        self.data = Data.find(config.data)

    def run(self):
        hyperband = Hyperband()
        hyperband.run(self._get, self._test)

    def _get(self, count):
        return np.arange(count)

    def _test(self, resource, parameters):
        return parameters
