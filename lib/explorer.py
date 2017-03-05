from hyperband import Hyperband
from input import Input
import numpy as np


class Explorer(Hyperband):
    def __init__(self, config):
        self.input = Input.find(config.input)

    def run(self):
        hyperband = Hyperband()
        hyperband.run(self._get, self._test)

    def _get(self, count):
        return np.arange(count)

    def _test(self, resource, parameters):
        return parameters
