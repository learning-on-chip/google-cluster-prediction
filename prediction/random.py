import numpy as np
import threading


class Random(np.random.RandomState):
    _SEED = 0
    _INSTANCES = {}
    _LOCK = threading.Lock()

    def initialize(seed):
        Random._SEED = seed

    def get():
        with Random._LOCK:
            key = threading.get_ident()
            if key not in Random._INSTANCES:
                Random._INSTANCES[key] = Random()
            return Random._INSTANCES[key]

    def get_seed():
        return Random._SEED

    def __init__(self):
        super(Random, self).__init__(Random.get_seed())
