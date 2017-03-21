import numpy as np
import threading

class Random(np.random.RandomState):
    _seed = 0
    _instances = {}
    _lock = threading.Lock()

    def initialize(seed):
        Random._seed = seed

    def get():
        with Random._lock:
            key = threading.get_ident()
            if key not in Random._instances:
                Random._instances[key] = Random()
            return Random._instances[key]

    def __init__(self):
        super(Random, self).__init__(Random._seed)
