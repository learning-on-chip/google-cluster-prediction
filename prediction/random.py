import numpy as np
import tensorflow as tf
import threading

class Random(np.random.RandomState):
    _SEED = 0
    _INSTANCES = {}
    _LOCK = threading.Lock()

    def initialize(seed):
        tf.set_random_seed(seed)
        Random._SEED = seed

    def get():
        with Random._LOCK:
            key = threading.get_ident()
            if key not in Random._INSTANCES:
                Random._INSTANCES[key] = Random()
            return Random._INSTANCES[key]

    def __init__(self):
        super(Random, self).__init__(Random._SEED)
