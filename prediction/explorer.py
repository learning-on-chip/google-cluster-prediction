from . import support
from .hyperband import Hyperband
from .input import Input
from .learner import Learner
import numpy as np
import os
import threading


class Explorer:
    def __init__(self, config):
        self.input = Input.find(config.input)
        self.sampler = Sampler(config.sampler)
        self.learner_config = config.learner
        self.output_path = config.output.path
        self.workers = {}

    def run(self):
        hyperband = Hyperband()
        hyperband.run(self._get, self._test)

    def _get(self, count):
        support.log(self, 'Generate: {} cases', count)
        return [self.sampler.get() for _ in range(count)]

    def _test(self, resource, cases):
        support.log(self, 'Evaluate: {} cases', len(cases))
        workers = []
        for case in cases:
            key = _serialize(case)
            worker = self.workers.get(key)
            if worker is None:
                config = self.learner_config.copy()
                config.output.path = os.path.join(self.output_path, key)
                del config.manager['show_address']
                worker = Worker(Learner(config))
                self.workers[key] = worker
            worker.submit(resource)
            workers.append(worker)
        return [worker.collect(resource) for worker in workers]


class Sampler:
    def __init__(self, config):
        self.parameters = config

    def get(self):
        parameters = {}
        for name in self.parameters:
            parameters[name] = np.random.choice(self.parameters[name])
        return parameters


class Worker:
    def __init__(self, learner):
        self.learner = learner
        self.results = {}
        self.lock = threading.Lock()

    def collect(self, resource):
        with self.lock:
            return self.results[resource]

    def submit(self, resource):
        with self.lock:
            if resource in self.results:
                return
            self.results[resource] = 0


def _serialize(parameters):
    names = sorted(parameters.keys())
    chunks = []
    for name in names:
        alias = ''.join([chunk[0] for chunk in name.split('_')])
        value = str(parameters[name])
        chunks.append('{}={}'.format(alias, value))
    return ','.join(chunks)
