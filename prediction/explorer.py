from . import support
from .hyperband import Hyperband
from .input import Input
from .learner import Learner
import glob
import numpy as np
import os
import re
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
            key = _tokenize_case(case)
            worker = self.workers.get(key)
            if worker is None:
                config = self.learner_config.copy()
                config.output.path = os.path.join(self.output_path, key)
                del config.manager['show_address']
                worker = Worker(config)
                self.workers[key] = worker
            worker.submit(resource)
            workers.append(worker)
        return [worker.collect(resource) for worker in workers]


class Sampler:
    def __init__(self, config):
        self.parameters = config

    def get(self):
        case = {}
        for name in self.parameters:
            case[name] = np.random.choice(self.parameters[name])
        return case


class Worker:
    def __init__(self, config):
        self.learner = Learner(config)
        self.output_path = config.output.path
        self.results = Worker._load(self.output_path)
        self.lock = threading.Lock()
        self.done = threading.Lock()

    def collect(self, resource):
        key = _tokenize_resource(resource)
        with self.done:
            return self.results[key]

    def submit(self, resource):
        key = _tokenize_resource(resource)
        with self.lock:
            if key in self.results:
                return
            self.results[key] = None
        self.done.acquire()
        worker = threading.Thread(target=self._run, args=(resource,),
                                  daemon=True)
        worker.start()

    def _load(path):
        results = {}
        for path in glob.glob(os.path.join(path, 'result-*.txt')):
            key = re.search('.*result-(.*).txt', path).group(1)
            results[key] = float(open(path).read())
            support.log(Worker, 'Result: {}', path)
        return results

    def _save(path, key, result):
        path = os.path.join(path, 'result-{}.txt'.format(key))
        with open(path, 'w') as file:
            file.write('{:.15e}'.format(result))

    def _run(self, resource):
        key = _tokenize_resource(resource)
        result = 0
        Worker._save(self.output_path, key, result)
        with self.lock:
            self.results[key] = result
        self.done.release()


def _tokenize_case(case):
    names = sorted(case.keys())
    chunks = []
    for name in names:
        alias = ''.join([chunk[0] for chunk in name.split('_')])
        value = str(case[name])
        chunks.append('{}={}'.format(alias, value))
    return ','.join(chunks)

def _tokenize_resource(resource):
    return str(int(resource))
