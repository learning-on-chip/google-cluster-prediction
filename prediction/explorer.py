from . import support
from . import tuner
from .learner import Learner
from .random import Random
import glob
import numpy as np
import os
import re
import threading


class Explorer:
    def __init__(self, input, config):
        self.input = input
        self.config = config.learner
        self.first = True
        self.tuner = getattr(tuner, config.tuner.name)
        self.tuner = self.tuner(**config.tuner.options)
        self.resource_scale = config.max_iteration_count / self.tuner.resource
        self.sampler = Sampler(config.sampler)
        self.semaphore = threading.BoundedSemaphore(config.concurrent_count)
        self.agents = {}

    def run(self):
        self.tuner.run(self._get, self._test)

    def _configure(self, case):
        key = _tokenize(case)
        config = self.config.copy()
        config.output.baseline = self.first
        config.output.auto_restore = True
        config.output.path = os.path.join(config.output.path, key)
        for name in case:
            _adjust(config, name, case[name])
        return config

    def _get(self, count):
        support.log(self, 'Generate: {} cases', count)
        return [self.sampler.get() for _ in range(count)]

    def _test(self, resource, cases):
        iteration_count = int(self.resource_scale * resource)
        support.log(self, 'Evaluate: {} cases, up to {} iterations',
                    len(cases), iteration_count)
        agents = []
        for case in cases:
            key = _tokenize(case)
            agent = self.agents.get(key)
            if agent is None:
                config = self._configure(case)
                learner = Learner(self.input.copy(), config)
                agent = Agent(learner, self.semaphore, config)
                self.agents[key] = agent
                self.first = False
            agent.submit(iteration_count)
            agents.append(agent)
        return [agent.collect(iteration_count) for agent in agents]


class Agent:
    def __init__(self, learner, semaphore, config):
        self.learner = learner
        self.semaphore = semaphore
        self.scores = Agent._load(config.output.path)
        self.output_path = config.output.path
        self.lock = threading.Lock()
        self.done = threading.Lock()

    def collect(self, iteration_count):
        with self.done:
            return self.scores[iteration_count]

    def submit(self, iteration_count):
        with self.lock:
            if iteration_count in self.scores:
                return
            self.scores[iteration_count] = None
        self.done.acquire()
        worker = threading.Thread(target=self._run, args=(iteration_count,),
                                  daemon=True)
        worker.start()

    def _load(path):
        scores = {}
        for path in glob.glob(os.path.join(path, 'score-*.txt')):
            iteration_count = re.search('.*score-(.*).txt', path).group(1)
            iteration_count = int(iteration_count)
            scores[iteration_count] = float(open(path).read())
            support.log(Agent, 'Score: {}', path)
        return scores

    def _save(path, iteration_count, score):
        path = os.path.join(path, 'score-{}.txt'.format(iteration_count))
        with open(path, 'w') as file:
            file.write('{:.15e}'.format(score))

    def _run(self, iteration_count):
        with self.semaphore:
            with self.lock:
                last_iteration_count = 0
                for key in self.scores:
                    if self.scores[key] is None:
                        continue
                    if key > last_iteration_count:
                        last_iteration_count = key
            assert(last_iteration_count < iteration_count)
            support.log(
                self, 'Learn: start at iteration {}, stop at iteration {}',
                last_iteration_count, iteration_count)
            self.learner.run_train(iteration_count - last_iteration_count)
            error = self.learner.run_test()['MNRMSE']
            decay = np.reshape(np.exp(-np.arange(len(error))), error.shape)
            score = np.sum(error * decay)
            Agent._save(self.output_path, iteration_count, score)
            self.learner.run_backup()
            with self.lock:
                self.scores[iteration_count] = score
            support.log(self, 'Learn: stop at iteration {}, score {}',
                        iteration_count, score)
            self.done.release()


class Sampler:
    def __init__(self, config):
        self.parameters = config

    def get(self):
        case = {}
        for name in sorted(self.parameters.keys()):
            case[name] = Random.get().choice(self.parameters[name])
        return case


def _adjust(config, name, value):
    if name == 'layer_count':
        config.model.layer_count = value
    elif name == 'unit_count':
        config.model.unit_count = value
    elif name == 'learning_rate':
        config.teacher.optimizer.options.learning_rate = value
    else:
        assert(False)

def _tokenize(case):
    names = sorted(case.keys())
    chunks = []
    for name in names:
        alias = ''.join([chunk[0] for chunk in name.split('_')])
        value = str(case[name])
        chunks.append('{}={}'.format(alias, value))
    return ','.join(chunks)
