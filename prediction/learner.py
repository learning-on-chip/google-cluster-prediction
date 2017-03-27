from . import support
from .baseline import Baseline
from .model import Model
from .teacher import Teacher
import glob
import numpy as np
import os
import re
import tensorflow as tf


class Learner:
    def __init__(self, input, config):
        self.input = input
        self.output = config.output
        graph = tf.Graph()
        with graph.as_default():
            with tf.variable_scope('model'):
                self.model = Model(config.model)
                with tf.variable_scope('state'):
                    self.state = State(self.input.train.sample_count)
            with tf.variable_scope('teacher'):
                self.teacher = Teacher(self.model, config.teacher)
            tf.summary.scalar('train_loss', self.teacher.train_loss)
            self.train_summary = tf.summary.merge_all()
            self.summarer = tf.summary.FileWriter(self.output.path, graph)
            initialize = tf.variables_initializer(
                tf.global_variables(), name='initialize')
            self.checkpoint = Checkpoint(self.output)
        self.session = tf.Session(graph=graph)
        self.session.run(initialize)
        self.checkpoint.load(self.session)
        self.state.load(self.session)
        self.input.on_epoch(self.state)
        support.log(self, 'Output path: {}', self.output.path)
        support.log(self, 'Initial state: step {}, epoch {}, sample {}',
                    self.state.step, self.state.epoch, self.state.sample)
        if self.output.baseline:
            baseline = Baseline(self.input, self.teacher, self.summarer)
            baseline.run()

    def run_backup(self):
        self.state.save(self.session)
        self.checkpoint.save(self.session, self.state)

    def run_test(self):
        errors = self.teacher.test(self.input.test, self._run_test)
        for name in errors:
            for i in range(len(errors[name])):
                tag = 'test_{}_{}'.format(name, i + 1)
                value = tf.Summary.Value(tag=tag, simple_value=errors[name][i])
                self.summarer.add_summary(
                    tf.Summary(value=[value]), self.state.step)
        self.summarer.flush()
        return errors

    def run_train(self, sample_count=1):
        for _ in range(sample_count):
            self._run_train()
            self._increment_time()

    def _increment_time(self):
        self.state.increment_time()
        if self.state.is_new_epoch():
            self.input.on_epoch(self.state)
            support.log(self, 'Current state: step {}, epoch {}, sample {}',
                        self.state.step, self.state.epoch, self.state.sample)

    def _run_test(self, sample, test_length):
        fetch = {
            'y_hat': self.model.y_hat,
            'finish': self.model.finish,
        }
        y_hat = np.empty(
            [sample.shape[0], test_length, self.input.dimension_count])
        for i in range(sample.shape[0]):
            feed = {
                self.model.start: self._zero_start(),
                self.model.x: np.reshape(sample[:(i + 1), :], [1, i + 1, -1]),
            }
            for j in range(test_length):
                result = self.session.run(fetch, feed)
                y_hat[i, j, :] = result['y_hat'][0, -1, :]
                feed[self.model.start] = result['finish']
                feed[self.model.x] = np.reshape(y_hat[i, j, :], [1, 1, -1])
        return y_hat

    def _run_train(self):
        sample = self.input.train.get(self.state.sample)
        feed = {
            self.model.start: self._zero_start(),
            self.model.x: np.reshape(
                sample, [1, -1, self.input.dimension_count]),
            self.model.y: np.reshape(
                support.shift(sample, -1, padding=0),
                [1, -1, self.input.dimension_count]),
        }
        fetch = {
            'step': self.teacher.train_step,
            'loss': self.teacher.train_loss,
            'summary': self.train_summary,
        }
        result = self.session.run(fetch, feed)
        self.summarer.add_summary(result['summary'], self.state.step)
        return result['loss']

    def _zero_start(self):
        return np.zeros(self.model.start.get_shape(), np.float32)


class Checkpoint:
    def __init__(self, output):
        self.saver = tf.train.Saver(max_to_keep=100)
        self.auto = output.get('auto_restore')
        self.path = output.path

    def load(self, session, state=None):
        paths = Checkpoint._load(self.path)
        if len(paths) == 0:
            return
        path = paths[np.max(list(paths.keys()))]
        should = self.auto
        if should is None:
            answer = input('Restore "{}"? '.format(path))
            should = not answer.lower().startswith('n')
        if not should:
            return
        self.saver.restore(session, path)
        support.log(self, 'Restore: {}', path)

    def save(self, session, state):
        path = os.path.join(self.path, 'model-{}'.format(state.step))
        path = self.saver.save(session, path)
        support.log(self, 'Save: {}', path)

    def _load(path):
        paths = {}
        for path in glob.glob(os.path.join(path, 'model-*.meta')):
            step_count = int(re.search('.*model-(.*).meta', path).group(1))
            paths[step_count] = re.sub('.meta$', '', path)
        return paths


class State:
    def __init__(self, sample_count):
        self.current = tf.Variable(
            [0, 0, 0], name='current', dtype=tf.int64, trainable=False)
        self.new = tf.placeholder(tf.int64, shape=3, name='new')
        self.assign_new = self.current.assign(self.new)
        self.step, self.epoch, self.sample = None, None, None
        self.sample_count = sample_count

    def increment_time(self):
        self.step += 1
        self.sample += 1
        if self.sample == self.sample_count:
            self.epoch += 1
            self.sample = 0

    def is_new_epoch(self):
        return self.sample == 0

    def load(self, session):
        state = session.run(self.current)
        self.step, self.epoch, self.sample = state

    def save(self, session):
        feed = {
            self.new: [self.step, self.epoch, self.sample],
        }
        session.run(self.assign_new, feed)
