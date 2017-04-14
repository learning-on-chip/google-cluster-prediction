from . import support
from .checkpoint import Checkpoint
from .learner import Learner
from .teacher import Teacher
import numpy as np
import tensorflow as tf


class Experiment:
    def __init__(self, input, config):
        self.input = input
        self.output = config.output
        learner = tf.make_template(
            'learner', lambda x, y: Learner(x, y, config.learner))
        graph = tf.Graph()
        with graph.as_default():
            shape = [None, None, input.dimension_count]
            x = tf.placeholder(tf.float32, shape, name='x')
            y = tf.placeholder(tf.float32, shape, name='y')
            self.training_learner = learner(x, y)
            self.validation_learner = self.training_learner
            self.test_learner = self.training_learner
            with tf.variable_scope('state'):
                self.state = State()
            with tf.variable_scope('teacher'):
                self.teacher = Teacher(self.training_learner, config.teacher)
            self.summarer = tf.summary.FileWriter(self.output.path, graph)
            initialize = tf.variables_initializer(
                tf.global_variables(), name='initialize')
            self.checkpoint = Checkpoint(self.output)
        self.session = tf.Session(graph=graph)
        self.session.run(initialize)
        self.checkpoint.load(self.session)
        self.state.load(self.session)
        self.input.training.restart(self.state.epoch)
        support.log(self, 'Output path: {}', self.output.path)
        support.log(self, 'Initial step: {}, epoch: {}, sample: {}',
                    self.state.step, self.state.epoch, self.state.sample)

    def run_backup(self):
        self.state.save(self.session)
        self.checkpoint.save(self.session, self.state)

    def run_comparison(self, target, summarize=True):
        errors = getattr(self, 'run_' + target)(summarize=False)
        if summarize:
            self._summarize_static(errors, 'comparison_' + target)
        return errors

    def run_testing(self, summarize=True):
        errors = self.teacher.test(self.input.testing, self._test)
        if summarize:
            self._summarize_dynamic(errors, 'testing')
        return errors

    def run_training(self, summarize=True, sample_count=1):
        for _ in range(sample_count):
            try:
                errors = self.teacher.train(self.input.training, self._train)
                if summarize:
                    self._summarize_dynamic(errors, 'training')
                self.state.increment_time()
            except StopIteration:
                self.state.increment_epoch()
                self.input.training.restart(self.state.epoch)
                support.log(
                    self, 'Current step: {}, epoch: {}, sample: {}',
                    self.state.step, self.state.epoch, self.state.sample)

    def run_validation(self, summarize=True):
        errors = self.teacher.validate(self.input.validation, self._validate)
        if summarize:
            self._summarize_dynamic(errors, 'validation')
        return errors

    def _summarize_dynamic(self, data, name):
        for key in data:
            for i in range(len(data[key])):
                tag = '{}_{}_{}'.format(name, key, i + 1)
                value = tf.Summary.Value(tag=tag, simple_value=data[key][i])
                self.summarer.add_summary(tf.Summary(value=[value]),
                                          self.state.step)
        self.summarer.flush()

    def _summarize_static(self, data, name):
        for key in data:
            tag = '{}_{}'.format(name, key)
            for i in range(len(data[key])):
                value = tf.Summary.Value(tag=tag, simple_value=data[key][i])
                self.summarer.add_summary(tf.Summary(value=[value]), i + 1)
        self.summarer.flush()

    def _train(self, sample):
        feed = {
            self.training_learner.start: np.zeros(
                self.training_learner.start.get_shape(), np.float32),
            self.training_learner.x: np.reshape(
                sample, [1, -1, self.input.dimension_count]),
            self.training_learner.y: np.reshape(
                support.shift(sample, -1, padding=0),
                [1, -1, self.input.dimension_count]),
        }
        fetch = {
            'optimize': self.teacher.optimize,
            'loss': self.teacher.loss,
        }
        return self.session.run(fetch, feed)['loss']

    def _test(self, sample, future_length):
        fetch = {
            'y_hat': self.test_learner.y_hat,
            'finish': self.test_learner.finish,
        }
        sample_length, dimension_count = sample.shape
        y_hat = np.empty([sample_length, future_length, dimension_count])
        for i in range(sample_length):
            feed = {
                self.test_learner.start: np.zeros(
                    self.test_learner.start.get_shape(), np.float32),
                self.test_learner.x: np.reshape(
                    sample[:(i + 1), :], [1, i + 1, -1]),
            }
            for j in range(future_length):
                result = self.session.run(fetch, feed)
                y_hat[i, j, :] = result['y_hat'][0, -1, :]
                feed[self.test_learner.start] = result['finish']
                feed[self.test_learner.x] = y_hat[i:(i + 1), j:(j + 1), :]
        return y_hat

    def _validate(self, sample):
        feed = {
            self.validation_learner.start: np.zeros(
                self.validation_learner.start.get_shape(), np.float32),
            self.validation_learner.x: np.reshape(
                sample, [1, -1, self.input.dimension_count]),
            self.validation_learner.y: np.reshape(
                support.shift(sample, -1, padding=0),
                [1, -1, self.input.dimension_count]),
        }
        fetch = {
            'loss': self.teacher.loss,
        }
        return self.session.run(fetch, feed)['loss']


class State:
    def __init__(self):
        self.current = tf.Variable(
            [0, 0, 0], name='current', dtype=tf.int64, trainable=False)
        self.new = tf.placeholder(tf.int64, shape=3, name='new')
        self.assign_new = self.current.assign(self.new)
        self.step, self.epoch, self.sample = None, None, None

    def increment_epoch(self):
        self.epoch += 1
        self.sample = 0

    def increment_time(self):
        self.step += 1
        self.sample += 1

    def load(self, session):
        state = session.run(self.current)
        self.step, self.epoch, self.sample = state

    def save(self, session):
        feed = {
            self.new: [self.step, self.epoch, self.sample],
        }
        session.run(self.assign_new, feed)
