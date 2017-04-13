from . import support
from .checkpoint import Checkpoint
from .model import Model
from .teacher import Teacher
import numpy as np
import tensorflow as tf


class Learner:
    def __init__(self, input, config):
        self.input = input
        self.output = config.output
        model = tf.make_template(
            'model', lambda x, y: Model(x, y, config.model))
        graph = tf.Graph()
        with graph.as_default():
            shape = [None, None, input.dimension_count]
            x = tf.placeholder(tf.float32, shape, name='x')
            y = tf.placeholder(tf.float32, shape, name='y')
            self.model = model(x, y)
            with tf.variable_scope('state'):
                self.state = State()
            with tf.variable_scope('teacher'):
                self.teacher = Teacher(self.model, config.teacher)
            tf.summary.scalar('loss', self.teacher.loss)
            self.summary = tf.summary.merge_all()
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
        support.log(self, 'Initial state: step {}, epoch {}, sample {}',
                    self.state.step, self.state.epoch, self.state.sample)

    def run_backup(self):
        self.state.save(self.session)
        self.checkpoint.save(self.session, self.state)

    def run_comparison(self, target):
        errors = self._run_assessment(target)
        for key in errors:
            tag = 'comparison_{}_{}'.format(target, key)
            for i in range(len(errors[key])):
                value = tf.Summary.Value(tag=tag, simple_value=errors[key][i])
                self.summarer.add_summary(tf.Summary(value=[value]), i + 1)
        self.summarer.flush()

    def run_testing(self):
        return self._run_assessment('testing')

    def run_training(self, sample_count=1):
        for _ in range(sample_count):
            try:
                self._run_training()
                self.state.increment_time()
            except StopIteration:
                self.state.increment_epoch()
                self.input.training.restart(self.state.epoch)
                support.log(
                    self, 'Current state: step {}, epoch {}, sample {}',
                    self.state.step, self.state.epoch, self.state.sample)

    def run_validation(self):
        return self._run_assessment('validation')

    def _assess(self, sample, future_length):
        fetch = {
            'y_hat': self.model.y_hat,
            'finish': self.model.finish,
        }
        sample_length, dimension_count = sample.shape
        y_hat = np.empty([sample_length, future_length, dimension_count])
        for i in range(sample_length):
            feed = {
                self.model.start: self._zero_start(),
                self.model.x: np.reshape(sample[:(i + 1), :], [1, i + 1, -1]),
            }
            for j in range(future_length):
                result = self.session.run(fetch, feed)
                y_hat[i, j, :] = result['y_hat'][0, -1, :]
                feed[self.model.start] = result['finish']
                feed[self.model.x] = y_hat[i:(i + 1), j:(j + 1), :]
        return y_hat

    def _run_assessment(self, target):
        errors = self.teacher.assess(getattr(self.input, target), self._assess)
        for name in errors:
            for i in range(len(errors[name])):
                tag = '{}_{}_{}'.format(target, name, i + 1)
                value = tf.Summary.Value(tag=tag, simple_value=errors[name][i])
                self.summarer.add_summary(tf.Summary(value=[value]),
                                          self.state.step)
        self.summarer.flush()
        return errors

    def _run_training(self):
        sample = self.input.training.next()
        feed = {
            self.model.start: self._zero_start(),
            self.model.x: np.reshape(
                sample, [1, -1, self.input.dimension_count]),
            self.model.y: np.reshape(
                support.shift(sample, -1, padding=0),
                [1, -1, self.input.dimension_count]),
        }
        fetch = {
            'optimize': self.teacher.optimize,
            'loss': self.teacher.loss,
            'summary': self.summary,
        }
        result = self.session.run(fetch, feed)
        self.summarer.add_summary(result['summary'], self.state.step)
        return result['loss']

    def _zero_start(self):
        return np.zeros(self.model.start.get_shape(), np.float32)


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
