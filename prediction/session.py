from . import support
from .learner import Learner
from .saver import Saver
from .teacher import Tester
from .teacher import Trainer
from .teacher import Validator
import numpy as np
import tensorflow as tf


class Session:
    def __init__(self, input, learner, config):
        support.log(self, 'Output path: {}', config.output.path)
        self.output = config.output
        graph = tf.Graph()
        with graph.as_default():
            with tf.variable_scope('input'):
                self.input = input()
            self.trainee = learner(self.input.training.x,
                                   self.input.training.y)
            with tf.variable_scope('trainer'):
                self.trainer = Trainer(self.trainee, config.teacher)
            self.validee = learner(self.input.validation.x,
                                   self.input.validation.y)
            with tf.variable_scope('validator'):
                self.validator = Validator(self.validee, config.teacher)
            self.testee = learner(self.input.testing.x, self.input.testing.y)
            with tf.variable_scope('tester'):
                self.tester = Tester(self.testee, config.teacher)
            with tf.variable_scope('state'):
                self.state = State()
            self.saver = Saver(self.output, name='saver')
        with graph.as_default():
            self.backend = tf.Session()
            self.backend.run(tf.variables_initializer(
                tf.global_variables(), name='initialize'))
        self.summarer = tf.summary.FileWriter(self.output.path, graph)
        self.saver.load(self.backend)
        self.state.load(self.backend)
        self.input.training.offset(self.backend, self.state.step)

    def run_comparison(self, target):
        errors = getattr(self, 'run_' + target)(summarize=False)
        support.summarize_static(self.summarer, errors, 'comparison_' + target)

    def run_saving(self):
        self.state.save(self.backend)
        self.saver.save(self.backend, self.state)

    def run_testing(self, summarize=True):
        errors = self.tester.run(
            self.input.testing, self.backend,
            lambda *arguments: self.testee.test(self.backend, *arguments))
        if summarize:
            support.summarize_dynamic(
                self.summarer, self.state, errors, 'testing')
        return errors

    def run_training(self, sample_count=1, summarize=True):
        errors = self.trainer.run(
            self.input.training, self.backend, sample_count,
            lambda *arguments: self.trainee.train(
                self.backend, self.trainer.optimize, self.trainer.loss,
                *arguments))
        self.state.advance(sample_count)
        if summarize:
            support.summarize_dynamic(
                self.summarer, self.state, errors, 'training')
        return errors

    def run_validation(self, summarize=True):
        errors = self.validator.run(
            self.input.validation, self.backend,
            lambda *arguments: self.validee.validate(
                self.backend, self.validator.loss, *arguments))
        if summarize:
            support.summarize_dynamic(
                self.summarer, self.state, errors, 'validation')
        return errors


class State:
    def __init__(self, report_each=10000):
        self.report_each = report_each
        state = np.zeros(1, dtype=np.int64)
        self.current = tf.Variable(
            state, name='current', dtype=tf.int64, trainable=False)
        self.new = tf.placeholder(tf.int64, shape=state.shape, name='new')
        self.assign_new = self.current.assign(self.new)
        self.step = None

    def advance(self, count=1):
        for _ in range(count):
            self.step += 1
            if self.step % self.report_each == 0:
                self._report()

    def load(self, session):
        state = session.run(self.current)
        self.step = state[0]
        self._report()

    def save(self, session):
        feed = {
            self.new: [self.step],
        }
        session.run(self.assign_new, feed)

    def _report(self):
        support.log(self, 'Current step: {}', self.step)
