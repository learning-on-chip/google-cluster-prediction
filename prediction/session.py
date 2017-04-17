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
        self.input = input
        self.output = config.output
        graph = tf.Graph()
        with graph.as_default():
            shape = [None, None, input.dimension_count]
            x = tf.placeholder(tf.float32, shape, name='x')
            y = tf.placeholder(tf.float32, shape, name='y')
            self.trainee = learner(x, y)
            with tf.variable_scope('trainer'):
                self.trainer = Trainer(self.trainee, config.teacher)
            self.validee = learner(x, y)
            with tf.variable_scope('validator'):
                self.validator = Validator(self.validee, config.teacher)
            self.testee = learner(x, y)
            with tf.variable_scope('tester'):
                self.tester = Tester(self.testee, config.teacher)
            with tf.variable_scope('state'):
                self.state = State()
            self.saver = Saver(self.output)
            initialize = tf.variables_initializer(tf.global_variables(),
                                                  name='initialize')
        self.summarer = tf.summary.FileWriter(self.output.path, graph)
        self.session = tf.Session(graph=graph)
        self.session.run(initialize)
        self.saver.load(self.session)
        self.state.load(self.session)
        self.input.training.restart(self.state.epoch)
        support.log(self, 'Output path: {}', self.output.path)
        support.log(self, 'Initial step: {}, epoch: {}, sample: {}',
                    self.state.step, self.state.epoch, self.state.sample)

    def run_comparison(self, target):
        errors = getattr(self, 'run_' + target)(summarize=False)
        support.summarize_static(self.summarer, errors, 'comparison_' + target)

    def run_saving(self):
        self.state.save(self.session)
        self.saver.save(self.session, self.state)

    def run_testing(self, summarize=True):
        def _compute(*arguments):
            return self.testee.test(self.session, *arguments)
        errors = self.tester.run(self.input.testing, _compute)
        if summarize:
            support.summarize_dynamic(self.summarer, self.state, errors,
                                      'testing')
        return errors

    def run_training(self, summarize=True, sample_count=1):
        def _compute(*arguments):
            return self.trainee.train(self.session, self.trainer.optimize,
                                      self.trainer.loss, *arguments)
        for _ in range(sample_count):
            try:
                errors = self.trainer.run(self.input.training, _compute)
                if summarize:
                    support.summarize_dynamic(self.summarer, self.state,
                                              errors, 'training')
                self.state.increment_time()
            except StopIteration:
                self.state.increment_epoch()
                self.input.training.restart(self.state.epoch)
                support.log(
                    self, 'Current step: {}, epoch: {}, sample: {}',
                    self.state.step, self.state.epoch, self.state.sample)

    def run_validation(self, summarize=True):
        def _compute(*arguments):
            return self.validee.validate(self.session, self.validator.loss,
                                         *arguments)
        errors = self.validator.run(self.input.validation, _compute)
        if summarize:
            support.summarize_dynamic(self.summarer, self.state, errors,
                                      'validation')
        return errors


class State:
    def __init__(self):
        self.current = tf.Variable([0, 0, 0], name='current', dtype=tf.int64,
                                   trainable=False)
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
