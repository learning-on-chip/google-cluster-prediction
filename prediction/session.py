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
        with graph.as_default():
            self.saver = Saver(self.output, name='saver')
            self.saver.subscribe(self.input.training)
            self.saver.subscribe(self.input.validation)
            self.saver.subscribe(self.input.testing)
        with graph.as_default():
            self.backend = tf.Session()
            self.backend.run(tf.variables_initializer(
                tf.global_variables(), name='initialize'))
        self.summarer = tf.summary.FileWriter(self.output.path, graph)
        self.saver.restore(self.backend)

    def run_comparison(self, target):
        errors = getattr(self, 'run_' + target)(summarize=False)
        summarize_static(self.summarer, errors, 'comparison_' + target)

    def run_saving(self):
        self.saver.save(self.backend, self.step)

    def run_testing(self, summarize=True):
        errors = self.tester.run(
            self.input.testing, self.backend,
            lambda *arguments: self.testee.test(self.backend, *arguments))
        if summarize:
            summarize_dynamic(self.summarer, self.step, errors, 'testing')
        return errors

    def run_training(self, step_count=1, summarize=True):
        errors = self.trainer.run(
            self.input.training, self.backend, step_count,
            lambda *arguments: self.trainee.train(
                self.backend, self.trainer.optimize, self.trainer.loss,
                *arguments))
        if summarize:
            summarize_dynamic(self.summarer, self.step, errors, 'training')
        return errors

    def run_validation(self, summarize=True):
        errors = self.validator.run(
            self.input.validation, self.backend,
            lambda *arguments: self.validee.validate(
                self.backend, self.validator.loss, *arguments))
        if summarize:
            summarize_dynamic(self.summarer, self.step, errors, 'validation')
        return errors

    @property
    def step(self):
        return self.input.training.step


def summarize_dynamic(summarer, step, data, name):
    for key in data:
        for i in range(len(data[key])):
            tag = '{}_{}_{}'.format(name, key, i + 1)
            value = tf.Summary.Value(tag=tag, simple_value=data[key][i])
            summarer.add_summary(tf.Summary(value=[value]), step)
    summarer.flush()

def summarize_static(summarer, data, name):
    for key in data:
        tag = '{}_{}'.format(name, key)
        for i in range(len(data[key])):
            value = tf.Summary.Value(tag=tag, simple_value=data[key][i])
            summarer.add_summary(tf.Summary(value=[value]), i + 1)
    summarer.flush()
