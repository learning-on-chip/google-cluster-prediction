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
            with tf.variable_scope('training'):
                with tf.variable_scope('input'):
                    input_ = input('training')
                    inputs = input_.initiate()
                learner_ = learner(*inputs)
                with tf.variable_scope('teacher'):
                    self.trainer = Trainer(input_, learner_, config.teacher)
            with tf.variable_scope('validation'):
                with tf.variable_scope('input'):
                    input_ = input('validation')
                    inputs = input_.initiate()
                learner_ = learner(*inputs)
                with tf.variable_scope('teacher'):
                    self.validator = Validator(input_, learner_,
                                               config.teacher)
            with tf.variable_scope('testing'):
                with tf.variable_scope('input'):
                    input_ = input('testing')
                    inputs = input_.initiate()
                learner_ = learner(*inputs)
                with tf.variable_scope('teacher'):
                    self.tester = Tester(input_, learner_, config.teacher)
        with graph.as_default():
            self.backend = tf.Session()
            self.backend.run(tf.variables_initializer(
                tf.global_variables(), name='initialize'))
        with graph.as_default():
            self.saver = Saver(self.output, name='saver')
            self.saver.subscribe(self.trainer.input)
            self.saver.subscribe(self.validator.input)
            self.saver.subscribe(self.tester.input)
            self.saver.restore(self.backend)
        self.summarer = tf.summary.FileWriter(self.output.path, graph)

    def run_comparison(self, target):
        errors = getattr(self, 'run_' + target)(summarize=False)
        summarize_static(self.summarer, errors, 'comparison_' + target)

    def run_saving(self):
        self.saver.save(self.backend, self.step)

    def run_testing(self, summarize=True):
        errors = self.tester.run(self.backend)
        if summarize:
            summarize_dynamic(self.summarer, self.step, errors, 'testing')
        return errors

    def run_training(self, step_count=1, summarize=True):
        errors = self.trainer.run(self.backend, step_count)
        if summarize:
            summarize_dynamic(self.summarer, self.step, errors, 'training')
        return errors

    def run_validation(self, summarize=True):
        errors = self.validator.run(self.backend)
        if summarize:
            summarize_dynamic(self.summarer, self.step, errors, 'validation')
        return errors

    @property
    def step(self):
        return self.trainer.input.step


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
