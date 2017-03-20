from . import support
from .manager import Manager
from .model import Model
from .teacher import Teacher
import glob
import numpy as np
import os
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
            tf.summary.scalar('train_loss', self.teacher.loss)
            tf.summary.scalar('unroll_count', self.model.unroll_count)
            self.train_summary = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(
                self.output.path, graph)
            initialize = tf.variables_initializer(
                tf.global_variables(), name='initialize')
            self.checkpoint = Checkpoint(self.output)
        self.manager = Manager(config.manager)
        self.session = tf.Session(graph=graph)
        self.session.run(initialize)
        self.checkpoint.load(self.session)
        self.state.load(self.session)
        self.input.on_epoch(self.state)
        support.log(self, 'Output path: {}', self.output.path)
        support.log(self, 'Initial state: iteration {}, epoch {}, sample {}',
                    self.state.iteration, self.state.epoch, self.state.sample)

    def increment_time(self):
        self.state.increment_time()
        if self.state.is_new_epoch():
            self.input.on_epoch(self.state)
            support.log(
                self, 'Current state: iteration {}, epoch {}, sample {}',
                self.state.iteration, self.state.epoch, self.state.sample)

    def run(self):
        should_backup = self.manager.should_backup(self.state)
        if self.manager.should_train(self.state):
            self.run_train()
        if self.manager.should_test(self.state):
            self.run_test()
        self.increment_time()
        if should_backup:
            self.run_backup()

    def run_backup(self):
        self.state.save(self.session)
        self.checkpoint.save(self.session)

    def run_test(self):
        loss = self.teacher.test(self.input.test, self._run_test)
        for i in range(len(loss)):
            value = tf.Summary.Value(
                tag=('test_loss_' + str(i + 1)), simple_value=loss[i])
            self.summary_writer.add_summary(
                tf.Summary(value=[value]), self.state.iteration)
        return loss

    def run_train(self):
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
            'step': self.teacher.step,
            'loss': self.teacher.loss,
            'train_summary': self.train_summary,
        }
        result = self.session.run(fetch, feed)
        self.summary_writer.add_summary(
            result['train_summary'], self.state.iteration)
        return result['loss']

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

    def _zero_start(self):
        return np.zeros(self.model.start.get_shape(), np.float32)


class Checkpoint:
    def __init__(self, output):
        self.saver = tf.train.Saver()
        self.path = os.path.join(output.path, 'model')
        self.auto = output.get('auto_restore')

    def load(self, session):
        if len(glob.glob('{}*'.format(self.path))) == 0:
            return
        should = self.auto
        if should is None:
            answer = input('Restore checkpoint "{}"? '.format(self.path))
            should = not answer.lower().startswith('n')
        if not should:
            return
        self.saver.restore(session, self.path)
        support.log(self, 'Restore: {}', self.path)

    def save(self, session):
        path = self.saver.save(session, self.path)
        support.log(self, 'Save: {}', path)


class State:
    def __init__(self, sample_count):
        self.current = tf.Variable(
            [0, 0, 0], name='current', dtype=tf.int64, trainable=False)
        self.new = tf.placeholder(tf.int64, shape=3, name='new')
        self.assign_new = self.current.assign(self.new)
        self.iteration, self.epoch, self.sample = None, None, None
        self.sample_count = sample_count

    def increment_time(self):
        self.iteration += 1
        self.sample += 1
        if self.sample == self.sample_count:
            self.epoch += 1
            self.sample = 0

    def is_new_epoch(self):
        return self.sample == 0

    def load(self, session):
        state = session.run(self.current)
        self.iteration, self.epoch, self.sample = state

    def save(self, session):
        feed = {
            self.new: [self.iteration, self.epoch, self.sample],
        }
        session.run(self.assign_new, feed)
