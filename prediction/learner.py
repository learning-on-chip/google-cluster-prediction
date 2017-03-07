from . import support
from .input import Input
from .manager import Manager
from .model import Model
from .trainer import Trainer
import glob
import numpy as np
import os
import tensorflow as tf


class Learner:
    def __init__(self, config):
        self.input = Input.instantiate(config.input)
        self.output = config.output
        graph = tf.Graph()
        with graph.as_default():
            with tf.variable_scope('model'):
                self.model = Model(config.model)
            with tf.variable_scope('state'):
                self.state = State()
            with tf.variable_scope('trainer'):
                self.trainer = Trainer(self.model, config.trainer)
            tf.summary.scalar('train_loss', self.trainer.loss)
            tf.summary.scalar('unroll_count', self.model.unroll_count)
            self.train_summary = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(
                self.output.path, graph)
            initialize = tf.variables_initializer(
                tf.global_variables(), name='initialize')
            self.checkpoint = Checkpoint(self.output)
        self.manager = Manager(config.manager)
        support.log(self, 'Parameters: {}', self.model.parameter_count)
        support.log(self, 'Train samples: {}', self.input.train.sample_count)
        support.log(self, 'Test samples: {}', self.input.test.sample_count)
        support.log(self, 'Output path: {}', self.output.path)
        self.session = tf.Session(graph=graph)
        self.session.run(initialize)
        self.checkpoint.load(self.session)
        self.state.load(self.session)
        self.input.on_epoch(self.state)
        support.log(self, 'Initial state: time {}, epoch {}, sample {}',
                    self.state.time, self.state.epoch, self.state.sample)

    def run(self):
        should_backup = self.manager.should_backup(self.state)
        if self.manager.should_train(self.state):
            self._run_train()
        if self.manager.should_test(self.state):
            self._run_test()
        if self.manager.should_show(self.state):
            self._run_show()
        self.state.increment_time()
        if self.state.sample == self.input.train.sample_count:
            self.state.increment_epoch()
            self.input.on_epoch(self.state)
            support.log(self, 'Current state: time {}, epoch {}, sample {}',
                        self.state.time, self.state.epoch, self.state.sample)
        if should_backup:
            self._run_backup()

    def _run_backup(self):
        self.state.save(self.session)
        path = self.checkpoint.save(self.session)
        support.log(self, 'New checkpoint: {}', path)

    def _run_sample(self, sample, callback):
        length = sample.shape[0]
        fetch = {
            'y_hat': self.model.y_hat,
            'finish': self.model.finish,
        }
        y_hat = np.empty([self.output.test_length, self.input.dimension_count])
        for i in range(length):
            feed = {
                self.model.start: self._zero_start(),
                self.model.x: np.reshape(sample[:(i + 1), :], [1, i + 1, -1]),
            }
            for j in range(self.output.test_length):
                result = self.session.run(fetch, feed)
                y_hat[j, :] = result['y_hat'][0, -1, :]
                feed[self.model.start] = result['finish']
                feed[self.model.x] = np.reshape(y_hat[j, :], [1, 1, -1])
            if not callback(y_hat, i + 1):
                break

    def _run_show(self):
        sample = self.input.train.get(self.state.sample)
        def _callback(y_hat, offset):
            return self.manager.on_show(sample, y_hat, offset)
        self._run_sample(sample, _callback)

    def _run_test(self):
        sums = np.zeros([self.output.test_length])
        counts = np.zeros([self.output.test_length], dtype=np.int)
        for sample in range(self.input.test.sample_count):
            sample = self.input.test.get(sample)
            def _callback(y_hat, offset):
                length = min(sample.shape[0] - offset, y_hat.shape[0])
                delta = y_hat[:length, :] - sample[offset:(offset + length), :]
                sums[:length] += np.sum(delta**2, axis=0)
                counts[:length] += 1
            self._run_sample(sample, _callback)
        loss = sums / counts
        for i in range(self.output.test_length):
            value = tf.Summary.Value(
                tag=('test_loss_' + str(i + 1)), simple_value=loss[i])
            self.summary_writer.add_summary(
                tf.Summary(value=[value]), self.state.time)

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
            'step': self.trainer.step,
            'loss': self.trainer.loss,
            'train_summary': self.train_summary,
        }
        result = self.session.run(fetch, feed)
        self.summary_writer.add_summary(
            result['train_summary'], self.state.time)

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
        support.log(self, 'Restore checkpoint: {}', self.path)
        self.saver.restore(session, self.path)

    def save(self, session):
        return self.saver.save(session, self.path)


class State:
    def __init__(self):
        self.current = tf.Variable(
            [0, 0, 0], name='current', dtype=tf.int64, trainable=False)
        self.new = tf.placeholder(tf.int64, shape=3, name='new')
        self.assign_new = self.current.assign(self.new)
        self.time, self.epoch, self.sample = None, None, None

    def increment_epoch(self):
        self.epoch += 1
        self.sample = 0

    def increment_time(self):
        self.time += 1
        self.sample += 1

    def load(self, session):
        state = session.run(self.current)
        self.time, self.epoch, self.sample = state

    def save(self, session):
        feed = {
            self.new: [self.time, self.epoch, self.sample],
        }
        session.run(self.assign_new, feed)
