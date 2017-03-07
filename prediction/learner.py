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
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope('model'):
                self.model = Model(config.model)
            with tf.variable_scope('trainer'):
                self.trainer = Trainer(self.model, config.trainer)
            tf.summary.scalar('train_loss', self.trainer.loss)
            tf.summary.scalar('unroll_count', self.model.unroll_count)
            self.train_summary = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(
                config.output.path, self.graph)
            self.initialize = tf.variables_initializer(
                tf.global_variables(), name='initialize')
        self.checkpoint = Checkpoint(self.graph, config.output)
        self.manager = Manager(config.manager)
        support.log(self, 'Parameters: {}', self.model.parameter_count)
        support.log(self, 'Train samples: {}', self.input.train.sample_count)
        support.log(self, 'Test samples: {}', self.input.test.sample_count)
        support.log(self, 'Output path: {}', config.output.path)

    def run(self):
        session = tf.Session(graph=self.graph)
        session.run(self.initialize)
        self.checkpoint.restore(session)
        state = self.trainer.get_state(session)
        support.log(self, 'Initial state: time {}, epoch {}, sample {}',
                    state.time, state.epoch, state.sample)
        self.input.on_epoch(state)
        while self.manager.should_continue(state):
            self._run(session, state)

    def _run(self, session, state):
        should_backup = self.manager.should_backup(state)
        if self.manager.should_train(state):
            self._run_train(session, state)
        if self.manager.should_test(state):
            self._run_test(session, state)
        if self.manager.should_show(state):
            self._run_show(session, state)
        state.increment_time()
        if state.sample == self.input.train.sample_count:
            state.increment_epoch()
            self.input.on_epoch(state)
            support.log(self, 'Current state: time {}, epoch {}, sample {}',
                        state.time, state.epoch, state.sample)
        if should_backup:
            self._run_backup(session, state)

    def _run_backup(self, session, state):
        self.trainer.set_state(session, state)
        support.log(self, 'New checkpoint: {}', self.checkpoint.save(session))

    def _run_sample(self, session, sample, callback):
        length = sample.shape[0]
        fetch = {
            'y_hat': self.model.y_hat,
            'finish': self.model.finish,
        }
        y_hat = np.empty(
            [self.trainer.test_length, self.input.dimension_count])
        for i in range(length):
            feed = {
                self.model.start: self._zero_start(),
                self.model.x: np.reshape(sample[:(i + 1), :], [1, i + 1, -1]),
            }
            for j in range(self.trainer.test_length):
                result = session.run(fetch, feed)
                y_hat[j, :] = result['y_hat'][0, -1, :]
                feed[self.model.start] = result['finish']
                feed[self.model.x] = np.reshape(y_hat[j, :], [1, 1, -1])
            if not callback(y_hat, i + 1):
                break

    def _run_show(self, session, state):
        sample = self.input.train.get(state.sample)
        def _callback(y_hat, offset):
            return self.manager.on_show(sample, y_hat, offset)
        self._run_sample(session, sample, _callback)

    def _run_test(self, session, state):
        sums = np.zeros([self.trainer.test_length])
        counts = np.zeros([self.trainer.test_length], dtype=np.int)
        for sample in range(self.input.test.sample_count):
            sample = self.input.test.get(sample)
            def _callback(y_hat, offset):
                length = min(sample.shape[0] - offset, y_hat.shape[0])
                delta = y_hat[:length, :] - sample[offset:(offset + length), :]
                sums[:length] += np.sum(delta**2, axis=0)
                counts[:length] += 1
            self._run_sample(session, sample, _callback)
        loss = sums / counts
        for i in range(self.trainer.test_length):
            value = tf.Summary.Value(
                tag=('test_loss_' + str(i + 1)), simple_value=loss[i])
            self.summary_writer.add_summary(
                tf.Summary(value=[value]), state.time)

    def _run_train(self, session, state):
        sample = self.input.train.get(state.sample)
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
        result = session.run(fetch, feed)
        self.summary_writer.add_summary(result['train_summary'], state.time)

    def _zero_start(self):
        return np.zeros(self.model.start.get_shape(), np.float32)


class Checkpoint:
    def __init__(self, graph, config):
        with graph.as_default():
            self.saver = tf.train.Saver()
        self.path = os.path.join(config.path, 'model')
        self.auto_restore = config.get('auto_restore')

    def restore(self, session):
        if len(glob.glob('{}*'.format(self.path))) == 0:
            return
        restore = self.auto_restore
        if restore is None:
            restore = input('Restore checkpoint "{}"? '.format(self.path))
            restore = not restore.lower().startswith('n')
        if not restore:
            return
        support.log(self, 'Restore checkpoint: {}', self.path)
        self.saver.restore(session, self.path)

    def save(self, session):
        return self.saver.save(session, self.path)
