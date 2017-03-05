from manager import Manager
from model import Model
from trainer import Trainer
import glob
import numpy as np
import os
import support
import tensorflow as tf


class Learner:
    def __init__(self, config):
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
        self.backup = Backup(
            self.graph, os.path.join(config.output.path, 'backup'))
        self.manager = Manager(config.manager)
        support.log(self, 'Output path: {}', config.output.path)

    def run(self, input):
        support.log(self, 'Parameters: {}', self.model.parameter_count)
        support.log(self, 'Train samples: {}', input.train.sample_count)
        support.log(self, 'Test samples: {}', input.test.sample_count)
        session = tf.Session(graph=self.graph)
        session.run(self.initialize)
        self.backup.restore(session)
        state = self.trainer.get_state(session)
        support.log(self, 'Initial state: time {}, epoch {}, sample {}',
                    state.time, state.epoch, state.sample)
        input.on_epoch(state)
        while self.manager.should_continue(state):
            self._run(session, state, input)

    def _run(self, session, state, input):
        should_backup = self.manager.should_backup(state)
        if self.manager.should_train(state):
            self._run_train(session, state, input)
        if self.manager.should_test(state):
            self._run_test(session, state, input)
        if self.manager.should_show(state):
            self._run_show(session, state, input)
        state.increment_time()
        if state.sample == input.train.sample_count:
            state.increment_epoch()
            input.on_epoch(state)
            support.log(self, 'Current state: time {}, epoch {}, sample {}',
                        state.time, state.epoch, state.sample)
        if should_backup:
            self._run_backup(session, state)

    def _run_backup(self, session, state):
        self.trainer.set_state(session, state)
        support.log(self, 'New backup: {}', self.backup.save(session))

    def _run_sample(self, session, input, sample, callback):
        length = sample.shape[0]
        fetch = {
            'y_hat': self.model.y_hat,
            'finish': self.model.finish,
        }
        y_hat = np.empty([self.trainer.test_length, input.dimension_count])
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

    def _run_show(self, session, state, input):
        sample = input.train.get(state.sample)
        def _callback(y_hat, offset):
            return self.manager.on_show(sample, y_hat, offset)
        self._run_sample(session, input, sample, _callback)

    def _run_test(self, session, state, input):
        sums = np.zeros([self.trainer.test_length])
        counts = np.zeros([self.trainer.test_length], dtype=np.int)
        for sample in range(input.test.sample_count):
            sample = input.test.get(sample)
            def _callback(y_hat, offset):
                length = min(sample.shape[0] - offset, y_hat.shape[0])
                delta = y_hat[:length, :] - sample[offset:(offset + length), :]
                sums[:length] += np.sum(delta**2, axis=0)
                counts[:length] += 1
            self._run_sample(session, input, sample, _callback)
        loss = sums / counts
        for i in range(self.trainer.test_length):
            value = tf.Summary.Value(
                tag=('test_loss_' + str(i + 1)), simple_value=loss[i])
            self.summary_writer.add_summary(
                tf.Summary(value=[value]), state.time)

    def _run_train(self, session, state, input):
        sample = input.train.get(state.sample)
        feed = {
            self.model.start: self._zero_start(),
            self.model.x: np.reshape(sample, [1, -1, input.dimension_count]),
            self.model.y: np.reshape(support.shift(sample, -1, padding=0),
                                     [1, -1, input.dimension_count]),
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


class Backup:
    def __init__(self, graph, path):
        with graph.as_default():
            self.backend = tf.train.Saver()
        self.path = path

    def restore(self, session):
        if len(glob.glob('{}*'.format(self.path))) > 0:
            answer = input('Restore backup "{}"? '.format(self.path))
            if not answer.lower().startswith('n'):
                self.backend.restore(session, self.path)

    def save(self, session):
        return self.backend.save(session, self.path)
