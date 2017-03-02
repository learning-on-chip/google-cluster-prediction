from model import Model
from optimizer import Optimizer
import glob
import numpy as np
import os
import support
import tensorflow as tf


class Learner:
    def __init__(self, config):
        assert(config.batch_size == 1)
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope('model'):
                self.model = Model(config.model)
            with tf.variable_scope('optimizer'):
                self.optimizer = Optimizer(self.model, config.optimizer)
            tf.summary.scalar('train_loss', self.optimizer.loss)
            tf.summary.scalar('unroll_count', self.model.unroll_count)
            self.train_summary = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(
                config.output_path, self.graph)
            self.initialize = tf.variables_initializer(
                tf.global_variables(), name='initialize')
            self.backup = Backup(config)

    def run(self, target, manager, config):
        support.log(self, 'Parameters: {}', self.model.parameter_count)
        support.log(self, 'Train samples: {}', target.train.sample_count)
        support.log(self, 'Test samples: {}', target.test.sample_count)
        session = tf.Session(graph=self.graph)
        session.run(self.initialize)
        self.backup.restore(session)
        state = self.optimizer.get_state(session)
        for _ in range(state.epoch, config.epoch_count):
            support.log(self, 'Current state: time {}, epoch {}, sample {}',
                        state.time, state.epoch, state.sample)
            self._run_epoch(session, state, target, manager, config)
            state.increment_epoch()

    def _run_epoch(self, session, state, target, manager, config):
        target.on_epoch(state)
        for _ in range(state.sample, target.train.sample_count):
            if manager.should_train(state.time):
                self._run_train(session, state, target, config)
            if manager.should_test(state.time):
                self._run_test(session, state, target, config)
            if manager.should_show(state.time):
                self._run_show(session, state, target, manager, config)
            if manager.should_backup(state.time):
                state.increment_time()
                self._run_backup(session, state)
            else:
                state.increment_time()

    def _run_backup(self, session, state):
        self.optimizer.set_state(session, state)
        path = self.backup.save(session)
        support.log(self, 'Backup: {}', path)

    def _run_sample(self, session, target, sample, callback, config):
        length = sample.shape[0]
        fetch = {
            'y_hat': self.model.y_hat,
            'finish': self.model.finish,
        }
        y_hat = np.empty([config.test_length, target.dimension_count])
        for i in range(length):
            feed = {
                self.model.start: self._zero_start(),
                self.model.x: np.reshape(sample[:(i + 1), :], [1, i + 1, -1]),
            }
            for j in range(config.test_length):
                result = session.run(fetch, feed)
                y_hat[j, :] = result['y_hat'][0, -1, :]
                feed[self.model.start] = result['finish']
                feed[self.model.x] = np.reshape(y_hat[j, :], [1, 1, -1])
            if not callback(y_hat, i + 1):
                break

    def _run_show(self, session, state, target, manager, config):
        sample = target.train.get(state.sample)
        def _callback(y_hat, offset):
            return manager.on_show(sample, y_hat, offset)
        self._run_sample(session, target, sample, _callback, config)

    def _run_test(self, session, state, target, config):
        sums = np.zeros([config.test_length])
        counts = np.zeros([config.test_length], dtype=np.int)
        for sample in range(target.test.sample_count):
            sample = target.test.get(sample)
            def _callback(y_hat, offset):
                length = min(sample.shape[0] - offset, y_hat.shape[0])
                delta = y_hat[:length, :] - sample[offset:(offset + length), :]
                sums[:length] += np.sum(delta**2, axis=0)
                counts[:length] += 1
            self._run_sample(session, target, sample, _callback, config)
        loss = sums / counts
        for i in range(config.test_length):
            value = tf.Summary.Value(
                tag=('test_loss_' + str(i + 1)), simple_value=loss[i])
            self.summary_writer.add_summary(
                tf.Summary(value=[value]), state.time)

    def _run_train(self, session, state, target, config):
        sample = target.train.get(state.sample)
        feed = {
            self.model.start: self._zero_start(),
            self.model.x: np.reshape(sample, [1, -1, target.dimension_count]),
            self.model.y: np.reshape(support.shift(sample, -1, padding=0),
                                     [1, -1, target.dimension_count]),
        }
        fetch = {
            'step': self.optimizer.step,
            'loss': self.optimizer.loss,
            'train_summary': self.train_summary,
        }
        result = session.run(fetch, feed)
        self.summary_writer.add_summary(result['train_summary'], state.time)

    def _zero_start(self):
        return np.zeros(self.model.start.get_shape(), np.float32)


class Backup:
    def __init__(self, config):
        self.backend = tf.train.Saver()
        self.path = os.path.join(config.output_path, 'backup')

    def restore(self, session):
        if len(glob.glob('{}*'.format(self.path))) > 0:
            answer = input('Restore backup "{}"? '.format(self.path))
            if not answer.lower().startswith('n'):
                self.backend.restore(session, self.path)

    def save(self, session):
        return self.backend.save(session, self.path)
