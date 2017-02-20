#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

from tensorflow.contrib import rnn as crnn
from tensorflow.python.ops import rnn
import glob
import math
import numpy as np
import queue
import socket
import subprocess
import tensorflow as tf
import threading

from support import Config
import support
import task_usage


class Backup:
    def __init__(self, config):
        self.backend = tf.train.Saver()
        self.path = config.backup_path

    def save(self, session):
        path = self.backend.save(session, self.path)
        support.log(self, 'New backup: {}', path)

    def restore(self, session):
        if len(glob.glob('{}*'.format(self.path))) > 0:
            answer = input('Restore backup "{}"? '.format(self.path))
            if not answer.lower().startswith('n'):
                self.backend.restore(session, self.path)


class DummyTarget:
    def __init__(self, config):
        self.dimension_count = 1
        sample_count = min(10000, config.max_sample_count)
        self.train_sample_count = int(config.train_fraction * sample_count)
        self.test_sample_count = sample_count - self.train_sample_count
        self.train_samples = DummyTarget._generate(
            self.train_sample_count, config)
        self.test_samples = DummyTarget._generate(
            self.test_sample_count, config)

    def test(self, sample):
        return DummyTarget._compute(self.test_samples[sample, :])

    def train(self, sample):
        return DummyTarget._compute(self.train_samples[sample, :])

    def _compute(sample):
        a, b, n = sample[0], sample[1], int(sample[2])
        return np.reshape(np.sin(a * np.linspace(0, n - 1, n) + b), (-1, 1))

    def _generate(count, config):
        min = config.min_sample_length
        max = config.max_sample_length
        samples = np.random.rand(count, 3)
        samples[:, 0] = 0.5 + 1.5 * samples[:, 0]
        samples[:, 1] = 5 * samples[:, 1]
        samples[:, 2] = np.round(min + (max - min) * samples[:, 2])
        return samples


class Learner:
    def __init__(self, config):
        assert(config.batch_size == 1)
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope('model'):
                self.model = Model(config)
            with tf.variable_scope('optimization'):
                self.state = tf.Variable(
                    [0, 0], name='state', dtype=tf.int64, trainable=False)
                self.state_update = tf.placeholder(
                    tf.int64, shape=(2), name='state_update')
                self.update_state = self.state.assign(self.state_update)
                self.parameters = tf.trainable_variables()
                gradient = tf.gradients(self.model.loss, self.parameters)
                gradient, _ = tf.clip_by_global_norm(
                    gradient, config.gradient_clip)
                optimizer = tf.train.AdamOptimizer(config.learning_rate)
                self.train = optimizer.apply_gradients(
                    zip(gradient, self.parameters))
            self.train_summary = tf.summary.scalar(
                'train_loss', self.model.loss)
            self.summary_writer = tf.summary.FileWriter(
                config.summary_path, self.graph)
            self.initialize = tf.variables_initializer(
                tf.global_variables(), name='initialize')
            self.backup = Backup(config)

    @property
    def parameter_count(self):
        return np.sum([int(np.prod(p.get_shape())) for p in self.parameters])

    def run(self, target, manager, config):
        support.log(self, 'Parameters: {}', self.parameter_count)
        session = tf.Session(graph=self.graph)
        session.run(self.initialize)
        self.backup.restore(session)
        state = State.deserialize(session.run(self.state))
        for _ in range(config.epoch_count - state.epoch % config.epoch_count):
            self._run_epoch(target, manager, session, state, config)
            state.increment_epoch()
            session.run(self.update_state, {
                self.state_update: state.serialize(),
            })
            self.backup.save(session)

    def _run_epoch(self, target, manager, session, state, config):
        for _ in range(target.train_sample_count):
            if manager.should_train(state.time):
                self._run_train(target, manager, session, state, config)
            if manager.should_test(state.time):
                self._run_test(target, manager, session, state, config)
            if manager.should_show(state.time):
                self._run_show(target, manager, session, state, config)
            state.increment_time()

    def _run_sample(self, session, sample, callback, config):
        length = sample.shape[0]
        fetch = {
            'y_hat': self.model.y_hat,
            'finish': self.model.finish,
        }
        y_hat = np.empty([config.future_length, config.dimension_count])
        for i in range(length):
            feed = {
                self.model.start: self._zero_start(),
                self.model.x: np.reshape(sample[:(i + 1), :], [1, i + 1, -1]),
            }
            for j in range(config.future_length):
                result = session.run(fetch, feed)
                y_hat[j, :] = result['y_hat'][0, -1, :]
                feed[self.model.start] = result['finish']
                feed[self.model.x] = np.reshape(y_hat[j, :], [1, 1, -1])
            if not callback(y_hat, i + 1):
                break

    def _run_show(self, target, manager, session, state, config):
        sample = target.train(state.sample)
        def _callback(y_hat, offset):
            return manager.show(sample, y_hat, offset)
        self._run_sample(session, sample, _callback, config)

    def _run_test(self, target, manager, session, state, config):
        accumulator, count = [0], [0]
        for sample in range(target.test_sample_count):
            sample = target.test(sample)
            def _callback(y_hat, offset):
                length = min(sample.shape[0] - offset, y_hat.shape[0])
                delta = y_hat[:length, :] - sample[offset:(offset + length), :]
                accumulator[0] += np.sum(delta**2)
                count[0] += length
            self._run_sample(session, sample, _callback, config)
        loss = accumulator[0] / count[0]
        summary = tf.Summary(
            value=[tf.Summary.Value(tag='test_loss', simple_value=loss)])
        self.summary_writer.add_summary(summary, state.time)
        manager.test(loss, state)

    def _run_train(self, target, manager, session, state, config):
        sample = target.train(state.sample)
        feed = {
            self.model.start: self._zero_start(),
            self.model.x: np.reshape(sample, [1, -1, config.dimension_count]),
            self.model.y: np.reshape(support.shift(sample, -1, padding=0),
                                     [1, -1, config.dimension_count]),
        }
        fetch = {
            'train': self.train,
            'loss': self.model.loss,
            'train_summary': self.train_summary,
        }
        result = session.run(fetch, feed)
        self.summary_writer.add_summary(result['train_summary'], state.time)
        manager.train(result['loss'], state)

    def _zero_start(self):
        return np.zeros(self.model.start.get_shape(), np.float32)


class Manager:
    def __init__(self, config):
        self.train_sample_count = config.train_sample_count
        self.show_address = config.show_address
        self.train_schedule = Schedule(config.train_schedule)
        self.train_report_schedule = Schedule(config.train_report_schedule)
        self.test_schedule = Schedule(config.test_schedule)
        self.show_schedule = Schedule(config.show_schedule)
        self.listeners = {}
        self.lock = threading.Lock()
        worker = threading.Thread(target=self._show_server, daemon=True)
        worker.start()

    def should_show(self, time):
        return len(self.listeners) > 0 and self.show_schedule.should(time)

    def should_test(self, time):
        return self.test_schedule.should(time)

    def should_train(self, time):
        return self.train_schedule.should(time)

    def show(self, sample, y_hat, offset):
        count0 = sample.shape[0]
        count1 = count0 - offset
        count2 = y_hat.shape[0]
        count0 = count0 + count2
        message = (np.array([count0, count1, count2]),
                   sample[offset:, :], y_hat)
        with self.lock:
            for listener in self.listeners:
                listener.put(message)
            return len(self.listeners) > 0

    def test(self, loss, state):
        support.log(self, '{} {:12.4e} (test)', self._stamp(state), loss)

    def train(self, loss, state):
        if self.train_report_schedule.should(state.time):
            support.log(self, '{} {:12.4e}', self._stamp(state), loss)

    def _stamp(self, state):
        time, epoch, sample = state.time + 1, state.epoch + 1, state.sample + 1
        return '{:10d} {:4d} {:10d} ({:6.2f}%)'.format(
            time, epoch, sample, 100 * sample / self.train_sample_count)

    def _show_client(self, connection, address):
        support.log(self, 'New listener: {}', address)
        listener = queue.Queue()
        with self.lock:
            self.listeners[listener] = True
        try:
            client = connection.makefile(mode='w')
            while True:
                values = []
                for chunk in listener.get():
                    values.extend([str(value) for value in chunk.flatten()])
                client.write(','.join(values) + '\n')
                client.flush()
        except Exception as e:
            support.log(self, 'Disconnected listener: {} ({})', address, e)
        with self.lock:
            del self.listeners[listener]

    def _show_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(self.show_address)
        server.listen(1)
        support.log(self, 'Show address: {}', self.show_address)
        while True:
            try:
                connection, address = server.accept()
                worker = threading.Thread(target=self._show_client,
                                          args=(connection, address),
                                          daemon=True)
                worker.start()
            except Exception as e:
                support.log(self, 'Exception: {}', e)


class Model:
    def __init__(self, config):
        shape = [None, None, config.dimension_count]
        self.x = tf.placeholder(tf.float32, shape, name='x')
        self.y = tf.placeholder(tf.float32, shape, name='y')
        with tf.variable_scope('batch_size'):
            self.batch_size = tf.shape(self.x)[0]
        with tf.variable_scope('network'):
            self.start, self.finish, h = Model._network(self.x, config)
        with tf.variable_scope('unroll_count'):
            self.unroll_count = tf.shape(h)[1]
        with tf.variable_scope('regression'):
            w, b = Model._regression(self.y, self.batch_size,
                                     self.unroll_count, config)
        with tf.variable_scope('y_hat'):
            self.y_hat = tf.matmul(h, w) + b
        with tf.variable_scope('loss'):
            self.loss = Model._loss(self.y, self.y_hat)

    def _finalize(state, config):
        parts = []
        for i in range(config.layer_count):
            parts.append(state[i].c)
            parts.append(state[i].h)
        return tf.stack(parts, name='finish')

    def _initialize(config):
        shape = [2 * config.layer_count, 1, config.unit_count]
        start = tf.placeholder(tf.float32, shape, name='start')
        parts = tf.unstack(start)
        state = []
        for i in range(config.layer_count):
            c, h = parts[2 * i], parts[2 * i + 1]
            state.append(crnn.LSTMStateTuple(c, h))
        return start, tuple(state)

    def _loss(y, y_hat):
        return tf.reduce_mean(tf.squared_difference(y, y_hat))

    def _network(x, config):
        cell = crnn.LSTMCell(
            config.unit_count, cell_clip=config.cell_clip,
            forget_bias=config.forget_bias, use_peepholes=config.use_peepholes,
            initializer=config.network_initializer)
        cell = crnn.MultiRNNCell([cell] * config.layer_count)
        start, state = Model._initialize(config)
        h, state = rnn.dynamic_rnn(cell, x, initial_state=state)
        finish = Model._finalize(state, config)
        return start, finish, h

    def _regression(y, batch_size, unroll_count, config):
        w = tf.get_variable(
            'w', [1, config.unit_count, config.dimension_count],
            initializer=config.regression_initializer)
        b = tf.get_variable('b', [1, 1, config.dimension_count])
        w = tf.tile(w, [batch_size, 1, 1])
        b = tf.tile(b, [batch_size, unroll_count, 1])
        return w, b


class Schedule:
    def __init__(self, schedule):
        self.schedule = np.cumsum(schedule)

    def should(self, time):
        time = time % self.schedule[-1] + 1
        phase = np.nonzero(self.schedule >= time)[0][0]
        return phase % 2 == 1


class State:
    def deserialize(state):
        return State(state[0], state[1])

    def __init__(self, time, epoch):
        self.time = time
        self.epoch = epoch
        self.sample = 0

    def increment_epoch(self):
        self.epoch += 1
        self.sample = 0

    def increment_time(self):
        self.time += 1
        self.sample += 1

    def serialize(self):
        return [self.time, self.epoch]


class Target:
    def __init__(self, config):
        support.log(self, 'Index: {}', config.index_path)
        self.dimension_count = 1
        self.train_samples = []
        self.test_samples = []
        found_count, selected_count = 0, 0
        with open(config.index_path, 'r') as file:
            for record in file:
                found_count += 1
                record = record.split(',')
                length = int(record[-1])
                if length < config.min_sample_length:
                    continue
                if length > config.max_sample_length:
                    continue
                selected_count +=1
                sample = (record[0], int(record[1]), int(record[2]))
                if np.random.rand() < config.train_fraction:
                    self.train_samples.append(sample)
                else:
                    self.test_samples.append(sample)
        np.random.shuffle(self.train_samples)
        np.random.shuffle(self.test_samples)
        if selected_count > config.max_sample_count:
            def _limit(samples, fraction, total):
                return samples[:min(len(samples), int(fraction * total))]
            self.train_samples = _limit(self.train_samples,
                                        config.train_fraction,
                                        config.max_sample_count)
            self.test_samples = _limit(self.test_samples,
                                       1 - config.train_fraction,
                                       config.max_sample_count)
        self.train_sample_count = len(self.train_samples)
        self.test_sample_count = len(self.test_samples)
        def _format(count, total):
            return '{} ({:.2f}%)'.format(count, 100 * count / total)
        support.log(self, 'Selected samples: {}',
                    _format(selected_count, found_count))
        support.log(self, 'Train samples: {}',
                    _format(self.train_sample_count, selected_count))
        support.log(self, 'Test samples: {}',
                    _format(self.test_sample_count, selected_count))
        standard_count = min(config.standard_count, self.train_sample_count)
        self.standard = self._standardize(standard_count)
        support.log(self, 'Mean: {:e}, deviation: {:e} ({} samples)',
                    self.standard[0], self.standard[1], standard_count)

    def test(self, sample):
        return (self._test(sample) - self.standard[0]) / self.standard[1]

    def train(self, sample):
        return (self._train(sample) - self.standard[0]) / self.standard[1]

    def _standardize(self, count):
        standard = (0.0, 1.0)
        data = np.array([], dtype=np.float32)
        for sample in range(count):
            data = np.append(data, self._train(sample))
        if len(data) > 0:
            standard = (np.mean(data), np.std(data))
        return standard

    def _test(self, sample):
        return task_usage.select(*self.test_samples[sample])

    def _train(self, sample):
        return task_usage.select(*self.train_samples[sample])


def main(config):
    target = DummyTarget(config) if config.dummy else Target(config)
    config.update({
        'dimension_count': target.dimension_count,
    })
    learner = Learner(config)
    config.update({
        'train_sample_count': target.train_sample_count,
    })
    manager = Manager(config)
    learner.run(target, manager, config)

if __name__ == '__main__':
    support.loggalize()
    config = Config({
        # Target
        'dummy': len(sys.argv) == 1,
        'index_path': sys.argv[1] if len(sys.argv) > 1 else None,
        'max_sample_count': 1000000,
        'min_sample_length': 5,
        'max_sample_length': 50,
        'standard_count': 1000,
        # Model
        'layer_count': 1,
        'unit_count': 200,
        'cell_clip': 1.0,
        'forget_bias': 1.0,
        'use_peepholes': True,
        'network_initializer': tf.random_uniform_initializer(-0.01, 0.01),
        'regression_initializer': tf.random_normal_initializer(stddev=0.01),
        # Train
        'batch_size': 1,
        'train_fraction': 0.7,
        'gradient_clip': 1.0,
        'learning_rate': 1e-3,
        'epoch_count': 100,
        'train_schedule': [0, 1],
        'train_report_schedule': [1000 - 1, 1],
        # Test
        'future_length': 10,
        'test_schedule': [10000 - 1, 1],
        # Show
        'show_schedule': [10000 - 1, 1],
        'show_address': ('0.0.0.0', 4242),
        # Backup
        'backup_path': os.path.join('output', 'backup'),
        # Summay
        'summary_path': os.path.join('output', 'summary'),
    })
    main(config)
