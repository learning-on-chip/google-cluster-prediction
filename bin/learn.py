#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

from tensorflow.contrib import rnn as crnn
from tensorflow.python.ops import rnn
import argparse
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

    def restore(self, session):
        if len(glob.glob('{}*'.format(self.path))) > 0:
            answer = input('Restore backup "{}"? '.format(self.path))
            if not answer.lower().startswith('n'):
                self.backend.restore(session, self.path)

    def save(self, session):
        return self.backend.save(session, self.path)


class Learner:
    def __init__(self, config):
        assert(config.batch_size == 1)
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope('model'):
                self.model = Model(config)
            with tf.variable_scope('optimization'):
                self.state = tf.Variable(
                    [0, 0, 0], name='state', dtype=tf.int64, trainable=False)
                self.state_update = tf.placeholder(
                    tf.int64, shape=(3), name='state_update')
                self.update_state = self.state.assign(self.state_update)
                self.parameters = tf.trainable_variables()
                gradient = tf.gradients(self.model.loss, self.parameters)
                gradient, _ = tf.clip_by_global_norm(
                    gradient, config.gradient_clip)
                optimizer = tf.train.AdamOptimizer(config.learning_rate)
                self.train = optimizer.apply_gradients(
                    zip(gradient, self.parameters))
            tf.summary.scalar('train_loss', self.model.loss)
            tf.summary.scalar('unroll_count', self.model.unroll_count)
            self.train_summary = tf.summary.merge_all()
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
        support.log(self, 'Train samples: {}', target.train.sample_count)
        support.log(self, 'Test samples: {}', target.test.sample_count)
        session = tf.Session(graph=self.graph)
        session.run(self.initialize)
        self.backup.restore(session)
        state = State.deserialize(session.run(self.state))
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
        session.run(self.update_state, {
            self.state_update: state.serialize(),
        })
        path = self.backup.save(session)
        support.log(self, 'Backup: {}', path)

    def _run_sample(self, session, sample, callback, config):
        length = sample.shape[0]
        fetch = {
            'y_hat': self.model.y_hat,
            'finish': self.model.finish,
        }
        y_hat = np.empty([config.test_length, config.dimension_count])
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
        self._run_sample(session, sample, _callback, config)

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
            self._run_sample(session, sample, _callback, config)
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

    def _zero_start(self):
        return np.zeros(self.model.start.get_shape(), np.float32)


class Manager:
    def __init__(self, config):
        self.test_schedule = Schedule(config.test_schedule)
        self.backup_schedule = Schedule(config.backup_schedule)
        self.show_schedule = Schedule(config.show_schedule)
        self.show_address = config.show_address
        self.listeners = {}
        self.lock = threading.Lock()
        worker = threading.Thread(target=self._show_server, daemon=True)
        worker.start()

    def on_show(self, sample, y_hat, offset):
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

    def should_backup(self, time):
        return self.backup_schedule.should(time)

    def should_show(self, time):
        return len(self.listeners) > 0 and self.show_schedule.should(time)

    def should_test(self, time):
        return self.test_schedule.should(time)

    def should_train(self, _):
        return True

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
        cell = crnn.LSTMCell(config.unit_count,
                             initializer=config.network_initializer,
                             **config.cell_options)
        cell = crnn.DropoutWrapper(cell, **config.dropout_options)
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
        return State(*state)

    def __init__(self, time, epoch, sample):
        self.time = time
        self.epoch = epoch
        self.sample = sample

    def increment_epoch(self):
        self.epoch += 1
        self.sample = 0

    def increment_time(self):
        self.time += 1
        self.sample += 1

    def serialize(self):
        return [self.time, self.epoch, self.sample]


class Target:
    def on_epoch(self, state):
        random_state = np.random.get_state()
        np.random.seed(state.epoch)
        np.random.shuffle(self.train.samples)
        np.random.set_state(random_state)


class TargetFake(Target):
    class Part:
        def __init__(self, samples):
            self.sample_count = len(samples)
            self.samples = samples

        def get(self, sample):
            return TargetFake._compute(self.samples[sample, :])

    def __init__(self, config):
        self.dimension_count = 1
        sample_count = 10000
        train_sample_count = int(config.train_fraction * sample_count)
        test_sample_count = sample_count - train_sample_count
        self.train = TargetFake.Part(TargetFake._generate(train_sample_count))
        self.test = TargetFake.Part(TargetFake._generate(test_sample_count))

    def _compute(sample):
        a, b, n = sample[0], sample[1], int(sample[2])
        return np.reshape(np.sin(a * np.linspace(0, n - 1, n) + b), (-1, 1))

    def _generate(count):
        samples = np.random.rand(count, 3)
        samples[:, 0] = 0.5 + 1.5 * samples[:, 0]
        samples[:, 1] = 5 * samples[:, 1]
        samples[:, 2] = np.round(5 + 45 * samples[:, 2])
        return samples


class TargetReal(Target):
    class Part:
        def __init__(self, samples, standard):
            self.sample_count = len(samples)
            self.samples = samples
            self.standard = standard

        def get(self, sample):
            data = task_usage.select(*self.samples[sample])
            return (data - self.standard[0]) / self.standard[1]

    def __init__(self, config):
        self.dimension_count = 1
        support.log(self, 'Index: {}', config.index_path)
        found_count = 0
        samples = []
        with open(config.index_path, 'r') as file:
            for record in file:
                found_count += 1
                record = record.split(',')
                length = int(record[-1])
                if length < config.min_sample_length:
                    continue
                if length > config.max_sample_length:
                    continue
                samples.append((record[0], int(record[1]), int(record[2])))
        np.random.shuffle(samples)
        selected_count = len(samples)
        if selected_count > config.max_sample_count:
            samples = samples[:config.max_sample_count]
        preserved_count = len(samples)
        support.log(self, 'Found samples: {}', found_count)
        support.log(self, 'Selected samples: {}',
                    support.format_percentage(selected_count, found_count))
        support.log(self, 'Preserved samples: {}',
                    support.format_percentage(preserved_count, found_count))
        train_sample_count = int(config.train_fraction * len(samples))
        test_sample_count = len(samples) - train_sample_count
        train_samples = samples[:train_sample_count]
        test_samples = samples[train_sample_count:]
        standard_count = min(config.standard_count, train_sample_count)
        standard = TargetReal._standardize(train_samples, standard_count)
        support.log(self, 'Mean: {:e}, deviation: {:e} ({} samples)',
                    standard[0], standard[1], standard_count)
        self.train = TargetReal.Part(train_samples, standard)
        self.test = TargetReal.Part(test_samples, standard)

    def _standardize(samples, count):
        data = np.array([], dtype=np.float32)
        for sample in np.random.permutation(len(samples))[:count]:
            data = np.append(data, task_usage.select(*samples[sample]))
        if len(data) > 0:
            return (np.mean(data), np.std(data))
        else:
            return (0.0, 1.0)


def main(config):
    if config.has('index_path'):
        target = TargetReal(config)
    else:
        target = TargetFake(config)
    config.update({
        'dimension_count': target.dimension_count,
    })
    learner = Learner(config)
    manager = Manager(config)
    learner.run(target, manager, config)

if __name__ == '__main__':
    support.loggalize()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument(
        '--output', default=os.path.join('output', support.format_timestamp()))
    parser.add_argument('--seed', default=0)
    arguments = parser.parse_args()
    output_path = arguments.output
    np.random.seed(arguments.seed)
    config = Config({
        # Model
        'layer_count': 1,
        'unit_count': 200,
        'dropout_options': {
            'input_keep_prob': 1.0,
            'output_keep_prob': 1.0,
        },
        'cell_options': {
            'cell_clip': 1.0,
            'forget_bias': 1.0,
            'use_peepholes': True,
        },
        'network_initializer': tf.random_uniform_initializer(minval=-0.01,
                                                             maxval=0.01),
        'regression_initializer': tf.random_normal_initializer(stddev=0.01),
        # Train
        'batch_size': 1,
        'train_fraction': 0.7,
        'gradient_clip': 1.0,
        'learning_rate': 1e-4,
        'epoch_count': 100,
        # Test
        'test_schedule': [1000, 1],
        'test_length': 10,
        # Backup
        'backup_schedule': [10000, 1],
        'backup_path': os.path.join(output_path, 'backup'),
        # Show
        'show_schedule': [1000, 1],
        'show_address': ('0.0.0.0', 4242),
        # Summay
        'summary_path': output_path,
    })
    if arguments.input is not None:
        config.update({
            # Target
            'index_path': arguments.input,
            'standard_count': 1000,
            'max_sample_count': 1000000,
            'min_sample_length': 5,
            'max_sample_length': 50,
        })
    main(config)
