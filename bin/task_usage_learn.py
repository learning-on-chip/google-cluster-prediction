#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

import numpy as np
import glob, math, queue, random, socket, subprocess, threading
import tensorflow as tf

from support import Config
import support, task_usage

class Learn:
    def __init__(self, config):
        graph = tf.Graph()
        with graph.as_default():
            model = Model(config)
            with tf.variable_scope('optimization'):
                state = tf.Variable(
                    [0, 0], name='state', dtype=tf.int64, trainable=False)
                state_update = tf.placeholder(
                    tf.int64, shape=(2), name='state_update')
                update_state = state.assign(state_update)
                parameters = tf.trainable_variables()
                gradient = tf.gradients(model.loss, parameters)
                gradient, _ = tf.clip_by_global_norm(
                    gradient, config.gradient_clip)
                optimizer = tf.train.AdamOptimizer(config.learning_rate)
                train = optimizer.apply_gradients(zip(gradient, parameters))
            with tf.variable_scope('summary'):
                tf.summary.scalar(
                    'log_loss', tf.log(tf.reduce_sum(model.loss)))
            logger = tf.summary.FileWriter(config.log_path, graph)
            summary = tf.summary.merge_all()
            initialize = tf.variables_initializer(
                tf.global_variables(), name='initialize')
            saver = Saver(config)
        self.graph = graph
        self.model = model
        self.state = state
        self.state_update = state_update
        self.update_state = update_state
        self.parameters = parameters
        self.train = train
        self.logger = logger
        self.summary = summary
        self.initialize = initialize
        self.saver = saver

    def count_parameters(self):
        return np.sum([int(np.prod(p.get_shape())) for p in self.parameters])

    def run(self, target, manager, config):
        support.log(self, 'Parameters: {}', self.count_parameters())
        session = tf.Session(graph=self.graph)
        session.run(self.initialize)
        self.saver.restore(session)
        state = State.deserialize(session.run(self.state))
        for _ in range(config.epoch_count - state.epoch % config.epoch_count):
            self._run_epoch(target, manager, config, session, state)
            state.increment_epoch()
            session.run(self.update_state, {
                self.state_update: state.serialize(),
            })
            self.saver.save(session)

    def _run_epoch(self, target, manager, config, session, state):
        for _ in range(target.train_sample_count):
            if manager.should_train(state.time):
                self._run_train(target, manager, config, session, state)
            if manager.should_test(state.time):
                self._run_test(target, manager, config, session, state)
            if manager.should_show(state.time):
                self._run_show(target, manager, config, session, state)
            state.increment_time()

    def _run_show(self, target, manager, config, session, state):
        sample = target.train((state.sample + 1) % target.train_sample_count)
        step_count = sample.shape[0]
        feed = {
            self.model.start: self._zero_start(),
        }
        fetch = {
            'y_hat': self.model.y_hat,
            'finish': self.model.finish,
        }
        for i in range(step_count):
            feed[self.model.x] = np.reshape(
                sample[:(i + 1), :], [1, i + 1, -1])
            y_hat = np.zeros([step_count, target.dimension_count])
            for j in range(step_count - i - 1):
                result = session.run(fetch, feed)
                feed[self.model.start] = result['finish']
                y_hat[j, :] = result['y_hat'][-1, :]
                feed[self.model.x] = np.reshape(y_hat[j, :], [1, 1, -1])
            if not manager.show(support.shift(sample, -i - 1), y_hat):
                break

    def _run_test(self, target, manager, config, session, state):
        manager.test()

    def _run_train(self, target, manager, config, session, state):
        sample = target.train(state.sample)
        feed = {
            self.model.start: self._zero_start(),
            self.model.x: np.reshape(sample, [1, -1, target.dimension_count]),
            self.model.y: np.reshape(
                support.shift(sample, -1), [1, -1, target.dimension_count]),
        }
        fetch = {
            'train': self.train,
            'loss': self.model.loss,
            'summary': self.summary,
        }
        result = session.run(fetch, feed)
        loss = result['loss'].flatten()
        assert(np.all([not math.isnan(loss) for loss in loss]))
        self.logger.add_summary(result['summary'], state.time)
        manager.train(loss, state)

    def _zero_start(self):
        return np.zeros(self.model.start.get_shape(), np.float32)

class Model:
    def __init__(self, config):
        self.x = tf.placeholder(
            tf.float32, [1, None, config.dimension_count], name='x')
        self.y = tf.placeholder(
            tf.float32, [1, None, config.dimension_count], name='y')
        with tf.variable_scope('network'):
            self.start, self.finish, h = Model._network(self.x, config)
        with tf.variable_scope('regression'):
            self.y_hat, self.loss = Model._regress(h, self.y, config)

    def _finalize(state, config):
        parts = []
        for i in range(config.layer_count):
            parts.append(state[i].c)
            parts.append(state[i].h)
        return tf.pack(parts, name='finish')

    def _initialize(config):
        start = tf.placeholder(
            tf.float32, [2 * config.layer_count, 1, config.unit_count],
            name='start')
        parts = tf.unpack(start)
        state = []
        for i in range(config.layer_count):
            c, h = parts[2 * i], parts[2*i + 1]
            state.append(tf.nn.rnn_cell.LSTMStateTuple(c, h))
        return start, tuple(state)

    def _network(x, config):
        cell = tf.nn.rnn_cell.LSTMCell(
            config.unit_count, state_is_tuple=True, cell_clip=config.cell_clip,
            forget_bias=config.forget_bias, use_peepholes=config.use_peepholes,
            initializer=config.network_initializer)
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [cell] * config.layer_count, state_is_tuple=True)
        start, state = Model._initialize(config)
        h, state = tf.nn.dynamic_rnn(
            cell, x, initial_state=state, parallel_iterations=1)
        finish = Model._finalize(state, config)
        return start, finish, h

    def _regress(x, y, config):
        unroll_count = tf.shape(x)[1]
        x = tf.squeeze(x, squeeze_dims=[0])
        y = tf.squeeze(y, squeeze_dims=[0])
        w = tf.get_variable(
            'w', [config.unit_count, config.dimension_count],
            initializer=config.regression_initializer)
        b = tf.get_variable('b', [1, config.dimension_count])
        y_hat = tf.matmul(x, w) + tf.tile(b, [unroll_count, 1])
        loss = tf.reduce_mean(tf.squared_difference(y_hat, y))
        return y_hat, loss

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

    def show(self, y, y_hat):
        with self.lock:
            for listener in self.listeners:
                listener.put((y, y_hat))
        return len(self.listeners) > 0

    def test(self):
        pass

    def train(self, loss, state):
        if not self.train_report_schedule.should(state.time):
            return
        time, epoch, sample = state.time + 1, state.epoch + 1, state.sample + 1
        line = '{:10d} {:4d} {:10d} ({:6.2f}%)'.format(
            time, epoch, sample, 100 * sample / self.train_sample_count)
        for loss in loss:
            line += ' {:12.4e}'.format(loss)
        support.log(self, line)

    def _show_client(self, connection, address):
        support.log(self, 'New listener: {}', address)
        listener = queue.Queue()
        with self.lock:
            self.listeners[listener] = True
        try:
            client = connection.makefile(mode='w')
            while True:
                y, y_hat = listener.get()
                values = [str(value) for value in y.flatten()]
                client.write(','.join(values) + ',')
                values = [str(value) for value in y_hat.flatten()]
                client.write(','.join(values) + '\n')
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

class Saver:
    def __init__(self, config):
        self.backend = tf.train.Saver()
        self.path = config.save_path

    def save(self, session):
        path = self.backend.save(session, self.path)
        support.log(self, 'New checkpoint: {}', path)

    def restore(self, session):
        if len(glob.glob('{}*'.format(self.path))) > 0:
            if input('Restore "{}"? '.format(self.path)) != 'no':
                self.backend.restore(session, self.path)

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
        trace_count = 0
        with open(config.index_path, 'r') as file:
            for record in file:
                trace_count += 1
                record = record.split(',')
                length = int(record[-1])
                if length < config.min_length:
                    continue
                if length > config.max_length:
                    continue
                sample = (record[0], int(record[1]), int(record[2]))
                if random.random() < config.train_fraction:
                    self.train_samples.append(sample)
                else:
                    self.test_samples.append(sample)
        random.shuffle(self.train_samples)
        random.shuffle(self.test_samples)
        self.train_sample_count = len(self.train_samples)
        self.test_sample_count = len(self.test_samples)
        sample_count = self.train_sample_count + self.test_sample_count
        support.log(self, 'Samples: {} ({:.2f}%)', sample_count,
                    100 * sample_count / trace_count)
        self.standard = self._standardize(config.standard_count)
        support.log(self, 'Mean: {:e}, deviation: {:e} ({} samples)',
                    self.standard[0], self.standard[1], config.standard_count)

    def test(self, sample):
        return (self._test(sample) - self.standard[0]) / self.standard[1]

    def train(self, sample):
        return (self._train(sample) - self.standard[0]) / self.standard[1]

    def _test(self, sample):
        return task_usage.select(*self.test_samples[sample])

    def _train(self, sample):
        return task_usage.select(*self.train_samples[sample])

    def _standardize(self, count):
        standard = (0.0, 1.0)
        data = np.array([], dtype=np.float32)
        for sample in range(count):
            data = np.append(data, self._train(sample))
        if len(data) > 0:
            standard = (np.mean(data), np.std(data))
        return standard

def main(config):
    target = Target(config)
    config.update({
        'dimension_count': target.dimension_count,
    })
    learn = Learn(config)
    config.update({
        'train_sample_count': target.train_sample_count,
    })
    manager = Manager(config)
    learn.run(target, manager, config)

if __name__ == '__main__':
    assert(len(sys.argv) == 2)
    support.loggalize()
    config = Config({
        # Target
        'index_path': sys.argv[1],
        'min_length': 0,
        'max_length': 50,
        'standard_count': 1000,
        # Modeling
        'layer_count': 1,
        'unit_count': 200,
        'cell_clip': 1.0,
        'forget_bias': 1.0,
        'use_peepholes': True,
        'network_initializer': tf.random_uniform_initializer(-0.01, 0.01),
        'regression_initializer': tf.random_normal_initializer(stddev=0.01),
        # Training
        'train_fraction': 0.7,
        'gradient_clip': 1.0,
        'learning_rate': 1e-3,
        'epoch_count': 100,
        # Managing
        'train_schedule': [0, 1],
        'train_report_schedule': [1000 - 1, 1],
        'test_schedule': [10000 - 10, 10],
        'show_schedule': [10000 - 10, 10],
        'show_address': ('0.0.0.0', 4242),
        # Other
        'log_path': os.path.join('output', 'log'),
        'save_path': os.path.join('output', 'model'),
    })
    random.seed(0)
    main(config)
