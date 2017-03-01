from model import Model
import glob
import numpy as np
import support
import tensorflow as tf


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
                    tf.int64, shape=3, name='state_update')
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
