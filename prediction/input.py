from . import database
from . import support
from .index import Index
from .random import Random
import hashlib
import json
import numpy as np
import os
import tensorflow as tf


class Input:
    def __init__(self, config):
        klass, training_path, validation_path, testing_path = \
            Input._identify(config)
        support.log(self, 'Training path: {}', training_path)
        support.log(self, 'Validation path: {}', validation_path)
        support.log(self, 'Testing path: {}', testing_path)
        if not os.path.exists(training_path) or \
           not os.path.exists(validation_path) or \
           not os.path.exists(testing_path):
            training_metas, validation_metas, testing_metas = klass._prepare(
                training_path, validation_path, testing_path, config)
            standard = Input._standartize(training_metas, klass._fetch)
            support.log(self, 'Standard mean: {}, deviation: {}', *standard)
            if not os.path.exists(training_path):
                Input._distribute(training_path, training_metas,
                                  klass._fetch, standard=standard)
            if not os.path.exists(validation_path):
                Input._distribute(validation_path, validation_metas,
                                  klass._fetch, standard=standard)
            if not os.path.exists(testing_path):
                Input._distribute(testing_path, testing_metas,
                                  klass._fetch, standard=standard)
        self.config = config.copy()
        self.config.training.path = training_path
        self.config.validation.path = validation_path
        self.config.testing.path = testing_path

    def __call__(self, target):
        return Instance(self.config[target])

    def _collect(path):
        paths = support.scan(path, '*.tfrecords')
        meta = json.loads(open(os.path.join(path, 'meta.json')).read())
        assert(meta['path_count'] == len(paths))
        return paths, meta

    def _distribute(path, metas, fetch, standard=(0, 1), separator=',',
                    granularity=2, report_each=10000):
        os.makedirs(path)
        sample_count = len(metas)
        names = [separator.join([str(meta) for meta in meta])
                 for meta in metas]
        names = [hashlib.md5(name.encode('utf-8')).hexdigest()[:granularity]
                 for name in names]
        seen = {}
        progress = support.Progress(description=('distributing ' + path),
                                    total_count=sample_count,
                                    report_each=report_each)
        progress.start()
        for i in range(sample_count):
            if names[i] in seen:
                continue
            seen[names[i]] = True
            writer = tf.python_io.TFRecordWriter(
                os.path.join(path, names[i] + '.tfrecords'))
            for j in range(i, sample_count):
                if names[i] != names[j]:
                    continue
                data = (fetch(*metas[j]) - standard[0]) / standard[1]
                writer.write(Input._encode(data))
                progress.advance()
            writer.close()
        progress.finish()
        with open(os.path.join(path, 'meta.json'), 'w') as file:
            file.write(json.dumps({
                'dimension_count': 1,
                'path_count': len(seen),
                'sample_count': sample_count,
            }))

    def _encode(data):
        float_list = tf.train.FloatList(value=np.ravel(data).tolist())
        feature = tf.train.Feature(float_list=float_list)
        features = tf.train.Features(feature={'data': feature})
        example = tf.train.Example(features=features)
        return example.SerializeToString()

    def _identify(config):
        real = 'path' in config
        klass = Real if real else Fake
        path = os.path.dirname(config.path) if real else 'input'
        key = support.tokenize(config).encode('utf-8')
        key = hashlib.md5(key).hexdigest()
        path = os.path.join(path, 'data-' + key)
        return [
            klass,
            os.path.join(path, 'training'),
            os.path.join(path, 'validation'),
            os.path.join(path, 'testing'),
        ]

    def _partition(available, config):
        preserved = min(available, config.max_sample_count)
        training = int(config.training.fraction * preserved)
        validation = int(config.validation.fraction * preserved)
        testing = int(config.testing.fraction * preserved)
        assert(preserved >= training + validation + testing)
        return preserved, training, validation, testing

    def _standartize(metas, fetch, report_each=10000):
        sample_count = len(metas)
        standard = support.Standard()
        progress = support.Progress(description='standardizing',
                                    total_count=sample_count,
                                    report_each=report_each)
        progress.start()
        for i in range(sample_count):
            standard.consume(fetch(*metas[i]))
            progress.advance()
        progress.finish()
        return standard.compute()


class Instance:
    def __init__(self, config):
        paths, meta = Input._collect(config.path)
        self.dimension_count = meta['dimension_count']
        self.sample_count = meta['sample_count']
        with tf.variable_scope('state'):
            self.state = State()
        with tf.variable_scope('source'):
            paths = tf.Variable(paths, name='paths', dtype=tf.string,
                                trainable=False)
            queue = tf.FIFOQueue(meta['path_count'], [tf.string])
            enqueue = queue.enqueue_many([tf.random_shuffle(paths)])
            tf.train.add_queue_runner(tf.train.QueueRunner(queue, [enqueue]))
            _, record = tf.TFRecordReader().read(queue)
        with tf.variable_scope('x'):
            features = tf.parse_single_example(record, {
                'data': tf.VarLenFeature(tf.float32),
            })
            data = tf.sparse_tensor_to_dense(features['data'])
            self.x = tf.reshape(data, [1, -1, self.dimension_count])
        with tf.variable_scope('y'):
            self.y = tf.pad(self.x[:, 1:, :], [[0, 0], [0, 1], [0, 0]])

    def iterate(self, session, step_count=None):
        for i in range(step_count if step_count else self.sample_count):
            self.state.advance()
            yield i
        raise StopIteration()

    def restore(self, session):
        self.state.restore(session)

    def save(self, session):
        self.state.save(session)

    @property
    def step(self):
        return self.state.step


class Fake:
    def _fetch(a, b, n):
        x = np.linspace(0, n - 1, n, dtype=np.float32)
        return np.reshape(np.sin(a * x + b), [-1, 1])

    def _generate(count):
        metas = Random.get().rand(count, 3)
        metas[:, 0] = 0.5 + 1.5 * metas[:, 0]
        metas[:, 1] = 5 * metas[:, 1]
        metas[:, 2] = np.round(5 + 15 * metas[:, 2])
        return metas

    def _prepare(training_path, validation_path, testing_path, config):
        _, training_count, validation_count, testing_count = \
            Input._partition(config.max_sample_count, config)
        training_metas = Fake._generate(training_count)
        validation_metas = Fake._generate(validation_count)
        testing_metas = Fake._generate(testing_count)
        return training_metas, validation_metas, testing_metas


class Real:
    def _fetch(*meta):
        return database.select_task_usage(*meta)

    def _index(config):
        metas = []
        def _callback(path, job, task, length, **_):
            if length < config.min_sample_length:
                return
            if length > config.max_sample_length:
                return
            metas.append((path, job, task))
        found_count = Index.decode(config.path, _callback)
        selected_count = len(metas)
        support.log(Real, 'Found samples: {}', found_count)
        support.log(Real, 'Selected samples: {}',
                    support.format_percentage(selected_count, found_count))
        Random.get().shuffle(metas)
        return metas

    def _prepare(training_path, validation_path, testing_path, config):
        support.log(Real, 'Index path: {}', config.path)
        metas = Real._index(config)
        available_count = len(metas)
        preserved_count, training_count, validation_count, testing_count = \
            Input._partition(available_count, config)
        support.log(Real, 'Preserved samples: {}',
                    support.format_percentage(preserved_count,
                                              available_count))
        metas = metas[:preserved_count]
        training_metas = metas[:training_count]
        metas = metas[training_count:]
        validation_metas = metas[:validation_count]
        metas = metas[validation_count:]
        testing_metas = metas[:testing_count]
        return training_metas, validation_metas, testing_metas


class State:
    def __init__(self):
        state = np.zeros(1, dtype=np.int64)
        self.current = tf.Variable(state, name='current', dtype=tf.int64,
                                   trainable=False)
        self.new = tf.placeholder(tf.int64, shape=state.shape, name='new')
        self.assign_new = self.current.assign(self.new)
        self.step = None

    def advance(self):
        self.step += 1

    def restore(self, session):
        state = session.run(self.current)
        self.step = state[0]
        support.log(self, 'Restored step: {}'.format(self.step))

    def save(self, session):
        feed = {
            self.new: [self.step],
        }
        session.run(self.assign_new, feed)
