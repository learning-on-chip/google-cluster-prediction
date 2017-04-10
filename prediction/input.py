from . import database
from . import support
from .index import Index
from .random import Random
import glob
import hashlib
import numpy as np
import os
import tensorflow as tf


class Input:
    class Part:
        def __init__(self, path):
            pattern = os.path.join(path, '**', '*.tfrecords')
            self.paths = sorted(glob.glob(pattern, recursive=True))
            self.restart()

        def copy(self):
            copy = Input.Part.__new__(Input.Part)
            copy.paths = self.paths
            copy.restart()
            return copy

        def iterate(self):
            for record in self._iterate():
                yield _parse(record)

        def next(self):
            return _parse(next(self.iterator))

        def restart(self, seed=0):
            random_state = Random.get().get_state()
            Random.get().seed(seed)
            index = Random.get().permutation(len(self.paths))
            Random.get().set_state(random_state)
            self.iterator = self._iterate(index)

        def _iterate(self, index=None):
            for i in index if not index is None else range(len(self.paths)):
                for record in tf.python_io.tf_record_iterator(self.paths[i]):
                    yield record
            raise StopIteration()

    def __init__(self, config):
        self.dimension_count = 1
        klass, training_path, validation_path, test_path = _identify(config)
        support.log(self, 'Training path: {}', training_path)
        support.log(self, 'Validation path: {}', validation_path)
        support.log(self, 'Test path: {}', test_path)
        if not os.path.exists(training_path) or \
           not os.path.exists(validation_path) or \
           not os.path.exists(test_path):
           klass._prepare(training_path, validation_path, test_path, config)
        self.training = Input.Part(training_path)
        self.validation = Input.Part(validation_path)
        self.test = Input.Part(test_path)

    def copy(self):
        copy = Input.__new__(Input)
        copy.dimension_count = self.dimension_count
        copy.training = self.training.copy()
        copy.validation = self.validation.copy()
        copy.test = self.test.copy()
        return copy


class Fake:
    def _distribute(path, metas):
        _distribute(path, metas, lambda i: Fake._fetch(*metas[i]))

    def _fetch(a, b, n):
        return np.sin(a * np.linspace(0, n - 1, n) + b).tolist()

    def _generate(count):
        metas = Random.get().rand(count, 3)
        metas[:, 0] = 0.5 + 1.5 * metas[:, 0]
        metas[:, 1] = 5 * metas[:, 1]
        metas[:, 2] = np.round(5 + 15 * metas[:, 2])
        return metas

    def _prepare(training_path, validation_path, test_path, config):
        _, training_count, validation_count, test_count = \
            _partition(config.max_sample_count, config)
        training_metas = Fake._generate(training_count)
        validation_metas = Fake._generate(validation_count)
        test_metas = Fake._generate(test_count)
        if not os.path.exists(training_path):
            Fake._distribute(training_path, training_metas)
        if not os.path.exists(validation_path):
            Fake._distribute(validation_path, validation_metas)
        if not os.path.exists(test_path):
            Fake._distribute(test_path, test_metas)


class Real:
    def _distribute(path, metas):
        _distribute(path, [meta[1:] for meta in metas],
                    lambda i: database.select_task_usage(*metas[i]))

    def _index(config):
        metas = []
        def _callback(path, job, task, length, **_):
            if length < config.min_sample_length:
                return
            if length > config.max_sample_length:
                return
            metas.append((path, job, task))
        found_count = Index.decode(config.path, _callback)
        support.log(Real, 'Found samples: {}', found_count)
        support.log(Real, 'Selected samples: {}',
                    support.format_percentage(len(metas), found_count))
        Random.get().shuffle(metas)
        return metas

    def _partition(metas, config):
        preserved_count, training_count, validation_count, test_count = \
            _partition(len(metas), config)
        support.log(Real, 'Preserved samples: {}',
                    support.format_percentage(preserved_count, len(metas)))
        metas = metas[:preserved_count]
        training_metas = metas[:training_count]
        metas = metas[training_count:]
        validation_metas = metas[:validation_count]
        metas = metas[validation_count:]
        test_metas = metas[:test_count]
        metas = metas[test_count:]
        assert(len(metas) == 0)
        return training_metas, validation_metas, test_metas

    def _prepare(training_path, validation_path, test_path, config):
        support.log(Real, 'Index path: {}', config.path)
        training_metas, validation_metas, test_metas = \
            Real._partition(Real._index(config), config)
        if not os.path.exists(training_path):
            Real._distribute(training_path, training_metas)
        if not os.path.exists(validation_path):
            Real._distribute(validation_path, validation_metas)
        if not os.path.exists(test_path):
            Real._distribute(test_path, test_metas)


def _distribute(path, metas, fetch, separator=',', granularity=2):
    os.makedirs(path)
    count = len(metas)
    support.log('Distribute samples: {}, path: {}', count, path)
    names = [separator.join([str(meta) for meta in meta]) for meta in metas]
    names = [hashlib.md5(name.encode('utf-8')).hexdigest() for name in names]
    names = [name[:granularity] for name in names]
    seen = {}
    for i in range(count):
        if names[i] in seen:
            continue
        seen[names[i]] = True
        writer = tf.python_io.TFRecordWriter(
            os.path.join(path, names[i] + '.tfrecords'))
        for j in range(i, count):
            if names[i] != names[j]:
                continue
            data = fetch(j)
            feature = tf.train.Feature(
                float_list=tf.train.FloatList(value=data))
            example = tf.train.Example(
                features=tf.train.Features(feature={'data': feature}))
            writer.write(example.SerializeToString())
        writer.close()

def _identify(config):
    real = 'path' in config
    klass = Real if real else Fake
    path = os.path.dirname(config.path) if real else 'input'
    key = hashlib.md5(support.tokenize(config).encode('utf-8')).hexdigest()
    path = os.path.join(path, 'data-' + key)
    return [
        klass,
        os.path.join(path, 'training'),
        os.path.join(path, 'validation'),
        os.path.join(path, 'test'),
    ]

def _parse(record):
    example = tf.train.Example()
    example.ParseFromString(record)
    data = example.features.feature['data'].float_list.value
    return np.reshape([value for value in data], [-1, 1])

def _partition(count, config):
    preserved_count = min(count, config.max_sample_count)
    training_count = int(config.training_fraction * preserved_count)
    assert(training_count > 0)
    validation_count = int(config.validation_fraction * preserved_count)
    assert(validation_count > 0)
    test_count = preserved_count - training_count - validation_count
    assert(test_count > 0)
    return preserved_count, training_count, validation_count, test_count
