from . import database
from . import support
from .index import Index
from .random import Random
import numpy as np


class BaseInput:
    class Part:
        def __init__(self, samples):
            self.sample_count = len(samples)
            self.samples = samples
            self.index = np.arange(self.sample_count)

        def get(self, sample):
            return self._get(self.index[sample])

        def shuffle(self):
            self.index = Random.get().permutation(self.sample_count)

    def __init__(self, training, validation, test):
        self.dimension_count = 1
        self.training = training
        self.validation = validation
        self.test = test
        support.log(self, 'Training samples: {}', training.sample_count)
        support.log(self, 'Validation samples: {}', validation.sample_count)
        support.log(self, 'Test samples: {}', test.sample_count)

    def on_epoch(self, state):
        random_state = Random.get().get_state()
        Random.get().seed(state.epoch)
        self.training.shuffle()
        Random.get().set_state(random_state)


class FakeInput(BaseInput):
    class Part(BaseInput.Part):
        def __init__(self, samples):
            super(FakeInput.Part, self).__init__(samples)

        def copy(self):
            return FakeInput.Part(self.samples)

        def _get(self, sample):
            return FakeInput._compute(self.samples[sample, :])

    def __init__(self, config):
        _, training_count, validation_count, test_count = \
            _partition(config.max_sample_count, config)
        super(FakeInput, self).__init__(
            FakeInput.Part(FakeInput._generate(training_count)),
            FakeInput.Part(FakeInput._generate(validation_count)),
            FakeInput.Part(FakeInput._generate(test_count)))

    def copy(self):
        copy = FakeInput.__new__(FakeInput)
        super(FakeInput, copy).__init__(
            self.training.copy(), self.validation.copy(), self.test.copy())
        return copy

    def _compute(sample):
        a, b, n = sample[0], sample[1], int(sample[2])
        return np.reshape(np.sin(a * np.linspace(0, n - 1, n) + b), (-1, 1))

    def _generate(count):
        samples = Random.get().rand(count, 3)
        samples[:, 0] = 0.5 + 1.5 * samples[:, 0]
        samples[:, 1] = 5 * samples[:, 1]
        samples[:, 2] = np.round(5 + 15 * samples[:, 2])
        return samples


class RealInput(BaseInput):
    class Part(BaseInput.Part):
        def __init__(self, samples, standard):
            super(RealInput.Part, self).__init__(samples)
            self.standard = standard

        def copy(self):
            return RealInput.Part(self.samples, self.standard)

        def _get(self, sample):
            data = database.select_task_usage(*self.samples[sample])
            return (data - self.standard[0]) / self.standard[1]

    def __init__(self, config):
        support.log(self, 'Input path: {}', config.path)
        samples = []
        def _process(path, job, task, length, **_):
            if length < config.min_sample_length:
                return
            if length > config.max_sample_length:
                return
            samples.append((path, job, task))
        processed_count = Index.decode(config.path, _process)
        Random.get().shuffle(samples)
        constrained_count = len(samples)
        preserved_count, training_count, validation_count, test_count = \
            _partition(constrained_count, config)
        samples = samples[:preserved_count]
        support.log(self, 'Processed samples: {}', processed_count)
        support.log(
            self, 'Constrained samples: {}',
            support.format_percentage(constrained_count, processed_count))
        support.log(
            self, 'Preserved samples: {}',
            support.format_percentage(preserved_count, processed_count))
        training_samples = samples[:training_count]
        samples = samples[training_count:]
        validation_samples = samples[:validation_count]
        samples = samples[validation_count:]
        test_samples = samples[:test_count]
        samples = samples[test_count:]
        assert(len(samples) == 0)
        standard_count = min(config.standard_count, training_count)
        standard = RealInput._standardize(training_samples, standard_count)
        support.log(self, 'Mean: {:e}, deviation: {:e} ({} samples)',
                    standard[0], standard[1], standard_count)
        super(RealInput, self).__init__(
            RealInput.Part(training_samples, standard),
            RealInput.Part(validation_samples, standard),
            RealInput.Part(test_samples, standard))

    def copy(self):
        copy = RealInput.__new__(RealInput)
        super(RealInput, copy).__init__(
            self.training.copy(), self.validation.copy(), self.test.copy())
        return copy

    def _standardize(samples, count):
        data = np.array([], dtype=np.float32)
        for sample in Random.get().permutation(len(samples))[:count]:
            data = np.append(
                data, database.select_task_usage(*samples[sample]))
        if len(data) > 0:
            return (np.mean(data), np.std(data))
        else:
            return (0.0, 1.0)


def Input(config):
    if config.get('path') is not None:
        return RealInput(config)
    else:
        return FakeInput(config)


def _partition(count, config):
    preserved_count = min(count, config.max_sample_count)
    training_count = int(config.training_fraction * preserved_count)
    assert(training_count > 0)
    validation_count = int(config.validation_fraction * preserved_count)
    assert(validation_count > 0)
    test_count = preserved_count - training_count - validation_count
    assert(test_count > 0)
    return preserved_count, training_count, validation_count, test_count
