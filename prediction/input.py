from . import database
from . import support
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
            self.index = np.random.permutation(self.sample_count)

    def __init__(self, train, test):
        self.dimension_count = 1
        self.train = train
        self.test = test

    def on_epoch(self, state):
        random_state = np.random.get_state()
        np.random.seed(state.epoch)
        self.train.shuffle()
        np.random.set_state(random_state)


class FakeInput(BaseInput):
    class Part(BaseInput.Part):
        def __init__(self, samples):
            super(FakeInput.Part, self).__init__(samples)

        def copy(self):
            return FakeInput.Part(self.samples)

        def _get(self, sample):
            return FakeInput._compute(self.samples[sample, :])

    def __init__(self, config):
        sample_count = 10000
        train_sample_count = int(config.train_fraction * sample_count)
        test_sample_count = sample_count - train_sample_count
        super(FakeInput, self).__init__(
            FakeInput.Part(FakeInput._generate(train_sample_count)),
            FakeInput.Part(FakeInput._generate(test_sample_count)))

    def copy(self):
        copy = FakeInput.__new__(FakeInput)
        super(FakeInput, copy).__init__(self.train.copy(), self.test.copy())
        return copy

    def _compute(sample):
        a, b, n = sample[0], sample[1], int(sample[2])
        return np.reshape(np.sin(a * np.linspace(0, n - 1, n) + b), (-1, 1))

    def _generate(count):
        samples = np.random.rand(count, 3)
        samples[:, 0] = 0.5 + 1.5 * samples[:, 0]
        samples[:, 1] = 5 * samples[:, 1]
        samples[:, 2] = np.round(5 + 45 * samples[:, 2])
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
        found_count = 0
        samples = []
        with open(config.path, 'r') as file:
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
        standard = RealInput._standardize(train_samples, standard_count)
        support.log(self, 'Mean: {:e}, deviation: {:e} ({} samples)',
                    standard[0], standard[1], standard_count)
        super(RealInput, self).__init__(
            RealInput.Part(train_samples, standard),
            RealInput.Part(test_samples, standard))

    def copy(self):
        copy = RealInput.__new__(RealInput)
        super(RealInput, copy).__init__(self.train.copy(), self.test.copy())
        return copy

    def _standardize(samples, count):
        data = np.array([], dtype=np.float32)
        for sample in np.random.permutation(len(samples))[:count]:
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
