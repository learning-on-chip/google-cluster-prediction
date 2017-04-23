import datetime
import glob
import inspect
import logging
import numpy as np
import os
import sys


class Manager:
    def __init__(self, config):
        self.config = config

    def __getattr__(self, name):
        assert(name.startswith('should_'))
        name = name.replace('should_', '')
        period = self.config.get(name + '_period', sys.maxsize)
        return lambda step: step > 0 and step % period == 0


class Progress:
    def __init__(self, subject=None, description=None, total_count=None,
                 report_each=None):
        self.subject = subject
        self.description = description
        self.total_count = total_count
        self.report_each = report_each
        self.finish_count = 0
        self.finished = False
        log(self.subject, 'Start {}...'.format(self.description))

    def advance(self, count=1):
        self.finish_count += count
        if self.finish_count == self.total_count:
            self.finish()
        elif self.report_each is not None and \
             self.finish_count % self.report_each == 0:
            log(self.subject, 'Finished: {}',
                format_percentage(self.finish_count, self.total_count))

    @property
    def count(self):
        return self.finish_count

    def finish(self):
        if not self.finished:
            self.finished = True
            log(self.subject,
                'Finished {} ({})'.format(self.description, self.count))


class Standard:
    def __init__(self):
        self.s, self.m, self.v, self.k = None, None, None, 0

    def compute(self):
        return (self.s / self.k, np.sqrt(self.v / (self.k - 1)))

    def consume(self, data):
        for value in data.flat:
            self.k += 1
            if self.k == 1:
                self.s = value
                self.m = value
                self.v = 0
            else:
                m = self.m
                self.s += value
                self.m += (value - self.m) / self.k
                self.v += (value - m) * (value - self.m)


def default_output():
    return os.path.join('output', format_timestamp())

def format_timestamp():
    return '{:%Y-%m-%d %H-%M-%S}'.format(datetime.datetime.now())

def format_percentage(count, total):
    return '{} ({:.2f}%)'.format(count, 100 * count / total)

def log(*arguments):
    arguments = list(arguments)
    first = arguments.pop(0)
    if first is None:
        log(*arguments)
        return
    if isinstance(first, str):
        message = first
        name = ''
        number = ''
    elif inspect.isclass(first):
        message = arguments.pop(0)
        name = first.__name__
        number = ''
    else:
        message = arguments.pop(0)
        name = first.__class__.__name__
        number = str(id(first))
    name = '{:5}'.format(name)[:5]
    number = '{:5}'.format(number)[-5:]
    logging.info('[%s|%s] %s', name, number, message.format(*arguments))

def loggalize(level=logging.INFO):
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', level=level)

def prompt(*options):
    print('Choose one of the following options:')
    for i, option in enumerate(options):
        print('    {}. {}'.format(i, option))
    while True:
        try:
            i = int(input('Your choice: '))
        except ValueError:
            continue
        if i < 0 or i >= len(options):
            continue
        return i

def scan(path, name):
    pattern = os.path.join(path, '**', name)
    return sorted(glob.glob(pattern, recursive=True))

def shift(data, amount, axis=0, padding=0):
    data = np.roll(data, amount, axis=axis)
    if amount > 0:
        data[:amount, :] = padding
    elif amount < 0:
        data[amount:, :] = padding
    return data

def tokenize(dictionary):
    chunks = []
    for key in sorted(dictionary.keys()):
        alias = ''.join([chunk[0] for chunk in key.split('_')])
        value = str(dictionary[key]).replace(' ', '')
        chunks.append('{}={}'.format(alias, value))
    return ','.join(chunks).lower()
