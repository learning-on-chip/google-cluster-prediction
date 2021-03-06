import datetime
import glob
import inspect
import logging
import numpy as np
import os


class Progress:
    def __init__(self, subject=None, description=None, total_count=None,
                 report_each=None):
        self.subject = subject
        self.description = description
        self.total_count = total_count
        self.report_each = report_each

    def advance(self, count=1):
        for _ in range(count):
            self._advance()

    def finish(self):
        log(self.subject, 'Done {} ({})'.format(self.description,
                                                self.done_count))

    def start(self):
        self.done_count = 0
        log(self.subject, 'Start {}...'.format(self.description))

    def _advance(self):
        self.done_count += 1
        if self.total_count is not None and self.done_count > self.total_count:
            raise Exception(
                'exceeded the total count ({})'.format(self.total_count))
        if self.report_each is None:
            return
        if self.done_count % self.report_each != 0:
            return
        if self.total_count is None:
            message = self.done_count
        else:
            message = format_percentage(self.done_count, self.total_count)
        log(self.subject, '{}: {}', title(self.description), message)


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

def loggalize(path=None, level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handler = logging.FileHandler(path)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

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
    pattern = os.path.join(glob.escape(path), '**', name)
    return sorted(glob.glob(pattern, recursive=True))

def title(text):
    return text[0].upper() + text[1:]

def tokenize(object):
    if not isinstance(object, dict):
        object = str(object).lower()
        for this, that in [(' ', ''), ('[', '('), (']', ')')]:
            object = object.replace(this, that)
    else:
        chunks = []
        for key in sorted(object.keys()):
            alias = ''.join([chunk[0] for chunk in key.split('_')]).lower()
            chunks.append('{}={}'.format(alias, tokenize(object[key])))
        object = ','.join(chunks)
    return object
