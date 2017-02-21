import datetime
import logging
import numpy as np


class Config:
    def __init__(self, options={}):
        self.update(options)

    def has(self, key):
        return hasattr(self, key)

    def update(self, options):
        for key in options:
            setattr(self, key, options[key])


def figure(width=14, height=6):
    import matplotlib.pyplot as pp
    pp.figure(figsize=(width, height), dpi=80, facecolor='w', edgecolor='k')

def format_timestamp():
    return '{:%Y-%m-%d %H-%M-%S}'.format(datetime.datetime.now())

def format_percentage(count, total):
    return '{} ({:.2f}%)'.format(count, 100 * count / total)

def log(*arguments, limit=7):
    arguments = list(arguments)
    first = arguments.pop(0)
    if isinstance(first, str):
        template, source = first, 'Main'
    else:
        template, source = arguments.pop(0), first.__class__.__name__
    if len(source) > limit:
        source = source[:(limit - 1)] + '…'
    logging.info('[%-' + str(limit) + 's] %s', source.upper(),
                 template.format(*arguments))

def loggalize(level=logging.INFO):
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', level=level)

def shift(data, amount, padding=0.0):
    data = np.roll(data, amount, axis=0)
    if amount > 0:
        data[:amount, :] = padding
    elif amount < 0:
        data[amount:, :] = padding
    return data
