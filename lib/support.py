import logging
import matplotlib.pyplot as pp
import numpy as np

LOG_SOURCE_LIMIT = 8

class Config:
    def __init__(self, options={}):
        self.update(options)

    def update(self, options):
        for key in options:
            setattr(self, key, options[key])

def figure(width=14, height=6):
    pp.figure(figsize=(width, height), dpi=80, facecolor='w', edgecolor='k')

def log(*arguments):
    arguments = list(arguments)
    first = arguments.pop(0)
    if isinstance(first, str):
        template, source = first, 'Main'
    else:
        template, source = arguments.pop(0), first.__class__.__name__
    if len(source) > LOG_SOURCE_LIMIT:
        source = source[:(LOG_SOURCE_LIMIT - 1)] + '…'
    logging.info('[%-' + str(LOG_SOURCE_LIMIT) + 's] %s', source.upper(),
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
