import logging
import matplotlib.pyplot as pp
import numpy as np

class Config:
    def __init__(self, options={}):
        self.update(options)

    def update(self, options):
        for key in options:
            setattr(self, key, options[key])

def figure(width=14, height=6):
    pp.figure(figsize=(width, height), dpi=80, facecolor='w', edgecolor='k')

def log(source, message, *arguments):
    if not isinstance(source, str):
        source = source.__class__.__name__
    if len(source) > 10:
        source = source[:9] + '…'
    logging.info("[%-10s] %s", source.upper(), message.format(*arguments))

def normalize(data):
    return (data - np.mean(data)) / np.sqrt(np.var(data))

def shift(data, amount, padding=0.0):
    data = np.roll(data, amount, axis=0)
    if amount > 0:
        data[:amount, :] = padding
    elif amount < 0:
        data[amount:, :] = padding
    return data

def standardize(data):
    unique = np.unique(data)
    for i, value in enumerate(unique):
        data[data == value] = i
    return data / len(unique)
