import matplotlib.pyplot as pp
import numpy as np

def figure(width=14, height=6):
    pp.figure(figsize=(width, height), dpi=80, facecolor='w', edgecolor='k')

def normalize(data):
    return (data - np.mean(data)) / np.sqrt(np.var(data))

def standardize(data):
    unique = np.unique(data)
    for i, value in enumerate(unique):
        data[data == value] = i
    return data / len(unique)
