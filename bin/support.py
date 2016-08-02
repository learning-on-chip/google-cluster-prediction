import math, sqlite3
import numpy as np

def arrange(data, window, labels):
    n = len(data) - window
    if labels:
        return np.reshape(data[window:], (n, 1))
    result = np.zeros((n, window, 1), dtype=np.float32)
    for i in range(n):
        result[i, :, 0] = data[i:(i + window)]
    return result

def normalize(data):
    mean, variance = np.mean(data), np.var(data)
    return (data - mean) / math.sqrt(variance)

def partition(data, window, validation, test):
    n = len(data)
    n -= n % window
    i = int(round(n * (1.0 - validation - test)))
    i -= i % window
    j = int(round(n * (1.0 - test)))
    j -= j % window
    return data[:i], data[i:j], data[j:n]

def prepare(data, window, validation=0.1, test=0.1):
    x = process(data, window, validation, test, False)
    y = process(data, window, validation, test, True)
    return x, y

def process(data, window, validation, test, labels):
    train, validation, test = partition(data, window, validation, test)
    train = arrange(train, window, labels)
    validation = arrange(validation, window, labels)
    test = arrange(test, window, labels)
    return dict(train=train, validation=validation, test=test)

def read(path='tests/fixtures/google.sqlite3'):
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    cursor.execute('SELECT time FROM arrivals ORDER BY time')
    data = np.diff(np.array([row[0] for row in cursor]))
    connection.close()
    return data
