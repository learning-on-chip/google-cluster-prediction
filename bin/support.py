import matplotlib.pyplot as pp
import sqlite3

def figure(width=12, height=8):
    pp.figure(figsize=(width, height), dpi=80, facecolor='w', edgecolor='k')

def read(path='tests/fixtures/google.sqlite3'):
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    cursor.execute('SELECT time FROM arrivals ORDER BY time')
    data = np.diff(np.array([row[0] for row in cursor]))
    connection.close()
    return data
