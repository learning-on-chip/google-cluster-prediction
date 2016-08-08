import matplotlib.pyplot as pp
import numpy as np
import sqlite3

DATABASE_PATH = 'tests/fixtures/google.sqlite3'

def count_apps(path=DATABASE_PATH):
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    cursor.execute('SELECT COUNT(DISTINCT app) FROM jobs')
    data = cursor.fetchone()[0]
    connection.close()
    return data

def count_users(path=DATABASE_PATH):
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    cursor.execute('SELECT COUNT(DISTINCT user) FROM jobs')
    data = cursor.fetchone()[0]
    connection.close()
    return data

def count_user_jobs(path=DATABASE_PATH):
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    cursor.execute('SELECT COUNT(time) FROM jobs GROUP BY user')
    data = np.array([row[0] for row in cursor])
    connection.close()
    return data

def figure(width=14, height=6):
    pp.figure(figsize=(width, height), dpi=80, facecolor='w', edgecolor='k')

def normalize(data):
    mean = np.mean(data)
    variance = np.var(data)
    return (data - mean) / np.sqrt(variance)

def select_interarrivals(app=None, user=None, path=DATABASE_PATH):
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    sql = 'SELECT time FROM jobs'
    if app is not None or user is not None: sql += ' WHERE'
    if app is not None: sql += ' app = {}'.format(app)
    if app is not None and user is not None: sql += ' AND'
    if user is not None: sql += ' user = {}'.format(user)
    sql += ' ORDER BY time'
    cursor.execute(sql)
    data = np.diff(np.array([row[0] for row in cursor]))
    connection.close()
    return data
