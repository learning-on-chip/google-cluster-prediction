import numpy as np
import sqlite3

def count_apps(path):
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    cursor.execute('SELECT COUNT(DISTINCT app) FROM job_events')
    data = cursor.fetchone()[0]
    connection.close()
    return data

def count_users(path):
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    cursor.execute('SELECT COUNT(DISTINCT user) FROM job_events')
    data = cursor.fetchone()[0]
    connection.close()
    return data

def count_user_jobs(path):
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    cursor.execute('SELECT user, COUNT(time) FROM job_events GROUP BY user')
    data = np.array([row for row in cursor])
    connection.close()
    return data

def select_jobs(path, app=None, user=None):
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    sql = 'SELECT time, app, user FROM job_events'
    if app is not None or user is not None: sql += ' WHERE'
    if app is not None:
        app = app if hasattr(app, '__iter__') else [app]
        sql += ' app in ({})'.format(', '.join([str(app) for app in app]))
    if app is not None and user is not None: sql += ' AND'
    if user is not None:
        user = user if hasattr(user, '__iter__') else [user]
        sql += ' user in ({})'.format(', '.join([str(user) for user in user]))
    sql += ' ORDER BY time'
    cursor.execute(sql)
    data = np.array([row for row in cursor])
    connection.close()
    return data
