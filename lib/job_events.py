import numpy as np
import sqlite3

def count_apps(path):
    query = """
        SELECT COUNT(DISTINCT app)
        FROM job_events
    """
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    cursor.execute(query)
    data = cursor.fetchone()[0]
    connection.close()
    return data

def count_users(path):
    query = """
        SELECT COUNT(DISTINCT user)
        FROM job_events
    """
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    cursor.execute(query)
    data = cursor.fetchone()[0]
    connection.close()
    return data

def count_user_jobs(path):
    query = """
        SELECT user, COUNT(time)
        FROM job_events
        GROUP BY user
    """
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    cursor.execute(query)
    data = np.array([row for row in cursor])
    connection.close()
    return data

def select(path, app=None, user=None):
    query = """
        SELECT time, app, user
        FROM job_events
    """
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    if app is not None or user is not None: query += ' WHERE'
    if app is not None:
        app = app if hasattr(app, '__iter__') else [app]
        apps = ', '.join([str(app) for app in app])
        query += ' app in ({})'.format(apps)
    if app is not None and user is not None: query += ' AND'
    if user is not None:
        user = user if hasattr(user, '__iter__') else [user]
        users = ', '.join([str(user) for user in user])
        query += ' user in ({})'.format(users)
    query += ' ORDER BY time'
    cursor.execute(query)
    data = np.array([row for row in cursor])
    connection.close()
    return data
