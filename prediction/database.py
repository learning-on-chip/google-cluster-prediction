import numpy as np
import sqlite3

def count_job_task_samples(path):
    query = """
        SELECT `job ID`, `task index`, COUNT(*)
        FROM `task_usage`
        GROUP BY `job ID`, `task index`
        ORDER BY `job ID`, `task index`
    """
    def _process(cursor):
        return cursor.fetchall()
    return _execute(path, query, _process)

def map_job_to_user_app(path):
    query = """
        SELECT `job ID`, `user`, `logical job name`
        FROM `job_events`
        WHERE `event type` = 0
        ORDER BY `job ID`
    """
    def _process(cursor):
        jobs, users, apps = {}, {}, {}
        for row in cursor:
            user = users.get(row[1], len(users))
            app = apps.get(row[2], len(apps))
            jobs[row[0]] = (user, app)
            users[row[1]] = user
            apps[row[2]] = app
        return jobs
    return _execute(path, query, _process)

def select_task_usage(path, job, task):
    query = """
        SELECT `CPU rate`
        FROM `task_usage`
        WHERE `job ID` = {} AND `task index` = {}
        ORDER BY `start time`
    """
    def _process(cursor):
        return np.reshape([row[0] for row in cursor], [-1, 1])
    return _execute(path, query.format(job, task), _process)

def _execute(path, query, process):
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    cursor.execute(query)
    data = process(cursor)
    connection.close()
    return data
