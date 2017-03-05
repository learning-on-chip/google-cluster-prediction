import numpy as np
import sqlite3

def count_job_task_samples(path):
    query = """
        SELECT `job ID`, `task index`, COUNT(*)
        FROM `task_usage`
        GROUP BY `job ID`, `task index`
        ORDER BY `job ID`, `task index`
    """
    return _execute(path, query, dtype=np.int)

def select_task_usage(path, job, task):
    query = """
        SELECT `CPU rate`
        FROM `task_usage`
        WHERE `job ID` = {} AND `task index` = {}
        ORDER BY `start time`
    """
    return _execute(path, query.format(job, task), dtype=np.float32)

def _execute(path, query, **arguments):
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    cursor.execute(query)
    data = np.array([row for row in cursor], **arguments)
    connection.close()
    return data
