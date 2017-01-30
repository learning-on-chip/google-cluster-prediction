import numpy as np
import sqlite3

def count_job_task_samples(path):
    query = """
        SELECT `job ID`, `task index`, COUNT(*)
        FROM `task_usage`
        GROUP BY `job ID`, `task index`
        ORDER BY `job ID`, `task index`
    """
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    cursor.execute(query)
    data = np.array([row for row in cursor], dtype=np.int)
    connection.close()
    return data
