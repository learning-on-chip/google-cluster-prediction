import glob, sqlite3
import numpy as np

import support

class Database:
    def __init__(self, path):
        self.connection = sqlite3.connect(path)

    def count_job_task_samples(self):
        query = """
            SELECT `job ID`, `task index`, COUNT(*)
            FROM `task_usage`
            GROUP BY `job ID`, `task index`
        """
        cursor = self.connection.cursor()
        cursor.execute(query)
        return np.array([row for row in cursor], dtype=np.int)

class DistributedDatabase:
    def __init__(self, path):
        paths = glob.glob('{}/**/*.sqlite3'.format(path))
        support.log(self, 'Databases: {}', len(paths))
        self.paths = paths

    def count(self):
        return len(self.paths)
