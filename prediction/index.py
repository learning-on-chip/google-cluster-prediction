from . import database
from . import support
import os

class Index:
    def decode(input_path, callback):
        support.log(Index, 'Input path: {}', input_path)
        count = 0
        with open(input_path, 'r') as file:
            for record in file:
                count += 1
                record = record.split(',')
                callback(
                    path=record[0],           # Path
                    user=int(record[1]),      # User
                    app=int(record[2]),       # App
                    job=int(record[3]),       # Job
                    task=int(record[4]),      # Task
                    length=int(record[5]),    # Length
                    maximum=float(record[6]), # Maximum
                )
        return count

    def encode(task_usage_path, job_events_path, index_path, report_each=10000):
        support.log(Index, 'Task usage path: {}', task_usage_path)
        support.log(Index, 'Job events path: {}', job_events_path)
        paths = support.scan(task_usage_path, '*.sqlite3')
        database_count = len(paths)
        job_events_meta_map = database.map_job_events_meta(job_events_path)
        progress = support.Progress(description='indexing',
                                    total_count=database_count,
                                    report_each=report_each)
        progress.start()
        with open(index_path, 'w') as file:
            for i in range(database_count):
                task_usage_meta = database.select_task_usage_meta(paths[i])
                for task_usage_meta in task_usage_meta:
                    job_events_meta = job_events_meta_map[task_usage_meta[0]]
                    record = [
                        paths[i],           # Path
                        job_events_meta[0], # User
                        job_events_meta[1], # App
                        task_usage_meta[0], # Job
                        task_usage_meta[1], # Task
                        task_usage_meta[2], # Length
                        task_usage_meta[3], # Maximum
                    ]
                    file.write(','.join([str(item) for item in record]) + '\n')
                progress.advance()
        progress.finish()
