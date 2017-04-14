from . import database
from . import support
import glob
import os

class Index:
    def decode(path, callback):
        count = 0
        with open(path, 'r') as file:
            for record in file:
                count += 1
                record = record.split(',')
                callback(
                    path=record[0],        # Path
                    user=int(record[1]),   # User
                    app=int(record[2]),    # App
                    job=int(record[3]),    # Job
                    task=int(record[4]),   # Task
                    length=int(record[5]), # Length
                )
        return count

    def encode(input_path, meta_path, index_path, **arguments):
        support.log(Index, 'Input path: {}', input_path)
        support.log(Index, 'Meta path: {}', meta_path)
        pattern = os.path.join(input_path, '**', '*.sqlite3')
        paths = sorted(glob.glob(pattern, recursive=True))
        database_count = len(paths)
        mapping = database.map_job_to_user_app(meta_path)
        progress = support.Progress(description='indexing',
                                    total_count=database_count, **arguments)
        with open(index_path, 'w') as file:
            for i in range(database_count):
                data = database.count_job_task_samples(paths[i])
                for record in data:
                    meta = mapping[record[0]]
                    record = [
                        paths[i],  # Path
                        meta[0],   # User
                        meta[1],   # App
                        record[0], # Job
                        record[1], # Task
                        record[2], # Length
                    ]
                    file.write(','.join([str(item) for item in record]) + '\n')
                progress.advance()
        progress.finish()
