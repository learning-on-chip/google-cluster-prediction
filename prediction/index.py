from . import database
from . import support
import glob
import os

class Index:
    def decode(path, callback):
        count = 0
        for record in open(path, 'r'):
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

    def encode(input_path, meta_path, index_path, report_each=10000):
        support.log(Index, 'Input path: {}', input_path)
        pattrn = os.path.join(input_path, '**', '*.sqlite3')
        paths = sorted(glob.glob(pattern, recursive=True))
        database_count = len(paths)
        support.log(Index, 'Databases: {}', database_count)
        support.log(Index, 'Meta path: {}', meta_path)
        mapping = database.map_job_to_user_app(meta_path)
        support.log(Index, 'Jobs: {}', len(mapping))
        trace_count = 0
        file = open(index_path, 'w')
        for i in range(database_count):
            data = database.count_job_task_samples(paths[i])
            trace_count += len(data)
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
            if (i + 1) % report_each == 0 or (i + 1) == database_count:
                support.log(Index, 'Processed: {} ({:.2f}%), traces: {}',
                            i + 1, 100 * (i + 1) / database_count, trace_count)
        file.close()
