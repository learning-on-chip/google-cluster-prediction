from . import database
from . import support
import glob

class Index:
    def decode(path, process):
        count = 0
        for record in open(path, 'r'):
            count += 1
            record = record.split(',')
            process(
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
        paths = glob.glob('{}/**/*.sqlite3'.format(input_path))
        database_count = len(paths)
        support.log(Index, 'Databases: {}', database_count)
        support.log(Index, 'Meta path: {}', meta_path)
        mapping = database.map_job_to_user_app(meta_path)
        support.log(Index, 'Jobs: {}', len(mapping))
        done_count, trace_count = 0, 0
        file = open(index_path, 'w')
        for path in sorted(paths):
            done_count += 1
            data = database.count_job_task_samples(path)
            trace_count += data.shape[0]
            for i in range(data.shape[0]):
                meta = mapping[data[i, 0]]
                record = [
                    path,       # Path
                    meta[0],    # User
                    meta[1],    # App
                    data[i, 0], # Job
                    data[i, 1], # Task
                    data[i, 2], # Length
                ]
                file.write(','.join([str(item) for item in record]) + '\n')
            if done_count % report_each == 0 or done_count == database_count:
                support.log(Index, 'Processed: {} ({:.2f}%), traces: {}',
                            done_count, 100 * done_count / database_count,
                            trace_count)
        file.close()
