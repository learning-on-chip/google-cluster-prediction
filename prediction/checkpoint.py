from . import support
import glob
import os
import re
import tensorflow as tf


class Checkpoint:
    def __init__(self, output):
        self.saver = tf.train.Saver(max_to_keep=100)
        self.auto = output.get('auto_restore')
        self.path = output.path

    def load(self, session, state=None):
        paths = Checkpoint._load(self.path)
        if len(paths) == 0:
            return
        step_counts = sorted(list(paths.keys()))
        if self.auto is None:
            print('Choose one of the following options:')
            print('    0. Start anew')
            for i, step_count in enumerate(step_counts):
                print('    {}. Load {}'.format(i + 1, paths[step_count]))
            while True:
                try:
                    i = int(input('Your choice: '))
                except ValueError:
                    continue
                if i < 0 or i > len(step_counts):
                    continue
                elif i == 0:
                    return
                else:
                    i -= 1
                    break
        elif self.auto is True:
            i = -1
        else:
            return
        path = paths[step_counts[i]]
        self.saver.restore(session, path)
        support.log(self, 'Restore: {}', path)

    def save(self, session, state):
        path = os.path.join(self.path, 'model-{}'.format(state.step))
        path = self.saver.save(session, path)
        support.log(self, 'Save: {}', path)

    def _load(path):
        paths = {}
        for path in glob.glob(os.path.join(path, 'model-*.meta')):
            step_count = int(re.search('.*model-(.*).meta', path).group(1))
            paths[step_count] = re.sub('.meta$', '', path)
        return paths
