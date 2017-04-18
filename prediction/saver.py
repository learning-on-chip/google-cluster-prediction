from . import support
import glob
import os
import re
import tensorflow as tf


class Saver:
    def __init__(self, config, **arguments):
        self.backend = tf.train.Saver(max_to_keep=100, **arguments)
        self.restore = config.get('restore')
        self.path = config.path

    def load(self, session):
        if self.restore is False:
            return
        paths = Saver._load(self.path)
        if len(paths) == 0:
            return
        step_counts = sorted(list(paths.keys()))
        if self.restore is None:
            options = ['Load ' + paths[key] for key in step_counts]
            i = support.prompt('Start anew', *options)
            if i == 0:
                return
            else:
                i -= 1
        elif self.restore is True:
            i = -1
        else:
            i = step_counts.index(self.restore)
        path = paths[step_counts[i]]
        self.backend.restore(session, path)
        support.log(self, 'Restore: {}', path)

    def save(self, session, state):
        path = os.path.join(self.path, 'session-{}'.format(state.step))
        path = self.backend.save(session, path)
        support.log(self, 'Save: {}', path)

    def _load(path):
        paths = {}
        for path in glob.glob(os.path.join(path, 'session-*.meta')):
            step_count = int(re.search('.*session-(.*).meta', path).group(1))
            paths[step_count] = re.sub('.meta$', '', path)
        return paths
