from . import support
import os
import re
import tensorflow as tf


class Saver:
    def __init__(self, config, **arguments):
        self.backend = tf.train.Saver(max_to_keep=100, **arguments)
        self.auto = config.get('restore')
        self.path = config.path
        self.subscribers = []

    def restore(self, session):
        if self.auto is False:
            self._notify('restore', session)
            return
        paths = Saver._find(self.path)
        if len(paths) == 0:
            self._notify('restore', session)
            return
        steps = sorted(list(paths.keys()))
        if self.auto is None:
            options = ['Load ' + paths[key] for key in steps]
            i = support.prompt('Start anew', *options)
            if i == 0:
                self._notify('restore', session)
                return
            else:
                i -= 1
        elif self.auto is True:
            i = -1
        else:
            i = steps.index(self.auto)
        path = paths[steps[i]]
        self.backend.restore(session, path)
        support.log(self, 'Restore: {}', path)
        self._notify('restore', session)

    def save(self, session, step):
        self._notify('save', session)
        path = os.path.join(self.path, 'session-{}'.format(step))
        path = self.backend.save(session, path)
        support.log(self, 'Save: {}', path)

    def subscribe(self, subscriber):
        self.subscribers.append(subscriber)

    def _find(path):
        paths = {}
        for path in support.scan(path, 'session-*.meta'):
            step = int(re.search('.*session-(.*).meta', path).group(1))
            paths[step] = re.sub('.meta$', '', path)
        return paths

    def _notify(self, action, session):
        for subscriber in self.subscribers:
            getattr(subscriber, action)(session)
