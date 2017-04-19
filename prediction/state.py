from . import support
import numpy as np
import tensorflow as tf


class State:
    def __init__(self, report_each=10000):
        self.report_each = report_each
        state = np.zeros(1, dtype=np.int64)
        self.current = tf.Variable(state, name='current', dtype=tf.int64,
                                   trainable=False)
        self.new = tf.placeholder(tf.int64, shape=state.shape, name='new')
        self.assign_new = self.current.assign(self.new)
        self.step = None

    def advance(self, count=1):
        for _ in range(count):
            self.step += 1
            if self.step % self.report_each == 0:
                self._report()

    def load(self, session):
        state = session.run(self.current)
        self.step = state[0]
        self._report()

    def save(self, session):
        feed = {
            self.new: [self.step],
        }
        session.run(self.assign_new, feed)

    def _report(self):
        support.log(self, 'Current step: {}', self.step)
