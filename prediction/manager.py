import sys


class Manager:
    def __init__(self, config):
        self.config = config

    def __getattr__(self, name):
        assert(name.startswith('should_'))
        name = name.replace('should_', '')
        period = self.config.get(name + '_period', sys.maxsize)
        return lambda state: state.step > 0 and state.step % period == 0
