from copy import deepcopy
import json
import os

RESERVED_KEYS = vars(dict).keys()


class Config(dict):
    def load(path):
        config = Config(json.loads(open(path).read()))
        for key in sorted(config.keys()):
            if key.startswith('__include'):
                another = os.path.join(os.path.dirname(path), config.pop(key))
                another = Config(json.loads(open(another).read()))
                another.update(config)
                config = another
        return config

    def __init__(self, options={}):
        super(Config, self).__init__()
        self.update(options)

    def copy(self):
        copy = Config()
        for key in self:
            value = self[key]
            if isinstance(value, Config):
                copy[key] = value.copy()
            else:
                copy[key] = deepcopy(value)
        return copy

    def update(self, options):
        copy = {}
        for key in options:
            value = options[key]
            if isinstance(value, dict) and not isinstance(value, Config):
                copy[key] = Config(value)
            else:
                copy[key] = value
        return super(Config, self).update(copy)

    def __getattr__(self, key):
        if key in RESERVED_KEYS:
            return getattr(self, key)
        else:
            return self[key]

    def __setattr__(self, key, value):
        assert(key not in RESERVED_KEYS)
        self.__setitem__(key, value)
