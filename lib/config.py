RESERVED_KEYS = vars(dict).keys()


class Config(dict):
    def __init__(self, options={}):
        super(Config, self).__init__()
        self.update(options)

    def copy(self):
        return self.__copy__()

    def update(self, options):
        copy = {}
        for key in options:
            value = options[key]
            if isinstance(value, dict) and not isinstance(value, Config):
                copy[key] = Config(value)
            else:
                copy[key] = value
        return super(Config, self).update(copy)

    def __copy__(self):
        return self.__class__(self)

    def __getattr__(self, key):
        if key not in RESERVED_KEYS:
            return self.get(key)
        else:
            return getattr(self, key)

    def __setattr__(self, key, value):
        assert(key not in RESERVED_KEYS)
        self.__setitem__(key, value)
