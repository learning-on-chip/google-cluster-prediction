RESERVED_KEYS = vars(dict).keys()


class Config(dict):
    def __init__(self, *arguments, **key_arguments):
        super(Config, self).__init__()
        [self.update(argument) for argument in arguments]
        self.update(key_arguments)

    def __getattr__(self, key):
        if key not in RESERVED_KEYS:
            return self.get(key)
        else:
            return getattr(self, key)

    def __setattr__(self, key, value):
        assert(key not in RESERVED_KEYS)
        self.__setitem__(key, value)
