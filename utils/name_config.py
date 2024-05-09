
class NameConfig:
    def __init__(self, **kwargs):
        self._dic = kwargs

    def __getattr__(self, item):
        try:
            return self._dic[item]
        except KeyError:
            return None

    def __setattr__(self, key, value):
        if key == '_dic':
            super(NameConfig, self).__setattr__(key, value)
            return
        self._dic[key] = value

    def __getitem__(self, item):
        try:
            return self._dic[item]
        except KeyError:
            return None

    def __setitem__(self, key, value):
        self._dic[key] = value

    def __delitem__(self, key):
        del self._dic[key]