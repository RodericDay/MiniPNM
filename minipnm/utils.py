import numpy as np

'''
utils houses software developer tools and helpers
'''

class FormulaDict(dict):
    '''
    A dictionary intended to hold either numpy arryas or formulae involving
    numpy arrays. Excel-like functionality.
    '''

    def __setitem__(self, key, value):
        try:
            assert hasattr(value, "__call__")
            super(FormulaDict, self).__setitem__(key, value)
        except AssertionError:
            array = np.array(value)
            # do some checks on array
            super(FormulaDict, self).__setitem__(key, array)

    def __getitem__(self, key):
        maybe_function = super(FormulaDict, self).__getitem__(key)
        if hasattr(maybe_function, "__call__"):
            arginfo = inspect.getargspec(maybe_function)
            return maybe_function(*(self[k] for k in arginfo.args))
        else:
            return maybe_function

    def __str__(self):
        return '\n'.join("{key:<10} : {value}".format(**locals()) \
                         for key, value in sorted(self.items()))


def property_from(list_of_keys, dtype=None, default=None):
    '''
    Shortcut for the process of creating thin properties from multiple arrays
    based on their key entries

    Gives the option of setting a dtype for casting
    '''
    def getter(self):
        return np.vstack([self.get(key, default) for key in list_of_keys]).T

    def setter(self, values):
        values = np.nan_to_num(np.atleast_2d(values))
        for key, array in zip(list_of_keys, values.T):
            self[key] = array.astype(dtype) if dtype else array

    def deleter(self):
        for key in list_of_keys:
            del self[key]

    return property(getter, setter)