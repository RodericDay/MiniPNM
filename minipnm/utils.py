import gzip
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage

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


