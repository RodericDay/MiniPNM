import gzip
import numpy as np
import matplotlib.pyplot as plt

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

def ubcread(path):
    string = gzip.open(path).read()
    im = np.fromstring(string, dtype='int8').reshape(512,512,512)
    im = im[:-1,:-1,:-1]
    return im

def view_stack(path):
    '''
    Simple shortcut to explore the .ubc files provided by Sheppard
    '''
    from matplotlib.widgets import Slider
    plt.rcParams['image.cmap'] = 'binary'
    string = gzip.open(path).read()
    im = np.fromstring(string, dtype='int8').reshape(512,512,512)

    fig, ax = plt.subplots(1)
    ax.imshow(im[0])
    fig.subplots_adjust(bottom=0.2)
    sax = fig.add_axes([0.2, 0.1, 0.6, 0.05])
    slider = Slider(sax, 'Z', 0, len(im), 0, closedmax=False)
    conn = slider.on_changed(lambda value: ax.imshow(im[int(value)]))
    plt.show()
