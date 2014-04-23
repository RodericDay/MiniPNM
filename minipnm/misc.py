from __future__ import absolute_import, division, print_function
import numpy as np
from scipy import misc

import minipnm as mini

def imread(path_or_ndarray, dmax=200):
    try:
        im = misc.imread(path_or_ndarray)
    except AttributeError:
        im = path_or_ndarray
    rf = np.true_divide(dmax, im.shape).clip(0,1).min()
    im = misc.imresize(im, rf)
    im = im.astype(float)
    im-= im.min()
    im/= im.max()
    im = im.T[0]
    im = im.reshape(im.shape+(1,))
    return im