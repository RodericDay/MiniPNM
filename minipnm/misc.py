from __future__ import absolute_import, division
import numpy as np
from scipy import misc, ndimage

import minipnm as mini

def imread(path_or_ndarray, dmax=200):
    try:
        im = misc.imread(path_or_ndarray)
    except AttributeError:
        im = path_or_ndarray
    if dmax:
        rf = np.true_divide(dmax, im.shape).clip(0,1).min()
        im = misc.imresize(im, rf)
    im = np.subtract(im, im.min())
    im = np.true_divide(im, im.max())
    im = im.T[0]
    im = im.reshape(im.shape+(1,))
    return im

def gaussian_noise(dims):
    R = np.random.random(dims)
    N = np.zeros_like(R)
    for i in 2**np.arange(6):
        N += ndimage.filters.gaussian_filter(R, i) * i**0.5
    # normalize
    N -= N.min()
    N /= N.max()
    return N
