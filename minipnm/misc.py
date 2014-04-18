from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.misc import imread, imresize

import minipnm as mini

def from_image(path_or_ndarray, dmax=200):
    try:
        im = imread(path_or_ndarray)
    except Exception as e:
        raise e
    rf = np.true_divide(dmax, im.shape).clip(0,1).min()
    im = imresize(im, [int(d*rf) for d in im.shape])
    im = im.astype(float)
    im-= im.min()
    im/= im.max()
    im = im.T[0]
    im = im.reshape(im.shape+(1,))

    cubic = mini.Cubic(im, im.shape)
    return cubic