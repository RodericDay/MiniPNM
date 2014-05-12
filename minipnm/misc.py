from __future__ import absolute_import, division
import numpy as np
from scipy import misc, ndimage

'''
misc houses scientific tools and helpers
'''

def normalize(array):
    array = np.atleast_1d(array)
    array = np.subtract(array, array.min())
    array = np.true_divide(array, array.max())
    return array

def imread(path_or_ndarray, dmax=200):
    try:
        im = misc.imread(path_or_ndarray)
    except AttributeError:
        im = path_or_ndarray
    if dmax:
        rf = np.true_divide(dmax, im.shape).clip(0,1).min()
        im = misc.imresize(im, rf)
    im = normalize(im)
    im = im.T[0]
    im = im.reshape(im.shape+(1,))
    return im

def gaussian_noise(dims):
    R = np.random.random(dims)
    N = np.zeros_like(R)
    for i in 2**np.arange(6):
        N += ndimage.filters.gaussian_filter(R, i) * i**0.5
    N = normalize(N)
    return N

def extract_spheres(im):
    '''
    credit to untubu @ stackoverflow for this
    still needs a lot of improvement
    '''
    data = ndimage.morphology.distance_transform_edt(im)
    max_data = ndimage.filters.maximum_filter(data, 10)
    maxima = data==max_data # but this includes some globally low voids
    min_data = ndimage.filters.minimum_filter(data, 10)
    diff = (max_data - min_data) > 1
    maxima[diff==0] = 0

    labels, num_maxima = ndimage.label(maxima)
    centers = [ndimage.center_of_mass(labels==i) for i in range(1, num_maxima+1)]
    radii = [data[center] for center in centers]
    return centers, radii