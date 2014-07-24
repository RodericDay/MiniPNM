from tempfile import NamedTemporaryFile
from subprocess import call
import gzip

import numpy as np
from scipy import misc, ndimage

'''
Functions related to generation and handling of 2D images, and stacks thereof
'''

def gaussian_noise(shape, exp=1, mode='wrap'):
    '''
    mode : reflect, constant, nearest, mirror, wrap
    '''
    R = np.random.random(shape)
    N = np.zeros_like(R)
    for sigma in 2**np.arange(6):
        N += ndimage.filters.gaussian_filter(R, sigma, mode=mode) * sigma**exp
    N = np.true_divide(N, np.abs(N).max())
    N = np.subtract(N, N.min())
    return N

def save_gif(image_stack, outfile='animated.gif'):
    slide_paths = []
    for slide in image_stack.T:
        f = NamedTemporaryFile(suffix='.png', delete=False)
        misc.imsave(f, slide)
        slide_paths.append( f.name )
    call(["convert"] + slide_paths + [outfile])

def view_stack(image_stack):
    '''
    quick matplotlib image stack explorer
    '''
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    slides = [slide for slide in image_stack.T]
    fig = plt.figure()
    im_axes = fig.add_axes([0.1,0.2,0.8,0.7])
    im_obj = im_axes.imshow(slides[0], interpolation='nearest', cmap='binary')
    slider_axes = fig.add_axes([0.2, 0.1, 0.6, 0.05])
    slider = Slider(slider_axes, 'Z', 0, len(slides), 0, closedmax=False)
    conn = slider.on_changed(lambda value: im_obj.set_data(slides[int(value)]))
    plt.show()

def read_ubc(path):
    string = gzip.open(path).read()
    im = np.fromstring(string, dtype='int8').reshape(512,512,512)
    im = im[:-1,:-1,:-1]
    return im

def read(path_or_ndarray, zoom=0.1):
    try:
        im = misc.imread(path_or_ndarray)
    except AttributeError:
        im = path_or_ndarray
    # collapse redundant rgba channels (ie: is grayscale)
    if not np.any(im - np.dstack([im[:,:,0]]*im.shape[-1])):
        im = im[:,:,0]
    im = ndimage.zoom(im, zoom, order=1)
    im = im.transpose()
    return im.squeeze()

def extract_spheres(im):
    '''
    credit to untubu @ stackoverflow for this
    still needs a lot of improvement
    '''
    im = np.atleast_3d(im)
    data = ndimage.morphology.distance_transform_edt(im)
    max_data = ndimage.filters.maximum_filter(data, 10)
    maxima = data==max_data # but this includes some globally low voids
    min_data = ndimage.filters.minimum_filter(data, 10)
    diff = (max_data - min_data) > 1
    maxima[diff==0] = 0

    labels, num_maxima = ndimage.label(maxima)
    centers = [ndimage.center_of_mass(labels==i) for i in range(1, num_maxima+1)]
    radii = [data[center] for center in centers]
    return np.array(centers), np.array(radii)