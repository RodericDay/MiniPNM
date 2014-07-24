import numpy as np
from scipy import misc, ndimage
from tempfile import NamedTemporaryFile
from subprocess import call

'''
Functions related to generation and handling of 2D images, and stacks thereof
'''

def imread(path_or_ndarray, zoom=0.1):
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

def read_ubc(path):
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

def gaussian_noise(shape, exp=0.5, mode='wrap'):
    '''
    mode : reflect, constant, nearest, mirror, wrap
    '''
    R = np.random.random(shape)
    N = np.zeros_like(R)
    for i in 2**np.arange(6):
        N += ndimage.filters.gaussian_filter(R, i, mode=mode) * i**exp
    N = np.true_divide(N, np.abs(N).max())
    N = np.subtract(N, N.min())
    return N

def vtk_stack():
    self.renWin.SetSize(*size)
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(self.renWin)
    writer = vtk.vtkPNGWriter()
    for i in range(frames):
        self.timeout()
        w2if.Modified()
        writer.SetFileName("tmp/{:0>3}.png".format(i))
        writer.SetInput(w2if.GetOutput())
        writer.Write()

def save_gif(image_stack, outfile='animated.gif'):
    slide_paths = []
    for slide in image_stack.T:
        f = NamedTemporaryFile(suffix='.png', delete=False)
        misc.imsave(f, slide)
        slide_paths.append( f.name )
    call(["convert"] + slide_paths + [outfile])