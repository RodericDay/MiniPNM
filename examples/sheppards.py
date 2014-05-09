import gzip
import numpy as np
from scipy import ndimage
import minipnm as mini

path = 'segmented_bead_pack_512.ubc.gz'
string = gzip.open(path).read()
im = np.fromstring(string, dtype='int8').reshape(512,512,512)
im = im[:-1,:-1,:-1]
im = ndimage.zoom(im, 0.2, order=1)
im = im[:,:,5]
im = ndimage.morphology.distance_transform_bf(im)

print im
network = mini.Cubic(im, im.shape)
void = network - (im==0)
structure = network - (im==1)
void.render(void['intensity'], alpha=1)