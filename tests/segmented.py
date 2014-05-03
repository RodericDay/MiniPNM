import gzip
import numpy as np
from scipy.ndimage import zoom
import minipnm as mini

path = '/Users/roderic/Developer/Thesis/MiniPNM/tests/segmented_bead_pack_512.ubc.gz'
string = gzip.open(path).read()
im = np.fromstring(string, dtype='int8').reshape(512,512,512)
im = im[:-1,:-1,:-1]
im = zoom(im, 0.4, order=1)[:,:,:5]
print im.shape

network = mini.Cubic(im, im.shape)
balls = network - (im==0)
network = network - (im==1)
x,y,z = network.coords
ics = 3*(x==x.min()) + 1*(x==x.max())
network['sol'] = mini.linear_solve(network, ics)
network.render(network['sol'])