# encoding: utf-8
import gzip
import numpy as np
from scipy import ndimage
import minipnm as mini

path = 'segmented_bead_pack_512.ubc.gz'
string = gzip.open(path).read()
im = np.fromstring(string, dtype='int8').reshape(512,512,512)
im = im[:-1,:-1,:-1]
im = ndimage.zoom(im, 0.2, order=1)
im = im[:30,:30,:30]

network = mini.Cubic(im, im.shape)
network = network - (im==1)
x,y,z = network.coords
dbcs = 3*(x==x.min()) + 1*(x==x.max())
sol = mini.linear_solve(network, dbcs)

centers, radii = mini.extract_spheres(im)
pores = mini.Delaunay(centers)
pores['radii'] = np.atleast_1d(radii)
source = pores['y'] == pores['y'].min()
threshold = pores['radii']
conditions = np.linspace(threshold.min(), threshold.max()*3, 100)
saturations = mini.percolation(pores, source, threshold, conditions)

scene = mini.Scene()
scene.add_wires(network.points, network.pairs, sol, alpha=0.7)
scene.add_wires(pores.points, pores.pairs, saturations)
scene.add_spheres(centers, radii, alpha=0.3)
scene.add_spheres(centers, saturations*radii, color=(0.2,0.3,0.8))
scene.play()