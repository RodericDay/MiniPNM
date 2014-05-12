# encoding: utf-8
import gzip
import numpy as np
from scipy import ndimage
import minipnm as mini

path = 'segmented_bead_pack_512.ubc.gz'
string = gzip.open(path).read()
im = np.fromstring(string, dtype='int8').reshape(512,512,512)
im = im[:-1,:-1,:-1]
im = ndimage.zoom(im, 0.1, order=1)

network = mini.Cubic(im, im.shape)
network = network - (im==1)

print "#voxels representing spheres =", np.sum(im)
centers, radii = mini.extract_spheres(im)

pores = mini.Delaunay(centers)

scene = mini.Scene()
scene.add_wires(network.points, network.pairs)
scene.add_wires(pores.points, pores.pairs)
scene.add_spheres(centers, radii)
scene.play()