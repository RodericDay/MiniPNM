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
im = im[:,:,100:101]

centers, radii = mini.extract_spheres(im)
pores = mini.PackedSpheres(centers, radii, tpt=5)
source = pores['y'] == pores['y'].max()
threshold = 1/pores['radii']
conditions = np.linspace(threshold.min(), threshold.max(), 100)
# saturations = mini.percolation(pores, source, threshold, conditions)
saturations = mini.invasion(pores, source, threshold)

network = mini.Cubic(im, im.shape) - (im==1)
network = mini.radial_join(network, pores)
elecs = []
for submask in saturations:
    mask = np.zeros(network.order, dtype=bool)
    mask[-len(submask):] = ~submask.astype(bool)
    temp = network - mask
    x,y,z = temp.coords
    dbcs = 3*(x==x.min()) + 1*(x==x.max())
    sol = mini.linear_solve(temp, dbcs)

    expanded = np.zeros_like(mask, dtype=float)
    expanded[~mask] = sol
    elecs.append(expanded)
sol = np.vstack(elecs)

scene = mini.Scene()
scene.add_wires(network.points, network.pairs, sol, alpha=1, cmap='jet')
scene.add_tubes(pores.points, pores.pairs, saturations, alpha=0.7, cmap='Blues')
scene.add_spheres(pores.points, pores['radii'], alpha=0.1, color=(1,1,1))
scene.add_spheres(pores.points, pores['radii']*saturations, color=(0.2,0.3,0.8))
# scene.save(frames=len(saturations))
scene.play(rate=100)