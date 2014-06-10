from __future__ import division
import numpy as np
import minipnm as mini

N = 3
im = np.ones([N,N,N], dtype=bool)
for i in [i for i, c in np.ndenumerate(im) if np.linalg.norm(np.subtract(i, N/2-0.5))>N/2.5]:
    im[i] = 0

network = mini.Cubic(im, im.shape)
print network.points[0]
print network.dims
network = network - im

centers, radii = mini.extract_spheres(im)
print centers

sphere = mini.PackedSpheres(centers, radii)

stitched = mini.binary.radial_join(network, sphere)

scene = mini.Scene()
scene.add_wires(stitched.points, stitched.pairs)
scene.add_spheres(sphere.points, sphere['radii'], alpha=0.5)
scene.play()