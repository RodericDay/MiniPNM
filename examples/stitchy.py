import numpy as np
import minipnm as mini

N = 20
x = np.arange(-N,N)
y = np.arange(-N,N)
X,Y = np.meshgrid(x,y)
Z = np.sqrt(X**2 + Y**2)
im = Z < Z.mean()

network = mini.Cubic(im, im.shape)
network = network - im

centers, radii = mini.extract_spheres(im)
sphere = mini.PackedSpheres(centers, radii)

stitched = mini.binary.radial_join(network, sphere)

scene = mini.Scene()
scene.add_wires(stitched.points, stitched.pairs)
scene.add_spheres(sphere.points, sphere['radii'], alpha=0.5)
scene.play()