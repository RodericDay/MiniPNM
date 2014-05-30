from __future__ import division
import numpy as np
import minipnm as mini

ubc = mini.utils.ubcread('segmented_bead_pack_512.ubc.gz')
im = mini.imread(ubc, zoom=0.1)
im = im[len(im)//2]

centers, radii = mini.extract_spheres(im==1)
network = mini.PackedSpheres(centers, radii, tpt=2)

x,y,z = network.coords
sources = x==x.min()
capacities = radii**3 * (4/3) * np.pi
print "Max capacity: {:>10.2f} m^3".format(sum(capacities))

volumes = mini.volumetric(network, sources, capacities, f=1)

radii_history = ( np.vstack(volumes) * (3/4) / np.pi )**(1/3)
# vis
scene = mini.Scene()
scene.add_spheres(network.points, network['radii'], color=(1,1,1), alpha=0.1)
scene.add_spheres(network.points, radii_history, color=(0.2,0.3,1))
scene.add_tubes(network.points, network.pairs, alpha=0.3)
scene.play()