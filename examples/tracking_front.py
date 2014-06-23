import numpy as np
import minipnm as mini

points = np.random.rand(500,3)
points.T[2] = 0
network = mini.Delaunay(points)
network = network - (mini.distances_to_furthest_neighbors(network) > 0.2)
x,y,z = network.coords
sources = x==x.min()
sinks = x==x.max()
radii = mini.distances_to_nearest_neighbors(network)/2
thresholds = radii**-1
saturations = mini.invasion(network, sources, thresholds)

boundary = np.zeros_like(saturations)
for i, s in enumerate(saturations):
    unstable_throats = network.pairs[network.cut(s)]
    boundary[i][unstable_throats[~s[unstable_throats]]] = 1


scene = mini.Scene()
scene.add_spheres(network.points, radii*saturations, color=(0,0,1))
scene.add_spheres(network.points, radii*boundary, color=(1,0,0))
scene.add_wires(network.points, network.pairs)
scene.play(100)