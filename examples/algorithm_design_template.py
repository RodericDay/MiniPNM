import numpy as np
import minipnm as mini

np.random.seed(43)
points = np.random.rand(20,3)
points.T[2] = 0
network = mini.Delaunay(points)
network = network - (mini.distances_to_furthest_neighbors(network) > 0.5)
network = network - (mini.distances_to_nearest_neighbors(network) < 0.05)
radii = mini.distances_to_nearest_neighbors(network)/2

x,y,z = network.coords
sources = sinks = x==x.min()

cmat = network.connectivity_matrix
cmat.data = (1/radii**2)[cmat.col]

pressures = np.linspace(0,100,10)

def flood(network, sources, pressures, cmat):
    saturations = np.zeros([pressures.size, network.order])
    for i, p in enumerate(pressures):
        saturations[i] = saturations[i-1] + 0.1
    return saturations

def drain(network, sinks, pressures, cmat):
    saturations = np.ones([pressures.size, network.order])
    for i, p in enumerate(pressures):
        saturations[i] = saturations[i-1] - 0.1
    return saturations

history = np.vstack([flood(network, sources, pressures, cmat), drain(network, sinks, pressures, cmat)])
print history.shape

scene = mini.Scene()
scene.add_spheres(network.points, radii, alpha=0.5, color=(1,1,1))
scene.add_spheres(network.points, radii*history, alpha=1, color=(0,0,1))
scene.add_wires(network.points, network.pairs)
scene.play(100)