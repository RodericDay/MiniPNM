import numpy as np
import minipnm as mini

network = mini.Bridson()
radii = network['sphere_radii']

x,y,z = network.coords
sources = sinks = x==x.min()

cmat = network.adjacency_matrix
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
scene.add_spheres(network.points, radii, alpha=0.3, color=(1,1,1))
scene.add_spheres(network.points, radii*history, color=(0,0,1))
scene.add_tubes(network.midpoints, network.spans, network['cylinder_radii'])
scene.play(100)