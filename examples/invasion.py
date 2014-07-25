import itertools as it
import numpy as np
import minipnm as mini

network = mini.Bridson([40,30,5])

x,y,z = network.coords
left = x < np.percentile(x, 10)
right = x > np.percentile(x, 90)

history = [np.zeros(network.order, dtype=bool)]
pden, oden = [], []
while not all(history[-1]==True):
    state = history[-1].copy()

    cut = network.copy()
    cut.prune(state)
    pden.append(mini.algorithms.bvp.solve(cut.laplacian, {20 : left, 0 : right}))
    oden.append(mini.algorithms.bvp.solve(cut.laplacian, {0 : left, 10 : right}))

    newly_flooded = np.random.choice((state==False).nonzero()[0])
    state[newly_flooded] = True
    history.append(state)
else:
    pden.append(pden[-1])
    oden.append(oden[-1])

x = np.arange(len(history))
y = (network['sphere_radii'] * history).sum(axis=1)

gui = mini.GUI()
gui.plotXY(x,y)
gui.scene.add_spheres(network.points, network['sphere_radii'], alpha=0.3)
gui.scene.add_spheres(network.points, network['sphere_radii'] * history, color=(0,0,1))
gui.scene.add_tubes(network.midpoints, network.spans, network['cylinder_radii'])
gui.scene.add_surface(network.points, network.pairs, pden, cmap='bone_r', offset=10)
gui.scene.add_surface(network.points, network.pairs, oden, cmap='gist_heat_r', offset=10)
# gui.scene.add_spheres(network.points, network['sphere_radii'] * history, color=(0,0,1))
gui.run()