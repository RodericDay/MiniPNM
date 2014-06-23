import numpy as np
import minipnm as mini

points = np.random.rand(1000,3)
points.T[2] *= 0
network = mini.Delaunay(points)
network = network - network.boundary()
x,y,z = network.coords
candidates = (0.4 < x) & (x < 0.6)

sources = np.zeros_like(x, dtype=bool)
sources[np.random.choice(candidates.nonzero()[0], 5)] = 1

sinks = (x < np.percentile(x,5)) #| (x > np.percentile(x,95))

radii = mini.distances_to_nearest_neighbors(network)/2

history = mini.invasion(network, sources, 1/radii, sinks=sinks)
history = np.vstack([history, np.atleast_2d(history[-1]).repeat(5, axis=0)])

sol = np.zeros_like(history, dtype=float)
for i, state in enumerate(history):
    copy = network.prune(state & ~sources, remove_pores=False)
    dbcs = 3*(x < np.percentile(x,5)) + 1*sources
    sol[i] = mini.linear_solve(copy, dbcs)

scene = mini.Scene()
scene.add_spheres((network-~sources).points, radii[sources]*1.1)
scene.add_spheres(network.points, radii, alpha=0.1, color=(1,1,1))
scene.add_wires(network.points, network.pairs, sol, cmap='copper')
scene.add_spheres(network.points, radii*history, color=(0,0,1))
scene.play()