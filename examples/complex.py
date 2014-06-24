import itertools as it
import random

import numpy as np

import minipnm as mini

pdf = (random.choice([0.5,1]) for i in it.count())
bridson = mini.Bridson(pdf, [40,40,1])
bridson['z'] *= 0
bridson.pairs = mini.Delaunay.edges_from_points(bridson.points)

# air diff stuff
x,y,z = bridson.coords
dbcs = 10 * (x > np.percentile(x, 95)) + 8 * (x < np.percentile(x,5))

air_history = [mini.linear_solve(bridson, dbcs)]
water_history = [np.zeros_like(x, dtype=bool)]
ice_history = [np.zeros_like(x, dtype=bool)]
for i in range(50):
    # resolve the state of air diffusion in the network, limited by water blobs
    air_state = mini.linear_solve(bridson.prune(water_history[-1], remove_pores=False), dbcs )
    air_history.append( air_state )

    # do the water thing. anywhere not invaded by ice is fair game to get some water
    floodable = bridson.prune(ice_history[-1], remove_pores=False)
    # but who can generate water? only pores with access to air
    sourceable = bridson.prune(air_state==0, remove_pores=False)
    # which pores will actually generate water though? a subset
    sources = np.in1d(sourceable.indexes, random.sample(sourceable.indexes, 1))
    # we can be wasteful and simulate the entire thing, and keep the water differential
    history = mini.invasion(floodable, sources, 1/bridson['radii'])
    # we only want to add the minimum amount possible beyond a threshold at every step...
    # but let's say we add the first 10, whether new or not
    water_history.append( history[10] | water_history[-1] )

    # water_history.append( water_history[-1] | mini.invasion(floodable, sources, 1/bridson['radii'])[10] )
    # ice_history.append( ice_history[-1] | np.in1d(bridson.indexes, random.sample( water_history[-1].nonzero()[0], 1 ) ) )

scene = mini.Scene()
scene.add_wires(bridson.points, bridson.pairs, air_history, cmap='copper')
scene.add_spheres(bridson.points, bridson['radii'], color=(1,1,1), alpha=0.1)
scene.add_spheres(bridson.points, bridson['radii'] * water_history, color=(0,0,1))
scene.add_spheres(bridson.points, bridson['radii'] * ice_history * 1.1, alpha=0.5, color=(1,1,1))
scene.play(10)