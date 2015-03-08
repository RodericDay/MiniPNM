import numpy as np
import minipnm as mini

def stuff_that_needs_done():
    # for water
    topology = mini.Cubic([40, 30, 1])
    x,y,z = topology.coords
    membrane = x==x.min()
    gdl = x==x.max()
    geometry = mini.Radial(
        centers=topology.points,
        radii=np.random.uniform(0.1, 0.5, topology.order),
        pairs=topology.pairs,
        prune=False)
    cmat = geometry.adjacency_matrix
    cmat.data = geometry.cylinders.radii
    water = mini.simulations.Invasion(
        cmat=cmat,
        capacities=geometry.spheres.volumes)

    # other transports
    from two_systems import Model
    model = Model(topology.asarray().shape, geometry.cylinders.radii)
    heat_transport = mini.bvp.System(model.cubic.pairs, conductances=1E-10)
    hbcs = { 350 : membrane, 340 : gdl }

    # simulation
    for_left = []
    for t in range(15):
        model.resolve(1.2)

        # get heat
        # heat = heat_transport.solve(hbcs, s=model.local_current)
        heat = model.local_current

        # water keeps track automatically, keep track of heat for vis
        for_left.append(heat)

        # generation depends on current activity (top 10)
        most_activity = np.in1d(topology.indexes,
            model.local_current.argsort()[-10:])
        G = 5
        try:
            water.distribute( G*most_activity*geometry.spheres.volumes.mean() )
        except water.NeighborsSaturated:
            pass

        # continue
        # block water where cold and full
        full = water.saturation > 0.9
        least_activity = np.in1d(topology.indexes,
            heat.argsort()[:100])
        freeze = full & least_activity
        # if t > 13: freeze *= False
        water.block( freeze )
        model.block( water.blocked )

    # leftover drill
    model.resolve(1.2)
    for_left.append(model.local_current)

    # this marks frozen spots as nan for render
    water_array = np.where(
        water.expand_block_states(),
        np.nan,
        water.history
    )


from minipnm.gui import floodview
W = np.dstack([topology.asarray(vals) for vals in water_array]).T
S = np.dstack([topology.asarray(vals) for vals in for_left]).T
G = topology.asarray(1/geometry.spheres.volumes).T
floodview(W, S, G)
