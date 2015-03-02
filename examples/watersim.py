import numpy as np
import minipnm as mini


# for water
topology = mini.Cubic([40, 30, 1])
x,y,z = topology.coords
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
small = geometry.spheres.radii < 0.25

# other transports
from two_systems import Model
model = Model(topology.asarray().shape, geometry.cylinders.radii)

# simulation
# water.block( ~source & small )
for_left = []
for t in range(10):
    model.resolve(1.2)
    current = model.local_current
    for_left.append(current)

    # flood the 10 most active pores?
    source = np.in1d(topology.indexes, np.argsort(current)[-10:])
    try:
        water.distribute( source*geometry.spheres.volumes*10 )
    except water.NeighborsSaturated:
        pass
    model.block( water.saturation==1 )
model.resolve(1.2)
for_left.append(model.local_current)


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

fig, (ax, ax2) = plt.subplots(1, 2)

radii2d = topology.asarray(1/geometry.spheres.radii)
background = ax.imshow(radii2d,
    interpolation='nearest',
    cmap='copper')

def update(i):
    i = int(min(i, len(water.history)-1))

    arr = topology.asarray(water.history[i])
    water_layer.set_data(arr)

    arr2 = topology.asarray(for_left[i])
    ax2.imshow(arr2, interpolation='nearest', vmin=arr2[arr2!=0].min())

    fig.canvas.draw()

water_cmap = matplotlib.cm.get_cmap('Blues')
water_cmap.set_under(alpha=0)
# water_cmap.set_over('w')
water_layer = ax.imshow(np.zeros_like(radii2d),
    vmin=0.5, vmax=0.9,
    cmap=water_cmap)

slider = Slider(
    ax=fig.add_axes([0.2,0.01,0.6,0.03]),
    label='t',
    valmin=0,
    valmax=len(water.history))
slider.on_changed(update)
update(0)
plt.show()
