import numpy as np
import minipnm as mini

network = mini.Bridson()
network.render()

x,y,z = network.coords
source = x==x.min()
history = mini.algorithms.invasion(network, source, 1./network['sphere_radii'])

spheres = mini.Spheres(network.points, network['sphere_radii'] * history, color=(0,0,1))
wires = mini.Wires(network.points, network.pairs)

x = np.arange(len(history))
y = (network['sphere_radii'] * history).sum(axis=1)

gui = mini.GUI()
gui.addActor(wires)
gui.addActor(spheres)
gui.plotXY(x,y)
gui.run()