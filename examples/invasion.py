import numpy as np
import minipnm as mini

pdf = (np.random.weibull(3)+0.5 for _ in iter(int,1))
network = mini.Bridson([30,30,5], pdf)

x,y,z = network.coords
source = x==x.min()
history = mini.algorithms.invasion(network, source, 1./network['sphere_radii'])

spheres = mini.Spheres(network.points, network['sphere_radii'] * history, color=(0,0,1))
wires = mini.Tubes(network.midpoints, network.spans, network['cylinder_radii'])

f = (network['sphere_radii'] * history)
x = f.sum(axis=1)
y = x / (f!=0).sum(axis=1)
# y = (network['sphere_radii'] * history).sum(axis=1) / network['sphere_radii'].sum()

gui = mini.GUI()
gui.addActor(wires)
gui.addActor(spheres)
gui.plotXY(x,y)
gui.run()