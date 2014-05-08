import numpy as np
import minipnm as mini

network = mini.Cubic.empty([30,10,10])
# network = mini.Delaunay(network.points)
x,y,z = network.coords

source = x == x.min()
threshold = np.random.rand(network.size[0])
conditions = np.logspace(np.log10(threshold.min()), np.log10(threshold.max()*3), 100)

saturations = mini.percolation(network, source, threshold, conditions)

network.render(saturations, rate=10)