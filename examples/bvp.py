import numpy as np
from scipy import sparse
import minipnm as mini

network = mini.Cubic.empty([30,30,10])
# network = mini.Delaunay.random(100)
# o, i = network.split(network.boundary())
# network = o | i

x,y,z = network.coords
dirichlet = {
    0 : x == x.min(),
}

neumann = {
    5 : network.cut( (x == x.max()) & (y > np.percentile(y, 50)), network.indexes, bijective=True),
}
s = mini.solve_bvp(network.laplacian, dirichlet, neumann)

network.render(s)