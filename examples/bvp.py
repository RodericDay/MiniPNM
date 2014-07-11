import numpy as np
np.set_printoptions(linewidth=200, edgeitems=30)
from scipy import sparse
import minipnm as mini

N = mini.gaussian_noise([100,100])
network = mini.Cubic(N,[1,1]) - (N < N.mean())
# network = mini.Delaunay.random(100)
# o, i = network.split(network.boundary())
# network = o | i

x,y,z = network.coords
dirichlet = {
    0 : x==x.min(),
}

up_right = (x==x.max()) & (y>y.mean())
neumann = {
    0.5 : network.cut( up_right, network.indexes),
}
s = mini.solve_bvp(network.laplacian, dirichlet, neumann)
influx = np.subtract(*network.cut(x!=x.min(), s)).sum()
if not influx:
    exit('broken network')
else:
    print influx
print s
network['z'] = s/s.max()/2
network.render(s)