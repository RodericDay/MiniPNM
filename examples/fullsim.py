import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import minipnm as mini

class DiffusionSimulation(object):

    def __init__(self, adj, u, insulated=None):
        adjsum = adj.sum(axis=1).A1
        if insulated is None:
            adjsum = adjsum.max()
        else:
            adjsum[~insulated] = adjsum.max()
        self.RHS = -u*adj + sparse.diags(1+adjsum*u, 0)
        self.LHS =  u*adj + sparse.diags(1-adjsum*u, 0)
        self.state = np.zeros_like(adjsum, dtype=float)

    def march(self, changes):
        self.state = spsolve(self.RHS, self.LHS*self.state + changes)
        return self.state


cubic = mini.Cubic([10,10])
rmin = 0.1
radii = np.random.rand(cubic.order)*(0.5-rmin)+rmin
capacities = 4./3. * np.pi * radii**3
radial = mini.Radial(cubic.points, radii, cubic.pairs)
conductance_matrix = radial.adjacency_matrix
conductance_matrix.data = 1./radial.lengths

tgrid = mini.Cubic([100,100], 0.091) # <- weird constant req'd. fix this pls.
x,y,z = tgrid.coords
top = y==y.max()
bottom = y==y.min()
right = x==x.min()
left = x==x.max()

water_generator = (0.01*(radial['x']==0) for _ in iter(int, 1))
water_handler = mini.algorithms.InvasionSimulation(capacities, conductance_matrix)
heat_generator = (0.01*(tgrid['x']==0) for _ in iter(int, 1))
temp_handler = DiffusionSimulation(tgrid.adjacency_matrix, u=100, insulated=top|bottom|left|right)
water_history, temp_history = [], []
for t in range(50):
    water_history.append( water_handler.distribute(next(water_generator)) )
    temp_history.append( temp_handler.march(next(heat_generator)) )

scene = mini.Scene()
scene.add_actors(radial.actors(water_history))
scene.add_actors(tgrid.actors(temp_history, alpha=0.2))
scene.play()
