import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from .. import graphics


class Simulation(object):
    '''
    Automates keeping track of an evolving history
    '''
    @property
    def state(self):
        return self.history[-1].copy()

    @state.setter
    def state(self, array):
        self.history.append(array)


class Diffusion(Simulation):
    '''
    Crank-Nicholson method
    cmat ~ conductance matrix
    u ~ CFL number
    '''
    def __init__(self, cmat, u, insulated=False, base=0):
        self.cmat = cmat
        adjsum = cmat.sum(axis=1).A1
        if insulated is True:
            pass
        elif insulated is False:
            adjsum[:] = adjsum.max()
        else:
            adjsum[~insulated] = adjsum.max()
        self.RHS = -u*cmat + sparse.diags(1+adjsum*u, 0)
        self.LHS =  u*cmat + sparse.diags(1-adjsum*u, 0)
        self.history = [np.ones_like(adjsum, dtype=float) * base]

    def march(self, changes=0):
        self.state = spsolve(self.RHS, self.LHS*self.state + changes)

    def render(self, points, scene, **kwargs):
        pairs = np.vstack([self.cmat.col, self.cmat.row]).T
        wires = graphics.Wires(points, pairs, self.history, **kwargs)
        scene.add_actors([wires])


class Invasion(Simulation):
    '''
    This class simulates alop invasion with fractional generation in arbitrary
    pores simultaneously
    '''
    class NeighborsSaturated(Exception):
        msg = "All neighbors of node {} saturated. Network saturation: {}%"
        def __init__(self, node, saturation):
            self.node = node
            self.saturation = saturation
        def __str__(self):
            return self.msg.format(self.node, self.saturation.mean()*100)

    def __init__(self, capacities, conductance_matrix):
        self.capacities = capacities
        self.con = conductance_matrix
        self.saturation = np.zeros_like(capacities)
        self.history = [np.zeros_like(capacities)]
        self.block_history = [np.zeros_like(capacities, dtype=bool)]

    def until_saturation(self, generator):
        while True:
            try:
                self.distribute(next(generator))
            except self.NeighborsSaturated as e:
                break
        return self.history

    def distribute(self, generation=0):
        content = self.capacities*self.saturation + generation
        excess = content.clip(self.capacities, content) - self.capacities
        content -= excess
        self.saturation = content / self.capacities
        if any(excess):
            for node in excess.nonzero()[0]:
                recipient = self.find_unsaturated_neighbor(node)
                if node == recipient:
                    raise self.NeighborsSaturated(node, self.saturation)
                excess[recipient] += excess[node]
                excess[node] = 0
            return self.distribute(excess)
        self.history.append( self.saturation )
        return self.saturation

    def find_unsaturated_neighbor(self, node):
        self.update_pressurized_clusters()
        full_sources = self.labels[self.con.col]==self.labels[node]
        non_full_sinks = self.saturation[self.con.row] < 0.999
        not_blocked = self.con.data > 0
        viable_throats = full_sources & non_full_sinks & not_blocked
        viable_throats = viable_throats.nonzero()[0] # bool -> idxs
        if len(viable_throats) == 0:
            return node
        best_throat = max(viable_throats, key=self.con.data.item)
        neighbor = self.con.row[best_throat]
        return neighbor

    def update_pressurized_clusters(self):
        '''
        this only needs to be checked when pores are newly saturated
        '''
        full_nodes = (self.saturation > 0.999).nonzero()[0]
        # superclusters require pressurized throats
        pressurized_throats = np.in1d(self.con.row, full_nodes) \
                            & np.in1d(self.con.col, full_nodes)
        i = self.con.row[pressurized_throats]
        j = self.con.col[pressurized_throats]
        v = np.ones_like(i)
        s = self.capacities.size
        coo = sparse.coo_matrix((v, (i, j)), shape=(s,s))
        self.labels = sparse.csgraph.connected_components(coo)[1]

    def block(self, nodes):
        self.block_history.append(self.block_history[-1] + nodes)
        indexes = nodes.nonzero()[0]
        blocked = np.in1d(self.con.col, indexes)
        self.con.data[blocked] = -1

    def render(self, points, scene):
        fill_radii = (self.capacities*self.history*3./4./np.pi)**(1./3.)
        balloons = graphics.Spheres(points, fill_radii, color=(0,0,1))
        scene.add_actors([balloons])

        radii = (self.capacities*3./4./np.pi)**(1./3.)
        snowballs = graphics.Spheres(points, radii*self.block_history, alpha=0.75)
        scene.add_actors([snowballs])
