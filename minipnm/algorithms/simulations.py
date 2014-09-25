import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from .. import graphics


class Simulation(object):
    '''
    The base simulation object is a utility class that performs some
    boilerplate functions.

    1) Keep track of a state history
    2) Handle the sparse aspects of pore blocking and unblocking
    '''

    def __init__(self, cmat, base=0):
        self._cmat = cmat
        n, m = cmat.shape
        assert n==m
        self.history = [base*np.ones(n)]

    @property
    def state(self):
        return self.history[-1].copy()

    @state.setter
    def state(self, entry):
        self.history.append(entry)

    def march(self):
        self.state = self.state

    def block(self, nodes=True):
        '''
        blocking turns history entries imaginary to indicate they are nulled,
        while retaining the magnitude

        True blocks all, None/False blocks None
        '''
        # numpy has weird defaults for slicing, so we hack it a little bit
        # to adapt to our English intuition
        if nodes is False or nodes is None: nodes=[] # select None
        elif nodes is True: nodes=None # select All
        indexes = np.arange(self.state.size)
        subset = indexes[nodes]
        mask = np.in1d(indexes, subset)
        self.history[-1] = np.absolute(self.state) * np.where(mask, 1j, 1)

    @property
    def blocked(self):
        return self.state.imag != 0

    @property
    def valid(self):
        '''
        return a boolean mask of valid connections by index
        that is, connections where neither head nor sink is blocked
        '''
        accessible = (self.state.imag == 0).nonzero()[0]
        good_tails = np.in1d(self._cmat.col, accessible)
        good_heads = np.in1d(self._cmat.row, accessible)
        return good_tails & good_heads

    @property
    def cmat(self):
        '''
        return a mask over the cmat s.t blocked (imaginary) nodes are isolated
        '''
        data = self._cmat.data[self.valid]
        col = self._cmat.col[self.valid]
        row = self._cmat.row[self.valid]
        return sparse.coo_matrix((data, (col, row)), self._cmat.shape)

    def render(self, points, scene, **kwargs):
        pairs = np.vstack([self._cmat.col, self._cmat.row]).T
        wires = graphics.Wires(points, pairs, np.absolute(self.history), **kwargs)
        scene.add_actors([wires])

    def __str__(self):
        return str(np.vstack(self.history).real)


class Diffusion(Simulation):
    '''
    Using Crank-Nicholson discretized in time and space
    '''

    def __init__(self, cmat, nCFL=1, insulated=False, base=0):
        super(Diffusion, self).__init__(cmat, base)
        self.insulated = insulated
        self.nCFL = nCFL
        self.build()

    def block(self, nodes):
        super(Diffusion, self).block(nodes)
        self.build()

    def build(self):
        csum = self.cmat.sum(axis=1).A1
        if self.insulated is True:
            pass
        elif self.insulated is False:
            csum[:] = csum.max()
        else:
            csum[~self.insulated] == csum.max()
        u = self.nCFL
        self.RHS = -u*self.cmat + sparse.diags(1+u*csum, 0)
        self.LHS =  u*self.cmat + sparse.diags(1-u*csum, 0)

    def march(self, inputs=0):
        x = spsolve(self.RHS, self.LHS*self.state.real + inputs)
        self.state = np.where(self.blocked, self.state, x)


class Invasion(object):
    '''
    ALOP simulatneous invasion with fractional generation.

    Negative saturation values are assumed to be frozen.
    '''
    class NeighborsSaturated(Exception):
        msg = "All neighbors of node {} saturated. Network saturation: {}%"
        def __init__(self, node, saturation):
            self.node = node
            self.saturation = saturation
        def __str__(self):
            return self.msg.format(self.node, self.saturation.mean()*100)

    def __init__(self, cmat, capacities):
        self.cmat = cmat
        self.capacities = capacities
        self.saturation = np.zeros_like(capacities)
        self.history = [np.zeros_like(capacities)]
        self.blocked = np.zeros_like(capacities, dtype=bool)

    @property
    def state(self):
        return self.history[-1].copy()

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
                    self.history.append( self.saturation * np.where(self.blocked, -1, 1) )
                    raise self.NeighborsSaturated(node, self.saturation)
                excess[recipient] += excess[node]
                excess[node] = 0
            return self.distribute(excess)
        self.history.append( self.saturation * np.where(self.blocked, -1, 1) )
        return self.saturation

    def find_unsaturated_neighbor(self, node):
        self.update_pressurized_clusters()
        full_sources = self.labels[self.cmat.col]==self.labels[node]
        non_full_sinks = self.saturation[self.cmat.row] < 0.999
        not_blocked = self.cmat.data > 0
        viable_throats = full_sources & non_full_sinks & not_blocked
        viable_throats = viable_throats.nonzero()[0] # bool -> idxs
        if len(viable_throats) == 0:
            return node
        best_throat = max(viable_throats, key=self.cmat.data.item)
        neighbor = self.cmat.row[best_throat]
        return neighbor

    def update_pressurized_clusters(self):
        '''
        this only needs to be checked when pores are newly saturated
        '''
        full_nodes = (self.saturation > 0.999).nonzero()[0]
        # superclusters require pressurized throats
        pressurized_throats = np.in1d(self.cmat.row, full_nodes) \
                            & np.in1d(self.cmat.col, full_nodes)
        i = self.cmat.row[pressurized_throats]
        j = self.cmat.col[pressurized_throats]
        v = np.ones_like(i)
        s = self.capacities.size
        coo = sparse.coo_matrix((v, (i, j)), shape=(s,s))
        self.labels = sparse.csgraph.connected_components(coo)[1]

    def block(self, nodes):
        self.blocked = nodes
        indexes = nodes.nonzero()[0]
        disconnected = np.in1d(self.cmat.col, indexes)
        self.cmat.data = np.abs(self.cmat.data) * np.where(disconnected, -1, 1)

    def render(self, points, scene):
        fill_radii = (self.capacities*np.abs(self.history)*3./4./np.pi)**(1./3.) * np.sign(self.history)

        balloons  = graphics.Spheres(points, fill_radii.clip( 0, np.inf), color=(0,0,1))
        scene.add_actors([balloons])

        snowballs = graphics.Spheres(points,-fill_radii.clip(-np.inf, 0), color=(1,0,0))
        scene.add_actors([snowballs])
