import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from . import graphics

class Simulation(object):
    '''
    The base simulation object is a utility class that performs some
    boilerplate functions.

    1) Keep track of a state history
    2) Handle the sparsity aspects of pore blocking and unblocking
    '''
    class BlockedStateChange(Exception):
        msg = "Unblock nodes before attempting state change."
        def __str__(self):
            return self.msg


    def __init__(self, cmat, base=0):
        m, n = cmat.shape
        assert m==n
        self._cmat = cmat
        self.state_history = [base*np.ones(n)]
        self.block_history = {}
        self.block(None)

    def reset(self, base=None):
        if base is None:
            self.state_history = [self.history[0]]
        else:
            self.state_history = [base*np.ones_like(self.state, dtype=float)]
        self.block_history = {}
        self.block(None)

    @property
    def indexes(self):
        return np.arange(self.state.size)

    @property
    def state(self):
        return self.state_history[-1].copy()

    @state.setter
    def state(self, entry):
        diff = self.state - entry
        if not np.allclose(diff[self.blocked], 0):
            raise self.BlockedStateChange()
        self.state_history.append(entry)

    @property
    def step(self):
        '''
        returns relative simulation 'step' for internal counting methods
        '''
        return len(self.state_history)-1

    @property
    def history(self):
        '''
        returns the state history. made into a property to encourage more
        sophisticated simulations to return arrays that take more into
        account
        '''
        return np.vstack(self.state_history)

    def block(self, targets=True):
        '''
        special options for target argument:
            True  : blocks all
            None  : blocks none
            False : blocks none
        '''
        if targets is False or targets is None: targets=[] # selects None
        elif targets is True: targets=None # selects All
        else: pass # assumes targets contains indexes
        subset = self.indexes[targets]
        self.block_history[self.step] = np.in1d(self.indexes, subset)
        self._filtered_cmat = None # set to None to force update

    @property
    def blocked(self):
        return self.block_history[max(self.block_history.keys())]

    def expand_block_states(self):
        '''
        the block history is stored in dictionary form
        this convenience function provides it in an array form that matches
        the state history
        '''
        bins = np.array(sorted(self.block_history.keys()))
        return np.array([self.block_history[step] for step in 
                bins[np.digitize(range(len(self.history)), bins)-1]])

    def valid_edges(self):
        '''
        return a boolean mask of valid edges, which are
        edges where neither head nor tail is blocked
        '''
        accessible = (~self.blocked).nonzero()[0]
        good_tails = np.in1d(self._cmat.col, accessible)
        good_heads = np.in1d(self._cmat.row, accessible)
        return good_tails & good_heads

    @property
    def cmat(self):
        '''
        returns a transformed conductance matrix without
        connections from or to blocked pores
        '''
        if self._filtered_cmat is None:
            value_filter = self.valid_edges()
            data = self._cmat.data[value_filter]
            col = self._cmat.col[value_filter]
            row = self._cmat.row[value_filter]
            shape = self._cmat.shape
            self._filtered_cmat = sparse.coo_matrix((data, (row, col)), shape)
        return self._filtered_cmat

    def render(self, points, scene=None, **kwargs):
        pairs = np.vstack([self._cmat.col, self._cmat.row]).T
        wires = graphics.Wires(points, pairs, self.history, **kwargs)
        scene.add_actors([wires])

    def steady_state(self, steps_back=5):
        '''
        naively checks to see if the changes in history have stagnated
        '''
        if self.step < steps_back:
            return False

        a = self.history[-steps_back]
        for i in range(1, steps_back):
            b = self.history[-i]
            if not np.allclose(a, b, rtol=1e-03, atol=1e-06):
                return False
        return True


class Diffusion(Simulation):
    '''
    Convection-diffusion equation

    State refers to molar concentration when doing mass diffusion,
    Temperature when doing heat diffusion

    Discretized by a weird extrapolation of the regular stencil in space,
    and Crank-Nicholson in time
    '''

    def __init__(self, cmat, nCFL=1, base=0, dbcs=None):
        self.nCFL = nCFL
        self.dbcs = {} if dbcs is None else dbcs
        super(Diffusion, self).__init__(cmat, base)

    def block(self, nodes):
        super(Diffusion, self).block(nodes)
        self.build()

    def build(self):
        csum = self.cmat.sum(axis=1).A1
        u = self.nCFL
        self.RHS = -u*self.cmat + sparse.diags(1+u*csum, 0)
        self.LHS =  u*self.cmat + sparse.diags(1-u*csum, 0)

    def march(self, inputs=0):
        losses = 0
        for value, locs in self.dbcs.items():
            losses += (self.state-value)*locs
        A = self.RHS.astype(float)
        b = self.LHS*self.state + inputs - losses
        x = spsolve(A, b)
        self.state = np.where(self.blocked, self.state, x) # dubious


class Invasion(Simulation):
    '''
    ALOP simulatneous invasion with fractional generation.
    '''
    class NeighborsSaturated(Exception):
        msg = "All neighbors of node {} saturated. Network saturation: {}%"
        def __init__(self, node, saturation):
            self.node = node
            self.saturation = saturation
        def __str__(self):
            return self.msg.format(self.node, self.saturation.mean()*100)


    def __init__(self, cmat, capacities, base=0):
        super(Invasion, self).__init__(cmat, base)
        self.capacities = np.ones_like(self.state) * capacities
        self.saturation = self.state
        self.update_pressurized_clusters()

    def distribute(self, generation=0):
        content = self.capacities*self.saturation + generation
        excess = content.clip(self.capacities, content) - self.capacities
        content -= excess
        self.saturation = content / self.capacities
        self.update_pressurized_clusters()
        excess = self.pool(excess, self.labels)
        if any(excess):
            for node in excess.nonzero()[0]:
                recipient = self.find_unsaturated_neighbor(node)
                excess[recipient] += excess[node]
                excess[node] = 0
            return self.distribute(excess)
        self.state = self.saturation
        return self.saturation

    @staticmethod
    def pool(excess, labels):
        '''
        consider that a single pore infused with the volumetric capacity of
        the entire network will sucessively spillover until it fills the whole
        thing. This behaviour cannot be improved upon.

        However, when multiple pores have excess production in the same time
        step, we can curtail unnecessary calls to intensive methods, by
        ensuring that all the excesses are concentrated in a single pore within
        every cluster. The spillover of a cluster is the sum of every spillover
        of its individual pores.

        The pool method "lumps" scalars based on provided labeling. Numpy 2.0
        will introduce a `find` function for arrays, which should improve upon
        the slow python list loop.

        >>> Invasion.pool([0.1, 0.1, 0.1, 0.1, 0.1], [0, 0, 0, 1, 1])
        array([ 0.3,  0. ,  0. ,  0.2,  0. ])
        '''
        vals = np.array(excess)
        idxs = np.arange(vals.size)
        mask = np.array(labels)
        ijv = (vals, (idxs, mask))

        binned = sparse.coo_matrix(ijv).sum(axis=0).A1
        lst = mask.tolist()
        vals *= 0
        for i,v in enumerate(binned):
            vals[lst.index(i)] = v
        return vals

    def find_unsaturated_neighbor(self, node):
        viable_throats = self.find_frontier_throats(node)
        if len(viable_throats) == 0:
            self.state = self.saturation
            raise self.NeighborsSaturated(node, self.saturation)
        best_throat = max(viable_throats, key=self.cmat.data.item)
        neighbor = self.cmat.row[best_throat]
        return neighbor

    def find_frontier_throats(self, node):
        full_sources = self.labels[self.cmat.col]==self.labels[node]
        non_full_sinks = self.saturation[self.cmat.row] < 0.999
        viable_throats = full_sources & non_full_sinks
        return viable_throats.nonzero()[0]

    def update_pressurized_clusters(self):
        '''
        this only needs to be run when pores are newly saturated
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

    def render(self, points, scene):
        balloons  = graphics.Spheres(points, self._radii(self.history), color=(0,0,1))
        scene.add_actors([balloons])

        block_states = self.expand_block_states()
        snowballs = graphics.Spheres(points, self._radii(block_states), color=(1,0,0))
        scene.add_actors([snowballs])

    def _radii(self, saturation=1):
        return (self.capacities*saturation/np.pi*3./4.)**(1./3.)
