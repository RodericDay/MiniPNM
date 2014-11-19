import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def solve(system, dirichlet, ssterms=0, neumann=None, units=None):
    '''
    boundary conditions given as dictionaries of { value : mask }
    the length of the Dirichlet masks should be of network.order,
    while Neumann masks are the indices of the tail and head of any specified
    edge gradient.
    '''
    if units is not None:
        potential_units = units
        flux_units = system.units * potential_units
        system = system( flux_units / potential_units )
        if ssterms is not 0:
            ssterms = ssterms( flux_units )
        if potential_units is not 1:
            dirichlet = { k(potential_units):v for k,v in dirichlet.items() }
            if neumann is not None:
                neumann = { k(potential_units):v for k,v in neumann.items() }

    A, b = build(system, dirichlet, ssterms, neumann, fast=True)
    x = spsolve(A, b).round(5)
    return x * (1 if units is None else units)

def build(system, dirichlet, ssterms=0, neumann=None , fast=False):
    if neumann is None: neumann = {}
    n_conditions_imposed = count_conditions(system, dirichlet, neumann)

    free = n_conditions_imposed == 0
    D = sparse.eye(system.shape[0]).tocsr()
    elements_of_A = [system[free]] if any(free) else []
    elements_of_b = [np.zeros(free.sum())]
    reindex = free.nonzero()

    for value, mask in dirichlet.items():
        if not any(mask):
            raise Exception("BC value <{}> has no domain.".format(value))
        elements_of_A.append( D[mask.nonzero()] )
        elements_of_b.append( value * np.ones(mask.sum()) )
        reindex += mask.nonzero()

    for v, (t, h) in neumann.items():
        if not any(t):
            raise Exception("BC value <{}> has no domain.".format(v))
        count = np.bincount(t)[np.unique(t)]
        elements_of_A.append( system[np.unique(t)] )
        elements_of_b.append( np.true_divide(v, count.sum()) * count )
        reindex += t.nonzero()

    reindex = np.hstack(reindex).astype('int32')

    if fast:
        A = sparse.vstack(e for e in elements_of_A if e.nnz)
        b = np.hstack(elements_of_b) + ssterms
        return A, b

    # reorders the system to preserve original structure
    A = sparse.vstack((e for e in elements_of_A if e.nnz), format='csc')
    A.indices = reindex[A.indices]
    b = np.hstack(elements_of_b)
    b[reindex] = b.copy()
    return A, b

def count_conditions(system, dirichlet, neumann):
    n_conditions_imposed = sum(dirichlet.values())

    _, label = sparse.csgraph.connected_components(system)
    isolated = set(label[n_conditions_imposed==0]) - set(label[n_conditions_imposed>0])
    targets = np.in1d(label, list(isolated))
    if any(targets):
        dirichlet[0] = dirichlet.get(0, np.zeros_like(label)) | targets
    n_conditions_imposed[targets] += 1

    target_idxs = targets.nonzero()[0]
    for v, (tail_ids, head_ids) in neumann.items():
        zeroed_out = np.in1d(tail_ids, target_idxs)
        neumann[v] = (tail_ids[~zeroed_out], head_ids[~zeroed_out])
        n_conditions_imposed[neumann[v][0]] += 1

    if any(n_conditions_imposed > 1):
        raise Exception("Overlapping BCs")
    return n_conditions_imposed


class System(object):
    '''
    this class compartmentalizes the creation of a system with BCs to an init
    step, but then allows the direct update of conductance values without
    rebuilding the entire matrix every time, for fast iterations over changing
    environmental conditions
    '''
    def __init__(self, pairs, dbcs, conductances):
        # gather units in case they are implemented, for typechecks
        self.cu = conductances.units if hasattr(conductances, 'units') else 1
        bcunits = [k.units if hasattr(k, 'units') else 1 for k in dbcs.keys()]
        assert all(bcunits[0]==u for u in bcunits)
        self.pu = bcunits[0]
        self.fu = self.cu * self.pu

        # if the node is not free, it will be a fixed value. consider rest.
        # warning: does not check for overlap
        self.free = ~np.sum( dbcs.values(), axis=0, dtype=bool )
        i, j = pairs.T
        self.valid = np.in1d(j, self.free.nonzero())
        i, j = i[self.valid], j[self.valid] # adjusting

        # first block: adjacencies
        k = np.arange( self.valid.sum() ) + 1
        n = self.free.size
        ijk = k, (j, i)
        A1 = sparse.coo_matrix(ijk, shape=(n, n))

        # second block: balances
        free_ids = self.free.nonzero()[0]
        l = np.arange(A1.nnz+1, A1.nnz+1 + free_ids.size)
        lmn = l, (free_ids, free_ids)
        A2 = sparse.coo_matrix(lmn, shape=(n, n))

        # third block: definitions
        bv_ids = (~self.free).nonzero()[0]
        o = np.arange(A1.nnz+A2.nnz+1, A1.nnz+A2.nnz+1+bv_ids.size)
        opq = o, (bv_ids, bv_ids)
        A3 = sparse.coo_matrix(opq, shape=(n, n))

        # save for recall
        self.indexed = (A1 + A2 + A3).tocsr().astype(float)
        self.reindexer = np.argsort(self.indexed.data)
        self.A1, self.A2, self.A3 = A1, A2, A3

        # delegation
        self.bvals = np.sum( value/self.pu * locations for value, locations in dbcs.items() )
        self.update(conductances)

    def update(self, conductances):
        '''
        upon receiving conductances we need to set them in the system matrix,
        and rebalance the nodes for flux conservation
        '''
        self.system = self.indexed.copy()
        conductances = (self.valid*conductances/self.cu)[self.valid]

        self.system.data[self.reindexer[self.A1.data-1]] = conductances
        self.system.data[self.reindexer[self.A2.data-1]] = 0
        sums = -self.system.sum(axis=1).A1
        self.system.data[self.reindexer[self.A2.data-1]] = sums[self.free]
        self.system.data[self.reindexer[self.A3.data-1]] = 1

    def solve(self, ssterms=0):
        A = self.system
        b = np.where(self.free, ssterms/self.fu, self.bvals)
        return spsolve(A, b) * self.pu
