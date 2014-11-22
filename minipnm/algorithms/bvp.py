import itertools
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
    this class compartmentalizes the creation of a system to an init
    step, but then allows the direct update of conductance values and 
    application of boundary conditions without rebuilding the entire matrix
    every time, for fast iterations over changing environmental conditions
    '''
    def __init__(self, pairs, conductances):
        # create diags
        self.n = n = np.max(pairs) + 1 # we need a better heuristic for stragglers
        diags = np.arange(n) + 1 # otherwise the csr conversion drops the 0 (1)
        D = sparse.diags(diags, 0)

        # create rest
        self.m = m = len(pairs)
        k = np.arange(n, n+m) + 1
        j, i = np.transpose(pairs)
        ijk = k, (i, j)
        A = sparse.coo_matrix(ijk, shape=(n, n))

        # quick recall map
        self.mapping = [[] for _ in diags]
        for row, idx in zip(A.row, A.data-1):
            self.mapping[row].append( idx )
        self.mapping = np.array(self.mapping)

        # combine to master
        self._adj = (A + D)
        self._adj.data -= 1 # recover proper indexing from (1)
        self.reindex = np.argsort(self._adj.data)

        self.update(conductances)

    def update(self, values):
        '''
        replace conductance values and rebalance nodes for flux conservation
        '''
        # create a skeleton and populate the conductance part
        self._con = np.hstack([np.zeros(self.n), np.ones(self.m)*values])

        # figure out all the sinks
        for sink, neighbors in enumerate(self.mapping):
            self._con[sink] -= self._con[neighbors].sum()

    def system(self, dbcs):
        A = self._adj.copy()
        A.data[self.reindex] = self._con
        b = np.zeros(self.n)

        for bvalue, locations in dbcs.items():
            A.data[self.reindex[locations]] = 1
            A.data[self.reindex[self.mapping[locations].sum()]] = 0
            b[locations] += bvalue

        return A, b

    def solve(self, dbcs, ssterms=0):
        A, b = self.system(dbcs)
        fixed = np.sum(dbcs.values(), axis=0)
        b = np.where(fixed, b, ssterms)
        return spsolve(A, b)
