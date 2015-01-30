import itertools
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

class System(object):
    '''
    this class compartmentalizes the creation of a system to an init
    step, but then allows the direct update of conductance values and 
    application of boundary conditions without rebuilding the entire matrix
    every time, for fast iterations over changing environmental conditions
    '''
    def __init__(self, pairs, flux_units=1, potential_units=1, conductances=None):
        # this may be handled better in the future, but these are unit defs
        self.fu = flux_units
        self.pu = potential_units

        # create nodes / diags
        self.n = n = np.max(pairs) + 1 # we need a better heuristic for stragglers
        diags = np.arange(n) + 1 # otherwise the csr conversion drops the 0 (1)
        D = sparse.diags(diags, 0)

        # create rest / adjacency matrix
        self.m = m = len(pairs)
        k = np.arange(n, n+m) + 1
        j, i = np.transpose(pairs)
        ijk = k, (i, j)
        A = sparse.coo_matrix(ijk, shape=(n, n))

        # creation of a mapping that associates nodes and neighbors
        self.mapping = [[] for _ in diags]
        for row, idx in zip(A.row, A.data-1):
            self.mapping[row].append( idx )
        self.mapping = np.array(self.mapping)

        # combine sparse matrices for the master copy
        self._adj = (A + D)
        self._adj.data -= 1 # recover proper indexing from (1)

        # the reindex is critical, it allows us to map natural arrays to the
        # matrix in csr form
        self.reindex = np.argsort(self._adj.data)

        if conductances is not None:
            self.conductances = conductances

    @property
    def conductances(self):
        '''
        the real conductance array, self._con, matches the number of entries
        in the matrix: n nodes followed by m connections, for quick updates.

        however, the value of the nodes is pre-determined by the value of the
        adjacent connections, so the conductance setter takes care of adjusting
        the first n methods in the array. therefore, to maintain symmetry, a
        call only exposes the last m
        '''
        return self._con[-self.m:] * (self.fu / self.pu)

    @conductances.setter
    def conductances(self, values):
        # unit check
        values = values * (self.pu / self.fu)
        assert not hasattr(values, 'units')

        # create a skeleton and populate the conductance part
        self._con = np.hstack([np.zeros(self.n), np.ones(self.m)*values])

        # add up all the sinks
        for ni, neighbors in enumerate(self.mapping):
            self._con[ni] -= self._con[neighbors].sum()

    def system(self, dbcs={}, k=None):
        A = self._adj.copy()
        A.data[self.reindex] = self._con
        b = np.zeros(self.n)

        # apply boundary conditions by turning equations into definitions
        for bvalue, locations in dbcs.items():
            bvalue = bvalue / self.pu

            A.data[self.reindex[locations]] = 1
            A.data[self.reindex[self.mapping[locations].sum()]] = 0
            b[locations] += bvalue

        # insert linear terms where applicable
        if k is not None:
            fixed = np.sum(list(dbcs.values()), axis=0)
            k /= (self.fu / self.pu)
            A.data[self.reindex[:self.n]] -= np.where(fixed, 0, k)

        return A, b

    def solve(self, dbcs, s=None, k=None):
        '''
        dbcs = dirichlet boundary conditions in {} form
        s = any leftover sink/source terms on the LHS
        k = any linearly dependent terms ie: kx
        '''
        A, b = self.system(dbcs, k)
        fixed = np.sum(list(dbcs.values()), axis=0)
        b = np.where(fixed, b, 0 if s is None else s / self.fu)
        return spsolve(A, b) * self.pu
