import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import spsolve

def label(network):
    V, E = network.size
    hs, ts = network.pairs.T
    fwds = np.hstack([hs, ts])
    bkds = np.hstack([ts, hs])
    N = sparse.coo_matrix((np.ones_like(fwds), (fwds, bkds)), shape=(V, V))
    return sparse.csgraph.connected_components(N)[1]

def linear_solve(network, ics):
    # the matrix to be solved will involve a stack of ICs and some balances
    # take as ICs points where ics is defined
    fixed = ics!=0

    # if there are unconnected islands, they need to be fixed at zero too
    labels = label(network)
    island_labels = set(labels[~fixed]) - set(labels[fixed])
    fixed = fixed | np.in1d(labels, list(island_labels))

    # this array shifts the ICs around in the correct way
    mapping = np.hstack(fixed.nonzero()+(~fixed).nonzero())

    # we build a diagonal matrix for ics
    IC= sparse.diags(np.ones_like(ics), 0).tocsr()
    # a connectivity matrix representing sources
    heads, tails = network.pairs.T
    i = np.hstack([heads, tails])
    j = np.hstack([tails, heads])
    C = sparse.coo_matrix((np.ones_like(i), (i,j)), shape=(ics.size, ics.size)).tocsr()
    # and a diagonal matrix representing sinks, as sum of sources
    D = sparse.diags(C.sum(axis=1).A1, 0).tocsr() # A1 is matrix -> array
    # the system matrix is a stack of the relevant rows of each
    A = sparse.vstack([(IC)[fixed], (C - D)[~fixed]]).tocsr()
    x = spsolve(A, ics[mapping])

    assert( np.allclose(x[fixed], ics[fixed]) )

    return x
