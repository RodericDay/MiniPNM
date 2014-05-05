import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import spsolve

def linear_solve(network, ics):
    # the matrix to be solved will involve a stack of ICs and some balances
    # take as ICs points where ics is defined
    fixed = ics!=0

    # if there are unconnected islands, they need to be fixed at zero too
    labels = network.clusters
    island_labels = set(labels[~fixed]) - set(labels[fixed])
    fixed = fixed | np.in1d(labels, list(island_labels))

    # this array shifts the ICs around in the correct way
    mapping = np.hstack(fixed.nonzero()+(~fixed).nonzero())

    # we build a diagonal matrix for ics
    IC= sparse.diags(np.ones_like(ics), 0).tocsr()
    # a connectivity matrix representing sources
    C = network.connectivity_matrix
    # and a diagonal matrix representing sinks, as sum of sources
    D = sparse.diags(C.sum(axis=1).A1, 0).tocsr() # A1 is matrix -> array
    # the system matrix is a stack of the relevant rows of each
    A = sparse.vstack([(IC)[fixed], (C - D)[~fixed]]).tocsr()
    x = spsolve(A, ics[mapping])

    assert( np.allclose(x[fixed], ics[fixed]) )

    return x
