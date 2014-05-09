import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import spsolve

def linear_solve(network, dbcs):
    # the matrix to be solved will involve a stack of ICs and some balances
    # take as fixed the points where dbcs is defined
    fixed = dbcs!=0

    # if there are unconnected islands, they need to be fixed at zero too
    labels = network.labels
    island_labels = set(labels[~fixed]) - set(labels[fixed])
    fixed = fixed | np.in1d(labels, list(island_labels))

    # this array shifts the ICs around in the correct way
    mapping = np.hstack(fixed.nonzero()+(~fixed).nonzero())

    # we build a diagonal matrix for dbcs
    IC= sparse.diags(np.ones_like(dbcs), 0).tocsr()
    # a connectivity matrix representing sources
    C = network.connectivity_matrix
    # and a diagonal matrix representing sinks, as sum of sources
    D = sparse.diags(C.sum(axis=1).A1, 0).tocsr() # A1 is matrix -> array
    # the system matrix is a stack of the relevant rows of each
    A = sparse.vstack([(IC)[fixed], (C - D)[~fixed]]).tocsr()
    x = spsolve(A, dbcs[mapping])

    assert( np.allclose(x[fixed], dbcs[fixed]) )

    return x

def percolation(network, sources, thresholds, condition_range,
                base=0.5, rate=1.3):
    network = network.copy()
    # this takes advantage of the auto-scaling of network properties
    network['indexes'] = network.indexes
    network['sources'] = sources
    network['thresholds'] = thresholds

    saturation = np.empty([len(condition_range), network.size[0]])
    for i, condition in enumerate(np.atleast_1d(condition_range)):

        # create placeholder, then replace where applicable
        saturation[i] = np.zeros(network.size[0])

        # threshold-bound accessibility
        inaccessible = network['thresholds']>condition
        sub_network = network - inaccessible

        # topologically-bound accessibility
        accessible_labels = sub_network.labels[sub_network['sources']]
        inaccessible = ~np.in1d(sub_network.labels, accessible_labels)
        sub_network = sub_network - inaccessible

        def late_pore_fill(ratios, base, rate):
            if rate==0:
                return np.ones_like(ratios)*base
            return base+(1-base)*(1-ratios)**np.true_divide(1,rate)

        ratios = np.true_divide(sub_network['thresholds'], condition)
        saturation[i][sub_network['indexes']] = late_pore_fill(ratios, base, rate)

    return saturation
