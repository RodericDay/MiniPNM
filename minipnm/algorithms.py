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
    fixed = ics!=0
    translate = np.hstack(fixed.nonzero()+(~fixed).nonzero())
    ics = ics[translate]
    
    # if there are unconnected islands, they need to be fixed at zero too
    labels = label(network)
    island_labels = set(labels) - set(labels[fixed])
    fixed = fixed | np.in1d(labels, list(island_labels))

    # build connectivity matrix
    heads, tails = network.pairs.T
    i = np.hstack([heads, tails])
    j = np.hstack([tails, heads])
    fi = fixed.nonzero()[0]
    IC = sparse.coo_matrix((np.ones_like(fi), (fi, fi)), shape=(ics.size, ics.size)).tocsr()
    C = sparse.coo_matrix((np.ones_like(i), (i,j)), shape=(ics.size, ics.size)).tocsr()
    D = sparse.diags(np.asarray(C.sum(axis=1).T)[0], 0).tocsr()
    
    A = sparse.vstack([(IC)[fixed], (C - D)[~fixed]]).tocsr()
    x = spsolve(A, ics)

    return x
