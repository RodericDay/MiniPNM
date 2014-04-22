import numpy as np

def label(network):
    from scipy import sparse

    V, E = network.size
    N = sparse.lil_matrix((V, V))
    hs, ts = network.pairs.T
    N[hs, ts] = 1
    N[ts, hs] = 1
    return sparse.csgraph.connected_components(N)[1]

def solve_linear(network, ics):
    fixed = (ics != 0)

    # if there are unconnected islands, they need to be fixed at zero too
    labels = label(network)
    island_labels = set(labels) - set(labels[fixed])
    fixed = fixed | np.in1d(labels, list(island_labels))

    # build connectivity matrix
    A = np.zeros([ics.size, ics.size])
    heads, tails = network.pairs.T
    A[heads, tails] = -1
    A[tails, heads] = -1
    # balance sinks and sources
    A[fixed] = 0
    A -= np.eye(ics.size)*A.sum(axis=1)
    # fix initial conditions
    A[fixed, fixed] = 1

    # verify initial conditions fulfill requirements
    assert np.allclose(A.sum(axis=1), fixed)

    x = np.linalg.solve(A, ics)
    # verify solution matches at boundaries
    assert np.allclose(x[fixed], ics[fixed])

    return x