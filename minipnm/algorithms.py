import numpy as np

def label(network):
    from scipy import sparse

    V, E = network.size
    N = sparse.lil_matrix((V, V))
    hs, ts = network.pairs.T
    N[hs, ts] = 1
    N[ts, hs] = 1
    return sparse.csgraph.connected_components(N)

def simple_linear_solver(network, ics):
    fixed = (ics != 0)

    # if there are unconnected islands, they need to be fixed at zero too
    nzones, labels = label(network)
    island_labels = set(labels) - set(labels[b.nonzero()])
    fixed = fixed | np.in1d(labels, list(island_labels))

    # build connectivity matrix
    A = np.zeros([ics.size, ics.size])
    heads, tails = network.pairs.T
    A[heads, tails] = -1
    A[tails, heads] = -1
    A[fixed] = 0

    # ensure sinks and sources are balanced
    A -= np.eye(b.size)*A.sum(axis=1)
    # fix initial conditions
    A[fixed, fixed] = 1
    # verify initial conditions fulfill requirements
    assert np.allclose(A.sum(axis=1), 1*fixed)

    x = np.linalg.solve(A, b)
    # verify solution matches at boundaries
    assert np.allclose(x[b.nonzero()], b[b.nonzero()])

    return x

if __name__ == '__main__':
    import os
    import minipnm as mini

    dirname = '/Users/roderic/Developer/Thesis/Crack Study/Images/'
    filename= os.listdir(dirname)[0]
    source  = dirname+filename
    # network = mini.from_image(source, dmax=30)
    # network = network - (network['intensity'] > 0.04)
    network = mini.Delaunay(np.random.rand(1000,3))

    # transform
    network = network - network.boundary()
    # turn into method
    x,y,z = network.coords
    w,h,t = network.dims
    block1 = (x > w*0.25) & (x < w*0.4) & (y > h*0.2)
    block2 = (x > w*0.55) & (x < w*0.8) & (y < h*0.8)
    network = network - (block1 | block2)

    # ics
    x, y, z = network.coords
    b = 100*(x==x.min()) + 10*(x==x.max())

    network.preview(simple_linear_solver(network, b))
