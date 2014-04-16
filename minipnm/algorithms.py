import numpy as np

if __name__ == '__main__':
    import minipnm as mini
    source = '/home/harday/Crack Study/Images/1050+C day1-Q.tif'
    network = mini.from_image(source, 60)

    x,y,z = network.coords
    network['voltage'] = 0.8*(x==x.min()) + 0.2*(x==x.max())
    network.preview(network['voltage'])

    # the ICs are just
    b = network['voltage']
    b = b.reshape(b.shape+(1,))

    # create connectivity matrix
    A = np.zeros([b.size, b.size])
    for hi, ti in network.pairs:
        A[hi,ti] = 1
        A[ti,hi] = 1
    row_totals = A.sum(axis=1)
    indexes = np.arange(b.size)
    A[indexes, indexes] = -row_totals

    # some cells have initial conditions imposed upon them
    A[b.nonzero(),:] = 0
    A[b.nonzero(),b.nonzero()] = 1

    # for those cells with no connections, just fix them at zero?
    # we may need to axe all the isolated clusters via labeling hth
    isolated = np.arange(b.size)[~np.any(A, axis=1)]
    A[isolated, isolated] = 1

    sol = np.linalg.solve(A, b)
    network.preview(sol)