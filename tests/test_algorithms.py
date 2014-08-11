import numpy as np
import minipnm as mini

def test_invasion():
    network = mini.Cubic.empty([20,20])
    x,y,z = network.coords
    sol = mini.algorithms.invasion(network.adjacency_matrix, x==x.min(), x==x.max())

def test_shortest_path():
    string = '''
    131 673 234 103  18
    201  96 342 965 150
    630 803 746 422 111
    537 699 497 121 956
    805 732 524  37 331
    '''
    matrix = np.matrix([[int(w) for w in row.split(' ') if w]
                       for row in string.strip().split('\n')])
    matrix = np.rot90(matrix, 3)
    network = mini.Cubic(matrix)
    cmat = network.adjacency_matrix
    cmat.data = matrix.A1[cmat.col]
    # some geometric definitions
    x,y,z = network.coords
    top_left = network.indexes[(x==x.min()) & (y==y.max())]
    bottom_right = network.indexes[(x==x.max()) & (y==y.min())]
    left = network.indexes[x==x.min()]
    right = network.indexes[x==x.max()]
    # top-left to bottom-right;           right and down:           2427
    blocked = (x[cmat.row] > x[cmat.col]) | (y[cmat.row] < y[cmat.col])
    np.set_printoptions(linewidth=200)
    cmat1 = cmat.copy()
    cmat1.row = cmat.row[~blocked]
    cmat1.col = cmat.col[~blocked]
    cmat1.data = cmat.data[~blocked]
    path1 = mini.algorithms.shortest_path(cmat1, top_left, bottom_right)
    assert matrix.A1[path1].sum() == 2427
    # any left start to any right end;    left, right, up, down:     994
    path2 = mini.algorithms.shortest_path(cmat, left, right)
    assert matrix.A1[path2].sum() == 994
    # top-left to bottom-right;           left, right, up, down:    2297
    path3 = mini.algorithms.shortest_path(cmat, top_left, bottom_right)
    assert matrix.A1[path3].sum() == 2297
