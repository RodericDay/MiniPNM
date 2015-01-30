import itertools as it
import numpy as np
import minipnm as mini

def test_bvp(flux_units=1, potential_units=1,
    diagonal_terms=0, rhs_terms=0):
    # units
    conductance_units = flux_units / potential_units
    # domain
    cubic = mini.Cubic([10, 1, 1])
    x, y, z = cubic.coords
    # init
    system = mini.bvp.System(cubic.pairs,
        flux_units=flux_units,
        potential_units=potential_units)
    system.conductances = 1 * conductance_units
    # solve
    expected = np.linspace(2, 5, 10)
    solution = system.solve(
        {
            2 * potential_units : x==x.min(),
            5 * potential_units : x==x.max(),
        })
    solution /= potential_units
    assert np.allclose(solution, expected)

try:
    import MKS
    MKS.define(globals())
    test_bvp_with_units = lambda: test_bvp(flux_units=A, potential_units=V)
except ImportError:
    pass

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
    network = mini.Cubic.from_source(matrix)
    cmat = network.adjacency_matrix
    cmat.data = matrix.A1[cmat.col]
    # some geometric definitions
    x,y,z = network.coords
    top_left = network.indexes[(x==x.min()) & (y==y.max())]
    bottom_right = network.indexes[(x==x.max()) & (y==y.min())]
    left = network.indexes[x==x.min()]
    right = network.indexes[x==x.max()]
    # block right and upwards directions
    blocked = (x[cmat.row] > x[cmat.col]) | (y[cmat.row] < y[cmat.col])
    cmat1 = cmat.copy()
    cmat1.row = cmat.row[~blocked]
    cmat1.col = cmat.col[~blocked]
    cmat1.data = cmat.data[~blocked]
    # calls
    path1 = mini.algorithms.shortest_path(cmat1, top_left, bottom_right)
    assert matrix.A1[path1].sum() == 2427
    path2 = mini.algorithms.shortest_path(cmat, left, right)
    assert matrix.A1[path2].sum() == 994
    path3 = mini.algorithms.shortest_path(cmat, top_left, bottom_right)
    assert matrix.A1[path3].sum() == 2297

def test_poisson_disk_sampling():
    xyz, r = mini.algorithms.poisson_disk_sampling(r=1, bbox=[10,10])
    # no colliding spheres
    collisions = []
    for i,j in it.combinations(range(len(r)), 2):
        r2 = ( r[i] + r[j] )**2
        d2 = sum( (p1-p2)**2 for p1, p2 in zip(xyz[i], xyz[j]) )
        if r2 > d2: collisions.append((i,j))
    assert len(collisions) == 0

if __name__ == '__main__':
    import pytest
    pytest.main(__file__)
