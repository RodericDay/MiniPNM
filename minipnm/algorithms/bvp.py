import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


class System(object):

    def __init__(self, pairs):
        self.ij = np.transpose(pairs)
        self._conductances = np.ones(self.ij.shape[1])
        self.prepare()

    @property
    def conductances(self):
        return self._conductances

    @conductances.setter
    def conductances(self, values):
        self._conductances[:] = values
        if any(self._conductances < 0):
            raise RuntimeError( "Negative conductances" )

        self.prepare()

    def prepare(self):
        ijk = self._conductances, self.ij
        self.C = sparse.coo_matrix(ijk).tocsr()
        self.C.eliminate_zeros()
        _, self.labels = sparse.csgraph.connected_components(self.C)

    def build(self, dirichlet, k=0, s=0, default=0):
        linear = k
        source = s
        constraints = sum(dirichlet.values())

        if constraints.max() > 1:
            raise RuntimeError( "Too many constraints" )

        constrained = constraints.astype(bool)
        b = sum(v * mask for v, mask in dirichlet.items())

        # see prepare
        C = self.C
        island = ~np.in1d( self.labels, self.labels[constrained] )
        b[island] = default
        constrained |= island

        d = np.where(constrained, 1, C.sum(axis=1).A1 + linear)
        D = sparse.diags(d, offsets=0).tocsr()

        A1 = (D-C)[~constrained]
        A2 = (D)[constrained]

        A = sparse.vstack([ A1, A2 ])
        b = np.hstack([ (b-source)[~constrained], b[constrained] ])

        return A, b

    def solve(self, *args, **kwargs):
        A, b = self.build(*args, **kwargs)
        x = spsolve(A, b)
        return x


def test_handling_of_void():
    Z = 10
    G = np.array([list(map(int, line)) for line in '''
    00100
    01010
    00100
    00010
    00000
    '''.strip().split()]).repeat(Z, axis=1).repeat(Z, axis=0)

    cubic = mini.Cubic(G.shape)
    x, y, z = cubic.coords
    t = y[::-1]
    sys = System(cubic.pairs)
    sys.conductances = 1-G.flat[cubic.pairs].max(axis=1)
    v = sys.solve({2:t==t.max()}, s=1, default=0)
    V = cubic.asarray(v)

    plt.subplot().matshow(V)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import minipnm as mini

    test_handling_of_void()
    plt.show()
