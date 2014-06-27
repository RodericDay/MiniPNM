import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
try:
    profile
except:
    profile = lambda x: x


class Matplotnet(dict):
    
    def __init__(self, array):
        self['v'] = array.flatten()
        self['x'], self['y'] = np.vstack(idx for idx, val in np.ndenumerate(array)).T
        I = self.indexes.reshape(array.shape)
        heads, tails = [], []
        for A,B in [ (I[:-1], I[1:]), (I[:,:-1],I[:,1:]) ]:
            # to
            tails.extend(A.flat)
            heads.extend(B.flat)
            # fro
            tails.extend(B.flat)
            heads.extend(A.flat)
        self['t'] = np.array(tails)
        self['h'] = np.array(heads)

    @property
    def order(self):
        return self['x'].size

    @property
    def size(self):
        return self['t'].size

    @property
    def indexes(self):
        return np.arange(self.order)

    @property
    def C(self):
        ijk = np.ones(self.size), (self['t'], self['h'])
        return sparse.coo_matrix(ijk, shape=(self.order, self.order))

    def show(self, values=None):
        if values is None:
            values = self['v']
        shape = np.unique(self['x']).size, np.unique(self['y']).size
        im = values.reshape(shape).T
        masked = np.ma.array(im, mask=np.isnan(im))
        cmap = plt.cm.coolwarm
        cmap.set_bad('g', 0.2)
        plt.pcolormesh(masked, cmap=cmap)
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def __str__(self):
        template = "'{key}': {value.__class__.__name__}({value.size})"
        body = ', '.join(template.format(**locals()) for key,value in self.items())
        return '{{{body}}}'.format(**locals())


@profile
def solve(network, dbcs, nbcs=None):
    '''
    dbcs : Array. Dirichlet boundary conditions. The value of a given vertex.
    nbcs : Array. Clusters of a given value emanate that value @ boundary.
    mode : affects Neumann. If bulk, the value specifies the total. If per,
            the value specifies a fixed flux per connection.
    '''
    '''
    What are we going for here? Consider a 1D network, in A | b form
    [  2  -1   0   0  |  0
      -1   2  -1   0  |  0
       0  -1   2  -1  |  0
       0   0  -1   2  |  0  ]
    
    for Dirichlet boundary conditions we will then replace the relevant rows
    ie. if the node with index 0 has value a:
    [  1   0   0   0  |  a  ] (0)

    for Neumann boundary conditions we have to focus on the connection, and
    ensure that we take into account the resolution of the network
    [  0   0  -1   1  |  b * dx  ] (-1)

    there are two possible types of conditions. we can prescribe what we want
    the total flux to be over a threshold, or we can prescribe what the flux
    through each selected throat should be

    For 2D network, each line will have 5 items (1,1,-4,1,1), and so on
    '''
    # no overlap
    if nbcs is not None:
        assert not ( (dbcs != 0) & (nbcs != 0) ).all()

    # create a pure network
    M = network.C - sparse.diags(network.C.sum(axis=1).A1, 0)
    z = np.zeros(network.order)

    # DIRICHLET
    # rows with dbcs need to be replaced with fixed values
    fd = np.in1d(network.indexes, dbcs.nonzero())
    D = sparse.diags(np.ones(network.order), 0).tocsr()

    # NEUMANN
    if nbcs is not None:
        # relevant throats
        could_be_source = (nbcs == 0).nonzero()
        could_be_sink = (nbcs != 0).nonzero()

        tails, heads = network.C.row, network.C.col
        valid = np.in1d(tails, could_be_source) & np.in1d(heads, could_be_sink)
        
        actual_tails, actual_heads = network.C.row[valid], network.C.col[valid]
        
        # change all of the entries for the 'actual heads' to Neumann form
        n = network.order
        N = sparse.diags(np.ones(n), 0) - sparse.diags(np.ones(n-1), 1)
        k = nbcs

        fn = np.in1d(network.indexes, actual_heads)
    else:
        fn = np.zeros_like(fd)
        k = np.zeros_like(fd)
        N = M

    # ASSEMBLE
    # build the system matrices by stacking
    A = sparse.vstack([D[fd], N[fn], M[~(fd|fn)]]).tocsr()
    b = np.hstack([dbcs[fd], k[fn], z[~(fd|fn)]])
    # solve
    sol = spsolve(A,b)

    # verify boundary conditions fulfilled
    assert (sol==dbcs)[dbcs.nonzero()].all()
    if nbcs is not None:
        assert np.allclose(nbcs[actual_heads], sol[actual_heads]-sol[actual_tails])
        sol[actual_heads] = np.nan
        sol[actual_tails] = np.nan

    return sol



R = np.ones([50,90])
network = Matplotnet(R)

bottom = network['y'] < network['y'].min() + 30
half_slice = (network['x'] == network['x'].max()//2) & (network['y'] >= np.percentile(network['y'],50))

dirichlet = 3*half_slice + 1*bottom
neumann = None
# neumann = 0.01*bottom

network['v'] = solve(network, dirichlet, neumann)
network.show()