import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def solve(system, dirichlet, neumann=None):
    '''
    boundary conditions given as dictionaries of { value : mask }
    the length of the Dirichlet masks should be of network.order,
    while Neumann masks are the indices of the tail and head of any specified
    edge gradient.

    example:
    >>> dirichlet = { 1 : x == x.min() }
    >>> neumann = { 5 : network.cut(x == x.max(), network.indexes).T }
    '''
    A, b = build(system, dirichlet, neumann, fast=True)
    x = spsolve(A, b).round(5)
    return x

def build(system, dirichlet, neumann=None, fast=False):
    if neumann is None: neumann = {}
    n_conditions_imposed = count_conditions(system, dirichlet, neumann)

    free = n_conditions_imposed == 0
    D = sparse.eye(system.shape[0]).tocsr()
    elements_of_A, elements_of_b = [system[free]], [np.zeros(free.sum())]
    reindex = free.nonzero()

    for value, mask in dirichlet.items():
        if not any(mask):
            raise Exception("BC value <{}> has no domain.".format(value))
        elements_of_A.append( D[mask.nonzero()] )
        elements_of_b.append( value * np.ones(mask.sum()) )
        reindex += mask.nonzero()

    for v, (t, h) in neumann.items():
        if not any(t):
            raise Exception("BC value <{}> has no domain.".format(v))
        count = np.bincount(t)[np.unique(t)]
        elements_of_A.append( system[np.unique(t)] )
        elements_of_b.append( np.true_divide(v, count.sum()) * count )
        reindex += t.nonzero()

    reindex = np.hstack(reindex).astype('int32')

    if fast:
        A = sparse.vstack(elements_of_A)
        b = np.hstack(elements_of_b)
        return A, b

    # reorders the system to preserve original structure
    A = sparse.vstack(elements_of_A, format='csc')
    A.indices = reindex[A.indices]
    b = np.hstack(elements_of_b)
    b[reindex] = b.copy()
    return A, b

def count_conditions(system, dirichlet, neumann):
    n_conditions_imposed = sum(dirichlet.values())

    _, label = sparse.csgraph.connected_components(system)
    isolated = set(label[n_conditions_imposed==0]) - set(label[n_conditions_imposed>0])
    targets = np.in1d(label, list(isolated))
    if any(targets):
        dirichlet[0] = dirichlet.get(0, np.zeros_like(label)) | targets
    n_conditions_imposed[targets] += 1

    target_idxs = targets.nonzero()[0]
    for v, (tail_ids, head_ids) in neumann.items():
        zeroed_out = np.in1d(tail_ids, target_idxs)
        neumann[v] = (tail_ids[~zeroed_out], head_ids[~zeroed_out])
        n_conditions_imposed[neumann[v][0]] += 1

    if any(n_conditions_imposed > 1):
        raise Exception("Overlapping BCs")
    return n_conditions_imposed