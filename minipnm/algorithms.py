from collections import Counter
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import spsolve
import warnings

def solve_bvp(laplacian, dirichlet, neumann={}):
    '''
    boundary conditions given as dictionaries of { value : mask }
    the length of the Dirichlet masks should be of network.order,
    while Neumann masks are the indices of the tail and head of any specified
    edge gradient.

    example:
    >>> dirichlet = { 1 : x == x.min() }
    >>> neumann = { 5 : network.cut(x == x.max(), network.indexes).T }
    '''
    n_conditions_imposed = sum(dirichlet.values())
    for source_like, sink_like in neumann.values():\
        # this basically replaces each value by its count
        # ie. [a a b a c b] = [3 3 2 3 1 2]
        count = np.bincount(source_like)[source_like]
        n_conditions_imposed[source_like] += count
    if any(n_conditions_imposed > 1):
        raise Exception("Overlapping BCs")

    _, membership = sparse.csgraph.connected_components(laplacian)
    isolated = set(membership[n_conditions_imposed==0]) - set(membership[n_conditions_imposed>0])
    targets = np.in1d(membership, list(isolated))
    dirichlet[0] = dirichlet.get(0, np.zeros_like(membership)) | targets
    n_conditions_imposed[targets] += 1

    free = n_conditions_imposed == 0
    D = sparse.eye(laplacian.shape[0]).tocsr()
    elements_of_A, elements_of_b = [laplacian[free]], [np.zeros(free.sum())]

    for value, mask in dirichlet.items():
        elements_of_A.append( D[mask.nonzero()] )
        elements_of_b.append( value * np.ones(mask.sum()) )

    for v, (t, h) in neumann.items():
        if not any(t):
            raise Exception("BC value <{}> has no domain.".format(v))
        elements_of_A.append( (D[t] - D[h]).todense() )
        elements_of_b.append( np.true_divide(v, t.size) * np.ones_like(t) )

    A = sparse.vstack(elements_of_A).tocsr()
    b = np.hstack(elements_of_b)
    x = spsolve(A, b).round(5)
    return x

def percolation(network, sources, thresholds, condition_range,
                base=0.5, rate=1.3):
    network = network.copy()
    # this takes advantage of the auto-scaling of network properties
    network['indexes'] = network.indexes
    network['sources'] = sources
    network['thresholds'] = thresholds

    saturation = np.empty([len(condition_range), network.order])
    for i, condition in enumerate(np.atleast_1d(condition_range)):

        # create placeholder, then replace where applicable
        saturation[i] = np.zeros(network.order)

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

    return np.nan_to_num(saturation)

def invasion(network, sources, thresholds, sinks=None):
    network = network.copy()

    # assume throat access is bound by minimal pore access
    throat_thresholds = thresholds[network.pairs].max(axis=1)

    # Assign a list of pores that are initially filled with liquid water
    saturation = [sources]

    # Repeat until percolation or predefined stopping point
    while True:
        # Identify interfacial throats (between unsaturated and saturated pores)
        interfacial_throats = network.cut(saturation[-1])

        # Check for breakthrough conditions
        if sinks is not None and any(saturation[-1][sinks]): break
        elif len(interfacial_throats) == 0: break

        # Identify the interfacial throat, thmin, with lowest entry pressure
        entry_pressures = throat_thresholds[interfacial_throats]
        th_min = interfacial_throats[entry_pressures.argmin()]

        # Invade any air-filled pore adjacent to thmin with liquid water
        new_saturation = saturation[-1].copy()
        new_saturation[network.pairs[th_min]] = 1
        saturation.append(new_saturation)

    return np.vstack(saturation)

def volumetric(network, sources, capacities, sinks=None, f=1, resolution=100):
    # start with an empty network
    volumes = [np.zeros_like(capacities)]

    bucket_size = sum(capacities)/resolution
    bucket = bucket_size
    fvpp = volumes[-1] # fluid volume per pore
    # exit condition: full network
    while (capacities-volumes[-1]).sum() > 0:

        # determine relevant indexes
        source_like = np.true_divide(fvpp,capacities)>=f # full enough to share
        boundary = np.unique(network.cut(source_like, network.indexes))
        i = sources | source_like | np.in1d(network.indexes, boundary)

        vvpp = (capacities-fvpp)[i] # void volume per pore
        # exit condition: nowhere to go
        if vvpp.sum() == 0:
            volumes.append(fvpp)
            break

        rvvpp = vvpp / vvpp.sum() # relative vvpp

        temp = fvpp.copy()
        temp[i] += bucket * rvvpp

        # handle possible overflow
        fvpp = np.clip(temp, a_min=0, a_max=capacities)
        bucket = (temp - fvpp).sum()

        if bucket == 0:
            volumes.append(fvpp)
            bucket = bucket_size

    return volumes

class PathNotFound(Exception):
    msg = "{} options exhausted, path not found"
    def __init__(self, exhausted):
        self.exhausted = list(exhausted)
    def __str__(self):
        return self.msg.format(len(self.exhausted))

def shortest_path(cmat, start=None, end=None, heuristic=None):
    '''
    finds the shortest path in a graph, given edge costs. can start at any
    point in [start] and end at any point in [end]
    '''
    if start is None:
        start = (0,)
    if end is None:
        end = (cmat.col.max(),)
    if heuristic is None:
        heuristic = lambda i: 0

    reached = set(start)
    exhausted = set()
    vertex_costs = np.inf * np.ones(cmat.col.max()+1)
    vertex_costs[start] = 0

    while not all(i in exhausted for i in end):
        # exit condition
        if not any(reached):
            if any(i in end for i in exhausted):
                break
            raise PathNotFound(exhausted)

        # source vertex index, where source is the lowest cost vertex available
        estimate = lambda i: vertex_costs[i] + heuristic(i)
        si = lowest_cost_index = min(reached, key=estimate)
        source_cost = vertex_costs[si]

        # examine adjacent paths by index
        for pi in (cmat.row==si).nonzero()[0]: # ensure target not exhausted?
            ni = neighbor_index = cmat.col[pi]
            if ni in exhausted:
                continue
            reached.add(ni)

            travel_cost = cmat.data[pi]
            vertex_costs[ni] = min(vertex_costs[ni], source_cost+travel_cost)

        reached.remove(si)
        exhausted.add(si)

    # reconstruct path
    best_exit = min(end, key=vertex_costs.item)
    path = [best_exit]
    while not any(i in path for i in start):
        parents = cmat.row[cmat.col==path[-1]]
        best_parent = min(parents, key=vertex_costs.item)
        path.append(best_parent)
    path.reverse()

    return path