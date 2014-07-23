from collections import Counter
import random
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import spsolve
import warnings





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

def build_bvp(system, dirichlet, neumann=None, fast=False):
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

def solve_bvp(system, dirichlet, neumann=None):
    '''
    boundary conditions given as dictionaries of { value : mask }
    the length of the Dirichlet masks should be of network.order,
    while Neumann masks are the indices of the tail and head of any specified
    edge gradient.

    example:
    >>> dirichlet = { 1 : x == x.min() }
    >>> neumann = { 5 : network.cut(x == x.max(), network.indexes).T }
    '''
    A, b = build_bvp(system, dirichlet, neumann, fast=True)
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





def poisson_disk_sampling(bbox, pdf, n_iter=100, p_max=10000):
    '''
    A 3D version of Robert Bridson's algorithm, perhaps best illustrated by
    Mike Bostock's following D3.js animation:
    http://bl.ocks.org/mbostock/dbb02448b0f93e4c82c3

    Takes in a virtual 'bounding box' and a generator from which to sample,
    and purports to pack the space as tightly and randomly as possible,
    outputting the coordinates and radii of the corresponding circles.
    '''
    # subfunctions that use local namespace
    def outside():
        ''' checks if point center is within bounds '''
        return any(abs(c) > d/2. for c,d in zip([xj,yj,zj], bbox))

    def near():
        ''' checks if distance between two centers larger than radii'''
        for (xk, yk, zk), rk in zip(points, radii):
            radii_sum = r + rk
            # manhattan distance filter for distant points
            dist_mhtn = [abs(xj-xk), abs(yj-yk), abs(zj-zk)]
            if any(dm > radii_sum for dm in dist_mhtn):
                yield False
            else:
                dist_btwn = np.linalg.norm(dist_mhtn)
                yield radii_sum > dist_btwn

    points = [(0,0,0)]
    radii = [next(pdf)]
    available = [0]
    is3d = True if len(bbox)==3 else False
    
    while available and len(points) <= p_max:
        r = float(next(pdf))
        source = i = random.choice(available)
        xi, yi, zi = points[i]
        inner_r = min_dist = radii[i] + r
        outer_r = min_dist * 2


        for j in range(n_iter):
            # try a random point in the sampling space
            aj = random.random() * 2 * np.pi if is3d else 0
            bj = random.random() * np.pi
            rj = ( random.random()*(outer_r**3 - inner_r**3) + inner_r**3 )**(1./3.)

            xj = rj * np.cos(aj) * np.sin(bj) + xi
            yj = rj * np.cos(aj) * np.cos(bj) + yi
            zj = rj * np.sin(aj) + zi

            # bail of checks fail
            if outside() or any(near()):
                continue
    
            # if we got here the point is valid!
            available.append( len(points) )
            points.append( (xj, yj, zj) )
            radii.append( r )
            break
            
        else:
            # somewhat unknown python feature, think of it as "nobreak"
            # if we got here, it's because no new points were able to be
            # generated. this source is probably too crowded to have new
            # neighbors, so we stop considering it
            available.remove(i)

    return points, radii