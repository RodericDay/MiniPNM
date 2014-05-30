import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import spsolve

def linear_solve(network, dbcs):
    # the matrix to be solved will involve a stack of ICs and some balances
    # take as fixed the points where dbcs is defined
    fixed = dbcs!=0

    # if there are unconnected islands, they need to be fixed at zero too
    labels = network.labels
    island_labels = set(labels[~fixed]) - set(labels[fixed])
    fixed = fixed | np.in1d(labels, list(island_labels))

    # this array shifts the ICs around in the correct way
    mapping = np.hstack(fixed.nonzero()+(~fixed).nonzero())

    # we build a diagonal matrix for dbcs
    IC= sparse.diags(np.ones_like(dbcs), 0).tocsr()
    # a connectivity matrix representing sources
    C = network.connectivity_matrix
    # and a diagonal matrix representing sinks, as sum of sources
    D = sparse.diags(C.sum(axis=1).A1, 0).tocsr() # A1 is matrix -> array
    # the system matrix is a stack of the relevant rows of each
    A = sparse.vstack([(IC)[fixed], (C - D)[~fixed]]).tocsr()
    x = spsolve(A, dbcs[mapping])

    assert( np.allclose(x[fixed], dbcs[fixed]) )

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
    real = volumes[-1]
    # exit condition: full network
    while (capacities-volumes[-1]).sum() > 0:

        # determine relevant indexes
        source_like = np.true_divide(real,capacities)>=f # full enough to share
        boundary = np.unique(network.cut(source_like, network.indexes))
        i = sources | source_like | np.in1d(network.indexes, boundary)

        vvpp = (capacities-real)[i] # void volume per pore
        # exit condition: nowhere to go
        if vvpp.sum() == 0:
            volumes.append(real)
            break

        rvvpp = vvpp / vvpp.sum() # relative vvpp

        temp = real.copy()
        temp[i] += bucket * rvvpp

        # handle possible overflow
        real = np.clip(temp, a_min=0, a_max=capacities)
        bucket = (temp - real).sum()

        if bucket == 0:
            volumes.append(real)
            bucket = bucket_size

    return volumes