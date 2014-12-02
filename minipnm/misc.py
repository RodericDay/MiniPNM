from __future__ import absolute_import, division
import bisect
import numpy as np
from scipy import misc, ndimage, sparse

'''
misc houses scientific tools and helpers
'''
class distributions:
    uniform = (1 for _ in iter(int,1))

def flux(adj, values):
    adj.data = abs(values[adj.col] - values[adj.row])
    return adj.mean(axis=1).A1

def distances_to_neighbors(network):
    output = [[] for i in range(network.order)]
    for (t,h), d in zip(network.pairs, network.lengths):
        output[h].append(d)
        output[t].append(d)
    return output

def filtered_distances(network, _filter, default):
    all_distances = distances_to_neighbors(network)
    distances = np.array([_filter(row) if row else default for row in all_distances])
    return distances.squeeze()

distances_to_nearest_neighbors = lambda network: filtered_distances(network, min, 1E-6)
distances_to_furthest_neighbors = lambda network: filtered_distances(network, max, np.inf)

def match(p1, p2):
    return [np.argmin(np.linalg.norm(p-p2, axis=1)) for p in p1]
