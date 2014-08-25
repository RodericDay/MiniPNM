from __future__ import absolute_import, division
import bisect
import numpy as np
from scipy import misc, ndimage, sparse

'''
misc houses scientific tools and helpers
'''
class distributions:
    uniform = (1 for _ in iter(int,1))

def mid(array):
    center = (array.max() - array.min())/2.
    return array[bisect.bisect_left(array, center)]

def normalize(array):
    array = np.atleast_1d(array)
    array = np.subtract(array, array.min())
    array = np.true_divide(array, array.max())
    return array

def laplacian(A):
    '''
    this is the graph theory version of the Laplacian matrix, given
    a value-weighted adjacency matrix A
    '''
    D = sparse.diags(A.sum(axis=1).A1, 0)
    return D - A

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
