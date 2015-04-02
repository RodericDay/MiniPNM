import numpy as np
from .network import Delaunay

'''
Binary operations create a new graph from two initial graphs
'''

def radial_join(voxel_like, sphere_like):
    superimposed = voxel_like | sphere_like
    overjoined = Delaunay(superimposed.points)
    # we only want pairs connecting one and the other
    mask = [True]*voxel_like.order + [False]*sphere_like.order # ugly
    new = overjoined.cut(mask)
    superimposed.pairs = np.vstack([superimposed.pairs, overjoined.pairs[new]])
    return superimposed
