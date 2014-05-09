from __future__ import division, absolute_import
import os
import numpy as np
from scipy import spatial, sparse
import minipnm

''' Subclassing rules

    __init__ must *somehow* append coordinate (x,y,z) and connectivity (heads & tails)
    arrays to the network. that's its role. you can add many other attributes if
    you so wish, but you MUST provide those two. the `points` and 
    `pairs` property setters are offered for convenience

    The other main rule is that aside from methods and properties, these objects
    should essentially be *just* dictionaries. One way to think about this is that
    the answer to the question "will I miss any data if I just write down the
    results of network.items?" should be NO. 

    Convenience methods that automate some work (ie. import an image from a file)
    can be found in the `misc` module
'''


class Network(dict):
    filename = None

    @property
    def coords(self):
        return np.vstack([self['x'], self['y'], self['z']])

    @property
    def points(self):
        ''' a graph consists of edges and vertices
            in this *network* the points are vertices expressed in terms of the
            coordinates associated with them
        '''
        return np.vstack([self['x'], self['y'], self['z']]).T

    @points.setter
    def points(self, _points):
        self['x'], self['y'], self['z'] = _points.T

    @property
    def pairs(self):
        ''' returns every 2-dimensional array representing an
            edge or connection between two points
        '''
        return np.vstack([self['heads'], self['tails']]).T

    @pairs.setter
    def pairs(self, _pairs):
        self['heads'], self['tails'] = _pairs.T

    @property
    def connectivity_matrix(self):
        heads, tails = self.pairs.T
        fwds = np.hstack([heads, tails])
        bkds = np.hstack([tails, heads])
        return sparse.coo_matrix((np.ones_like(fwds), (fwds, bkds)),
                                 shape=(self.size[0], self.size[0]))
    
    @property
    def labels(self):
        return sparse.csgraph.connected_components(self.connectivity_matrix)[1]

    @property
    def size(self):
        return len(self.points), len(self.pairs)

    @property
    def dims(self):
        w = self['x'].max() - self['x'].min()
        h = self['y'].max() - self['y'].min()
        t = self['z'].max() - self['z'].min() or 1
        return np.array([w, h, t])

    @property
    def centroid(self):
        x,y,z = self.coords
        w,h,t = self.dims
        return np.array([w/2.+x.min(), h/2.+y.min(), t/2.+z.min()])

    @property
    def indexes(self):
        return np.arange(self.size[0])

    def boundary(self):
        all_points = self.indexes
        boundary_points = spatial.ConvexHull(self.points).vertices
        return np.in1d(all_points, boundary_points).astype(bool)

    def save(self, filename):
        minipnm.save_vtp(self, filename)
        self.filename = filename

    def render(self, *args, **kwargs):
        renderer = minipnm.render(self, *args, **kwargs)
        renderer.Start()

    def merge(self, other, axis=2, spacing=None, centering=False, stitch=False):
        new = Network()

        # alignment along a common centroid
        if centering:
            center_distance = other.centroid - self.centroid
            center_distance[axis] = 0 # we take care of this one differently
            shifted_points = other.points - center_distance
        else:
            shifted_points = other.points

        # the distance between the max for base and min for other should
        # equal to spacing. rearranging, it gives us the required offset
        if spacing is not None:
            offset = other.coords[axis].min() \
                    - self.coords[axis].max() - spacing
            shifted_points.T[axis] -= offset
        new.points = np.vstack([self.points, shifted_points])

        # push the connectivity array by the number of already existing vertices
        Va, Ea = self.size
        new.pairs = np.vstack([self.pairs, other.pairs+Va])

        # merge the rest
        for key in set(self.keys()+other.keys())-{'x','y','z','heads','tails'}:
            values_self = self.get(key, -np.ones(self.size[0]))
            values_other = other.get(key, -np.ones(other.size[0]))
            new[key] = np.hstack([values_self, values_other])

        return new

    def cut(self, mask):
        ''' returns id of throats where the head and the tail are on opposite
            sides of the mask
        '''
        imask = np.array(mask.nonzero())
        heads, tails = self.pairs.T
        return (np.in1d(heads, imask) != np.in1d(tails, imask)).nonzero()

    def flux(self, data, mask):
        cut_ids = self.cut(mask)
        vertex_ids = self.pairs[cut_ids].flatten()
        data_pairs = data[vertex_ids].reshape(vertex_ids.size//2, 2)
        data_delta = np.abs(np.diff(data_pairs))
        return data_delta

    def prune(self, inaccessible, remove_pores=True):
        new = self.copy()

        accessible = self.indexes[~inaccessible.flatten()]
        good_heads = np.in1d(self['heads'], accessible)
        good_tails = np.in1d(self['tails'], accessible)
        if len(self.pairs) > 0:
            new.pairs = self.pairs[good_heads & good_tails]
        if not remove_pores:
            return new

        # now we need to update any other values to the new indexes
        if len(new.points):
            new.points = self.points[accessible]
        mapping = dict(zip(accessible, new.indexes))
        translate = np.vectorize(mapping.__getitem__)
        if len(new.pairs) > 0:
            new.pairs = translate(new.pairs)
        for key, array in self.items():
            if array.size == self.size[0]:
                new[key] = array[accessible]

        return new

    def copy(self):
        clone = self.__class__.__new__(self.__class__)
        clone.update(self)
        return clone

    def __repr__(self):
        return self.__class__.__name__+str(self.size)

    def __str__(self):
        entries = [self.__class__.__name__]
        for key, value in sorted(self.items()):
            entries.append('{:<15}: {:<10}: {:<15}'.format(
                    key,
                    value.dtype,
                    value.shape,))
        return '<'+'\n\t'.join(entries)+'\nSize: {}>'.format(self.size)

    def __add__(self, other):
        return self.merge(other, spacing=0, centering=True)

    def __or__(self, other):
        return self.merge(other)

    def __sub__(self, inaccessible):
        return self.prune(inaccessible)


class Cubic(Network):

    @classmethod
    def empty(cls, dims, shape=None):
        arr = np.zeros(dims)
        return cls(arr, shape or arr.shape)

    def __init__(self, ndarray, dims=[1,1,1]):
        ndarray = np.atleast_3d(ndarray)
        dims = tuple(dims) + (1,) * (3 - len(dims))

        self['intensity'] = ndarray.ravel()

        points_rel = np.array(
            [idx for idx,val in np.ndenumerate(ndarray)]).astype(float)
        points_abs = points_rel * dims / np.array(np.subtract(ndarray.shape,1)).clip(1, np.inf)
        self.points = points_abs

        I = np.arange(ndarray.size).reshape(ndarray.shape)
        heads, tails = [], []
        for A,B in [
            (I[:,:,:-1], I[:,:,1:]),
            (I[:,:-1], I[:,1:]),
            (I[:-1], I[1:]),
            ]:
            heads.extend(A.flat)
            tails.extend(B.flat)
        self.pairs = np.vstack([heads, tails]).T

    @property
    def resolution(self):
        return np.array([len(set(d)) for d in self.coords])

    def asarray(self, values=None):
        _ndarray = np.zeros(self.resolution)
        rel_coords = np.true_divide(self.points, self.dims)*(self.resolution-1)
        rel_coords = np.rint(rel_coords).astype(int) # absolutely bizarre bug

        actual_indexes = np.ravel_multi_index(rel_coords.T, self.resolution)
        if values==None:
            values = self['intensity']
        _ndarray.flat[actual_indexes] = values.ravel()
        return _ndarray


class Delaunay(Network):

    @classmethod
    def random(cls, npoints):
        points = np.random.rand(npoints,3)
        return cls(points)

    def __init__(self, points, mask=None):
        self.points = points
        self.pairs = self.edges_from_points(points)

    @staticmethod
    def edges_from_points(points, mask=None):
        triangulation = spatial.Delaunay(points)
        edges = set()
        for a,b,c,d in triangulation.vertices:
            if mask is not None:
                a,b,c,d = (mask[i] for i in [a,b,c,d])
            for edge in [(a,b),(b,c),(c,d),(d,a)]:
                edge = tuple(sorted(edge))
                if edge not in edges:
                    edges.add(edge)

        return np.array(list(edges))
