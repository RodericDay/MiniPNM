from __future__ import division, absolute_import
import os
import numpy as np
from scipy import spatial
import minipnm

''' Subclassing rules

    __init__ must *somehow* append coordinate (x,y,z) and connectivity (heads & tails)
    arrays to the network. that's its role. you can add many other attributes if
    you so wish, but you MUST provide those two. 

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

    @property
    def pairs(self):
        ''' returns every 2-dimensional array representing an
            edge or connection between two points
        '''
        return np.vstack([self['heads'], self['tails']]).T

    @property
    def size(self):
        return len(self.points), len(self.pairs)

    @property
    def dims(self):
        w = self['x'].max() - self['x'].min()
        h = self['y'].max() - self['y'].min()
        t = self['z'].max() - self['z'].min()
        return np.array([w, h, t])

    @property
    def centroid(self):
        x,y,z = self.coords
        w,h,t = self.dims
        return np.array([w/2.+x.min(), h/2.+y.min(), t/2.+z.min()])

    def boundary(self):
        all_points = np.arange(self.size[0])
        boundary_points = spatial.ConvexHull(self.points).vertices
        return np.in1d(all_points, boundary_points).astype(bool)

    def save(self, filename):
        minipnm.save_vtp(self, filename)
        self.filename = filename

    def preview(self, values=[]):
        minipnm.preview(self, values)

    def merge(self, other, axis=2, spacing=0, centering=True, stitch=True):
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
        offset = other.coords[axis].min() \
                - self.coords[axis].max() - spacing
        shifted_points.T[axis] -= offset
        new['x'], new['y'], new['z'] = np.vstack([self.points, shifted_points]).T

        # push the connectivity array by the number of already existing vertices
        Va, Ea = self.size
        new['heads'], new['tails'] = np.vstack([self.pairs,
                                                other.pairs+Va
                                                ]).T

        # merge the rest
        for key in set(self.keys()+other.keys()) - {'x','y','z','heads','tails'}:
            values_self = self.get(key, np.zeros(self.size[0]))
            values_other = other.get(key, np.zeros(other.size[0]))
            new[key] = np.hstack([values_self, values_other])

        return new

    def cut(self, head_mask, tail_mask):
        return np.array([i for i, (hix, tix) in enumerate(
            zip(self['heads'], self['tails']))
            if head_mask[hix] and tail_mask[tix]])

    def prune(self, inaccessible, remove_pores=True):
        new = self.copy()

        accessible = np.arange(new.size[0])[~inaccessible]
        good_heads = np.in1d(self['heads'], accessible)
        good_tails = np.in1d(self['tails'], accessible)
        new['heads'] = self['heads'][good_heads & good_tails]
        new['tails'] = self['tails'][good_heads & good_tails]
        if not remove_pores:
            return new

        # now we need to update any other values to the new indeces
        translate = dict(zip(accessible, np.arange(accessible.size)))
        new['x'] = new['x'][accessible]
        new['y'] = new['y'][accessible]
        new['z'] = new['z'][accessible]
        new['heads'] = np.array(map(translate.get, new['heads']))
        new['tails'] = np.array(map(translate.get, new['tails']))
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
        return '\n\t'.join(entries)

    def __add__(self, other):
        return self.merge(other)

    def __sub__(self, inaccessible):
        return self.prune(inaccessible)


class Cubic(Network):

    def __init__(self, ndarray, dims=[1,1,1]):
        self['intensity'] = ndarray.ravel()

        rel_coords = np.array(
            [idx for idx,val in np.ndenumerate(ndarray)]).astype(float)
        abs_coords = rel_coords * dims / np.array(np.subtract(ndarray.shape,1)).clip(1, np.inf)
        self['x'], self['y'], self['z'] = abs_coords.T

        I = np.arange(ndarray.size).reshape(ndarray.shape)
        heads, tails = [], []
        for A,B in [
            (I[:,:,:-1], I[:,:,1:]),
            (I[:,:-1], I[:,1:]),
            (I[:-1], I[1:]),
            ]:
            hs, ts = np.vstack([A.flat, B.flat])
            heads.extend(hs)
            tails.extend(ts)

        self['heads'] = np.array(heads)
        self['tails'] = np.array(tails)


class Delaunay(Network):

    def __init__(self, points, mask=None):
        self['x'], self['y'], self['z'] = points.T
        self['heads'], self['tails'] = self.edges_from_points(points).T

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