from __future__ import division, absolute_import

import os
import itertools
import random
import traceback
import warnings

import numpy as np
from scipy import spatial, sparse

from .misc import laplacian
from .algorithms import poisson_disk_sampling
from .geometry import cylinders, intersecting
from .graphics import Scene
from . import utils

''' Subclassing rules

    A "network" as defined by MiniPNM is essentially a graph and a coordinate
    array coupled together.

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

    When in doubt: http://en.wikipedia.org/wiki/Glossary_of_graph_theory
'''


class Network(dict):
    filename = None

    @property
    def coords(self):
        return np.vstack([self['x'], self['y'], self['z']])

    points = utils.property_from(['x','y','z'])
    pairs = utils.property_from(['tails','heads'], dtype=int, default=[])

    @property
    def midpoints(self):
        tails, heads = self.points[self.pairs.T]
        return tails + (heads - tails)/2

    @property
    def spans(self):
        tails, heads = self.points[self.pairs.T]
        return heads - tails

    @property
    def lengths(self):
        return np.linalg.norm(self.spans, axis=1).astype('float16')

    @property
    def adjacency_matrix(self):
        tails, heads = self.pairs.T
        ijk = np.ones_like(tails), (tails, heads)
        return sparse.coo_matrix(ijk, shape=(self.order, self.order))

    @property
    def laplacian(self):
        return laplacian(self.adjacency_matrix)
    
    @property
    def labels(self):
        return sparse.csgraph.connected_components(self.adjacency_matrix)[1]

    @property
    def clusters(self):
        return sparse.csgraph.connected_components(self.adjacency_matrix)[0]

    @property
    def order(self):
        return len(self.points)

    @property
    def size(self):
        return len(self.pairs)

    @property
    def dims(self):
        '''
        the `or 1` hack is somewhat ugly, but is a reasonable way to handle
        unit-thin networks. 
        '''
        w = self['x'].max() - self['x'].min() or 1
        h = self['y'].max() - self['y'].min() or 1
        t = self['z'].max() - self['z'].min() or 1
        return np.array([w, h, t])

    @property
    def centroid(self):
        x,y,z = self.coords
        w,h,t = self.dims
        return np.array([w/2.+x.min(), h/2.+y.min(), t/2.+z.min()])

    @property
    def indexes(self):
        return np.arange(self.order)

    def boundary(self):
        all_points = self.indexes
        boundary_points = spatial.ConvexHull(Delaunay.drop_coplanar(self.points)).vertices
        return np.in1d(all_points, boundary_points).astype(bool)

    def save(self, filename):
        minipnm.save_vtp(self, filename)
        self.filename = filename

    def render(self, values=None, *args, **kwargs):
        try:
            # to load as if given a key
            values = self[values]
        except KeyError:
            # show error, but plot anyway (fail gracefully?)
            if values:
                traceback.print_exc()
                values = None
        except TypeError:
            # probably an array, but make sure it fits!
            assert np.array(values).shape[-1] == self.order
        finally:
            scene = Scene()
            scene.add_wires(self.points, self.pairs, values, **kwargs)
            scene.play()

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
        new.pairs = np.vstack([self.pairs, other.pairs+self.order])

        # merge the rest
        for key in set(self.keys()+other.keys())-{'x','y','z','heads','tails'}:
            values_self = self.get(key, -np.ones(self.order))
            values_other = other.get(key, -np.ones(other.order))
            new[key] = np.hstack([values_self, values_other])

        return new

    def cut(self, mask, values=None, bijective=False):
        '''
        returns id of throats where the the tail is masked and the head is not.
        (ie: True for sources).

        for convenience, if a value array is given, the corresponding values
        are returned instead of indices 

        the bijective condition, if enabled, drops any edges that are not
        one-to-one
        '''
        imask = self.indexes[np.array(mask).nonzero()]
        heads, tails = self.pairs.T
        pair_mask = np.in1d(heads, imask) & ~np.in1d(tails, imask)

        if values is None:
            return pair_mask.nonzero()[0] # 1 dimension only
        else:
            tails, heads = values[self.pairs[pair_mask]].T
            if not bijective:
                return tails, heads
            valid = (np.bincount(tails)[tails]==1) & (np.bincount(heads)[heads]==1)
            return tails[valid], heads[valid]

    def prune(self, inaccessible, remove_pores=False):
        accessible = self.indexes[~inaccessible.flatten()]
        good_heads = np.in1d(self['heads'], accessible)
        good_tails = np.in1d(self['tails'], accessible)
        if len(self.pairs) > 0:
            self.pairs = self.pairs[good_heads & good_tails]
        if not remove_pores:
            return

        # remove the unwanted pores
        if len(self.points) > 0:
            self.points = self.points[accessible]
        # now we need to shift throat indexes accordingly
        if len(self.pairs) > 0:
            hs, ts = self.pairs.T
            mapping = np.zeros(inaccessible.size, dtype=int)
            mapping[accessible] = self.indexes
            self.pairs = np.vstack([mapping[hs], mapping[ts]]).T
        for key, array in self.data():
            if array.size == inaccessible.size:
                self[key] = array[accessible]
            else:
                warnings.warn("{} entry mismatch- not ported".format(key))

    def copy(self):
        clone = self.__class__.__new__(self.__class__)
        clone.update(self)
        return clone

    def split(self, mask):
        subnetwork_1 = self - mask
        subnetwork_2 = self - ~mask
        return subnetwork_1, subnetwork_2

    def data(self):
        '''
        data is a mask on dict.items that returns non-essential stored arrays
        '''
        for key, value in self.items():
            if key not in ('x','y','z','heads','tails'):
                yield key, value

    def __repr__(self):
        return self.__class__.__name__+str(self.size)

    def __str__(self):
        entries = [self.__class__.__name__]
        for key, value in sorted(self.data()):
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
        new = self.copy()
        new.prune(inaccessible, remove_pores=True)
        return new


class Cubic(Network):

    @classmethod
    def empty(cls, dims, shape=None):
        arr = np.zeros(dims)
        return cls(arr, shape if shape is not None else arr.shape)

    def __init__(self, ndarray, dims=[1,1,1]):
        ndarray = np.atleast_3d(ndarray)
        dims = tuple(dims) + (1,) * (3 - len(dims))

        self['source'] = np.array(ndarray.flat)

        points_rel = np.array(
            [idx for idx,val in np.ndenumerate(ndarray)]).astype(float)
        points_abs = points_rel * dims / np.array(np.subtract(ndarray.shape,1)).clip(1, np.inf)
        self.points = points_abs

        I = np.arange(ndarray.size).reshape(ndarray.shape)
        tails, heads = [], []
        for T,H in [
            (I[:,:,:-1], I[:,:,1:]),
            (I[:,:-1], I[:,1:]),
            (I[:-1], I[1:]),
            ]:
            tails.extend(T.flat)
            tails.extend(H.flat)
            heads.extend(H.flat)
            heads.extend(T.flat)
        self.pairs = np.vstack([tails, heads]).T

    @property
    def resolution(self):
        return np.array([len(set(d)) for d in self.coords])

    def asarray(self, values=None):
        _ndarray = np.zeros(self.resolution)
        rel_coords = np.true_divide(self.points, self.dims)*(self.resolution-1)
        rel_coords = np.rint(rel_coords).astype(int) # absolutely bizarre bug

        actual_indexes = np.ravel_multi_index(rel_coords.T, self.resolution)
        if values is None:
            values = self['source']
        _ndarray.flat[actual_indexes] = values.ravel()
        return _ndarray.squeeze()


class Delaunay(Network):

    @classmethod
    def random(cls, npoints):
        points = np.random.rand(npoints,3)
        return cls(points)

    def __init__(self, points, mask=None):
        self.pairs = self.edges_from_points(points)
        self.points = np.atleast_2d(points)

    @staticmethod
    def edges_from_points(points, mask=None, directed=True):
        triangulation = spatial.Delaunay(Delaunay.drop_coplanar(points))
        edges = set()
        for unindexed in triangulation.vertices:
            indexed = mask[unindexed] if mask else unindexed
            cyclable = itertools.cycle(unindexed)
            next(cyclable)
            for edge in zip(unindexed, cyclable):
                edge = tuple(sorted(edge))
                if edge not in edges:
                    edges.add(edge)

        one_way = np.array(list(edges))
        if directed:
            other_way = np.fliplr(one_way)
            return np.vstack([one_way, other_way])
        return one_way

    @staticmethod
    def drop_coplanar(points):
        not_coplanar = np.diff(points, axis=0).sum(axis=0).nonzero()
        return points.T[not_coplanar].T


class PackedSpheres(Network):

    def __init__(self, centers, radii, tpt=0):
        '''
        tpt = throat proximity threshold
        '''
        self.points = centers
        self['radii'] = radii

        pairs = []
        for ia, ib in itertools.combinations(range(len(centers)), 2):
            distance = np.linalg.norm(centers[ib]-centers[ia])
            radsum = radii[ia]+radii[ib]
            if (distance - radsum) < tpt:
                pairs.append([ia,ib])

        if pairs:
            self.pairs = np.array(pairs)


class Bridson(Network):
    '''
    A tightly packed network of spheres and cylinders following a given
    distribution

    Poisson disc sampling to generate points,
    Delaunay to generate throats,
    Some basic transformations to ensure geometric coherence,
    '''
    def __init__(self, bbox=[10,10,10], pdf=(1 for _ in itertools.count())):
        points, radii = poisson_disk_sampling(bbox, pdf)
        self.points, self['sphere_radii'] = np.array(points), np.array(radii)
        self.pairs = Delaunay.edges_from_points(self.points)
        self['cylinder_radii'] = self['sphere_radii'][self.pairs].min(axis=1)/4.
        self.spans, self.midpoints = cylinders(self.points, self['sphere_radii'], self.pairs)

        for center, radius in zip(self.points, self['sphere_radii']):
            safe = ~intersecting(center, radius,
                                 self.spans, self.midpoints, self['cylinder_radii'])
            
            # deletes and stuff
            self.pairs = self.pairs[safe]
            self.midpoints = self.midpoints[safe]
            self.spans = self.spans[safe]
            self['cylinder_radii'] = self['cylinder_radii'][safe]

    spans = utils.property_from(['cylinder_length_x','cylinder_length_y','cylinder_length_z'])
    midpoints = utils.property_from(['cylinder_center_x','cylinder_center_y','cylinder_center_z'])

    def render(self, saturation_history=None):
        scene = Scene()
        scene.add_spheres(self.points, self['sphere_radii'], color=(1,1,1), alpha=0.4)
        scene.add_tubes(self.midpoints, self.spans, self['cylinder_radii'])
        if saturation_history is not None:
            scene.add_spheres(self.points, self['sphere_radii']*saturation_history*0.99, color=(0,0,1))
        scene.play()


class Voronoi(Network):
    '''
    Network based on Voronoi tessellation, with non-uniform pore geometry

    Storage involves an additional set of points (vx, vy, vz), and an array of
    indexes (vi) coupled with an array of numbers of points (vn) that allows us
    to determine which intersecting points belong to each point.

    Throats are determined by noting that any two points which share 3+ indexes
    are connected by a surface
    '''
    def __init__(self):
        raise NotImplementedError()