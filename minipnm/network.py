from __future__ import division, absolute_import

import os
import itertools
import random
import traceback

import numpy as np
from scipy import spatial, sparse

from .graphics import Scene

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

    @property
    def points(self):
        ''' a graph consists of edges and vertices
            in this *network* the points are vertices expressed in terms of the
            coordinates associated with them
        '''
        return np.vstack([self['x'], self['y'], self['z']]).T

    @points.setter
    def points(self, _points):
        _points = np.nan_to_num(np.array(_points))
        self['x'], self['y'], self['z'] = _points.T

    @property
    def pairs(self):
        ''' returns every 2-dimensional array representing an
            edge or connection between two points
        '''
        return np.vstack([self.get('heads', []), self.get('tails', [])]).T

    @pairs.setter
    def pairs(self, _pairs):
        ipairs = np.array(_pairs, dtype=int)
        assert np.allclose(ipairs, _pairs)
        self['heads'], self['tails'] = ipairs.T

    @pairs.deleter
    def pairs(self):
        del self['heads']
        del self['tails']

    @property
    def lengths(self):
        return np.linalg.norm(np.diff(self.points[self.pairs], axis=1), axis=2).astype('float16')

    @property
    def adjacency_matrix(self):
        tails, heads = self.pairs.T
        ijk = np.ones_like(tails), (tails, heads)
        return sparse.coo_matrix(ijk, shape=(self.order, self.order))

    @property
    def laplacian(self):
        '''
        this is the graph theory version of the Laplacian matrix
        '''
        A = self.adjacency_matrix
        D = sparse.diags(A.sum(axis=1).A1, 0)
        return D - A
    
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
            cmap = kwargs.get('cmap', None)
            alpha = kwargs.get('alpha', 1)
            scene.add_wires(self.points, self.pairs, values, alpha=alpha, cmap=cmap)
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

    def prune(self, inaccessible, remove_pores=True):
        new = self.copy()

        accessible = self.indexes[~inaccessible.flatten()]
        good_heads = np.in1d(self['heads'], accessible)
        good_tails = np.in1d(self['tails'], accessible)
        if len(self.pairs) > 0:
            new.pairs = self.pairs[good_heads & good_tails]
        if not remove_pores:
            return new

        # remove the unwanted pores
        if len(new.points) > 0:
            new.points = self.points[accessible]
        # now we need to shift throat indexes accordingly
        if len(new.pairs) > 0:
            hs, ts = new.pairs.T
            mapping = np.zeros(self.order, dtype=int)
            mapping[accessible] = new.indexes
            new.pairs = np.vstack([mapping[hs], mapping[ts]]).T
        for key, array in self.items():
            if array.size == self.order:
                new[key] = array[accessible]

        return new

    def copy(self):
        clone = self.__class__.__new__(self.__class__)
        clone.update(self)
        return clone

    def split(self, mask):
        subnetwork_1 = self - mask
        subnetwork_2 = self - ~mask
        return subnetwork_1, subnetwork_2

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
        if values==None:
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
    def edges_from_points(points, mask=None):
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
        other_way = np.fliplr(one_way)
        return np.vstack([one_way, other_way])

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
    Poisson sampling algorithm, to generate networks that respect a
    given distribution
    '''
    n_iter = 100
    p_max = 10000

    def __init__(self, pdf, dims):

        self._points = [(0,0,0)]
        self._radii = [next(pdf)]
        self._available = [0]
        self._dims = dims

        while self._available and len(self._points) <= self.p_max:
            self.add_point(next(pdf))

        self.points = self._points
        del self._points
        self['radii'] = np.array(self._radii)
        del self._radii
        del self._available
        del self._dims

    def add_point(self, r):
        idx = random.choice(self._available)

        xi, yi, zi = self._points[idx]
        min_dist = self._radii[idx] + r
        inner_r = min_dist
        outer_r = min_dist*2

        for j in range(self.n_iter):
            aj = random.random() * 2 * np.pi
            bj = random.random() * np.pi
            rj = ( random.random()*(outer_r**3 - inner_r**3) + inner_r**3 )**(1./3.)

            xj = rj * np.cos(aj) * np.sin(bj) + xi
            yj = rj * np.cos(aj) * np.cos(bj) + yi
            zj = rj * np.sin(aj) + zi

            # check if the point center is within bounds
            def outside():
                return any(abs(c) > d/2. for c,d in zip([xj,yj,zj], self._dims))

            # make sure that the distance between any two centers is larger
            # than sum of corresponding radii
            def near():
                for (xk, yk, zk), rk in zip(self._points, self._radii):
                    radii_sum = r + rk
                    dist_mhtn = [abs(xj-xk), abs(yj-yk), abs(zj-zk)]
                    if any(dm > radii_sum for dm in dist_mhtn):
                        yield False
                    else:
                        dist_btwn = np.linalg.norm(dist_mhtn)
                        yield radii_sum > dist_btwn

            # if it isn't, bail!
            if outside() or any(near()):
                continue

            # if we got here the point is valid!
            self._available.append( len(self._points) )
            self._points.append( (xj, yj, zj) )
            self._radii.append( r )
            return

        # if we got here, it's because no new points were able to be generated
        # the point is probably too crowded to have new neighbors, so we should
        # stop considering it
        self._available.remove(idx)
        if not any(self._available):
            return
        self.add_point(r)

    def render(self):
        scene = Scene()
        scene.add_spheres(self.points, self['radii'], color=(0,0,1))
        scene.add_tubes(self.points, self.pairs)
        scene.play()