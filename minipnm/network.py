from __future__ import division, absolute_import

import os
import itertools
import random
import traceback
import warnings

import numpy as np
from scipy import spatial, sparse

from . import utils
from . import geometry
from . import graphics


class Network(dict):
    '''
    A network as defined by MiniPNM is essentially a graph and a coordinate
    array coupled together.

    __init__ must *somehow* append coordinate (x,y,z) and connectivity (tails &
    heads) arrays to the network. Other attributes may be added if required,
    but the aforementioned 5 are mandatory. They are packaged into properties
    for convenience.

    As a general rule, aside from methods and properties, these objects
    should essentially be *just* dictionaries. In order to save the data and
    recover it later, it should suffice to call `network.items()` and write
    the output to a file.

    Lastly, for the sake of communicability, there is a reference of choice for 
    terminology:
    http://en.wikipedia.org/wiki/Glossary_of_graph_theory
    '''
    points = utils.property_from(['x','y','z'])
    pairs = utils.property_from(['tails','heads'], dtype=int, default=[])
    filename = None

    @classmethod
    def load(cls, dict_):
        inst = cls.__new__(cls)
        inst.update(dict_)
        return inst

    @property
    def order(self):
        return len(self.points)

    @property
    def size(self):
        return len(self.pairs)

    @property
    def coords(self):
        return self.points.T

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
        return np.linalg.norm(self.spans, axis=1).astype('float32')

    @property
    def diagonals(self):
        return sparse.diags(np.ones(self.order), 0)

    @property
    def adjacency_matrix(self):
        tails, heads = self.pairs.T
        ijk = np.ones_like(tails), (heads, tails)
        return sparse.coo_matrix(ijk, shape=(self.order, self.order), dtype=float)

    @property
    def labels(self):
        return sparse.csgraph.connected_components(self.adjacency_matrix)[1]

    @property
    def clusters(self):
        return sparse.csgraph.connected_components(self.adjacency_matrix)[0]

    @property
    def bbox(self):
        return self.points.max(axis=0) - self.points.min(axis=0)

    @property
    def centroid(self):
        return self.points.min(axis=0) + self.bbox/2.

    @property
    def indexes(self):
        return np.arange(self.order)

    def system(self, cvalues=1, units=None):
        '''
        Returns a matrix representing a system of equations
        '''
        if units is not None:
            cvalues = cvalues(units)
        A = self.adjacency_matrix.astype(float)
        A.data *= cvalues
        D = self.diagonals.astype(float)
        D.data *= -A.sum(axis=1).A1
        return (A + D) * (1 if units is None else units)

    def boundary(self):
        all_points = self.indexes
        boundary_points = spatial.ConvexHull(geometry.drop_coplanar(self.points)).vertices
        return np.in1d(all_points, boundary_points).astype(bool)

    def save(self, filename):
        minipnm.save_vtp(self, filename)
        self.filename = filename

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

        for key in set(itertools.chain(self.keys(),other.keys())) - \
            {'x','y','z','heads','tails'}:
            values_self = self.get(key, -np.ones(self.order))
            values_other = other.get(key, -np.ones(other.order))
            new[key] = np.hstack([values_self, values_other])

        return new

    def cut(self, mask, values=None, bijective=False, directed=True):
        '''
        returns id of throats where the the tail is masked and the head is not.
        (ie: True for sources).

        for convenience, if a value array is given, the corresponding values
        are returned instead of indices

        the bijective condition, if enabled, drops any edges that are not
        one-to-one

        if directed is set to false, the method will ignore which sides of the
        mask are true and which are false. default is to consider throats
        where the mask is a selection and we want throats reaching out of it
        '''
        imask = self.indexes[np.array(mask).nonzero()]
        heads, tails = self.pairs.T
        if directed:
            pair_mask = np.in1d(heads, imask) & ~np.in1d(tails, imask)
        else:
            pair_mask = np.in1d(heads, imask) == ~np.in1d(tails, imask)

        if values is None:
            return pair_mask
        else:
            tails, heads = values[self.pairs[pair_mask]].T
            if not bijective:
                return tails, heads
            valid = (np.bincount(tails)[tails]==1) & (np.bincount(heads)[heads]==1)
            return tails[valid], heads[valid]

    def prune(self, inaccessible, remove_pores=True):
        '''
        the update calls have some probability of messing things up if the
        network.order somehow ends up being equal to the network.size
        '''
        old_order, old_size, old_keys = self.order, self.size, self.keys()

        accessible = self.indexes[~inaccessible.flatten()]
        good_heads = np.in1d(self['heads'], accessible)
        good_tails = np.in1d(self['tails'], accessible)
        valid = good_heads & good_tails
        if len(self.pairs) > 0:
            self.pairs = self.pairs[valid]
        self.update({key:array[valid] for key,array in self.data() if array.size==old_size})
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
        self.update({key:array[accessible] for key,array in self.data() if array.size==old_order})

        left_out = set(self.keys()) ^ set(old_keys)
        if any(left_out):
            warnings.warn("{}".format(left_out)) # make more verbose

    def copy(self):
        return self.load(self)

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
        return '<'+'\n\t'.join(entries)+\
            '\nOrder: {self.order}, Size: {self.size}>'.format(**locals())

    def __add__(self, other):
        return self.merge(other, spacing=0, centering=True)

    def __or__(self, other):
        return self.merge(other)

    def __sub__(self, inaccessible):
        new = self.copy()
        new.prune(inaccessible, remove_pores=False)
        return new

    def render(self, *args, **kwargs):
        wait = True
        if 'scene' in kwargs:
            scene = kwargs.pop('scene')
        else:
            scene = graphics.Scene()
            wait = False

        scene.add_actors(self.actors(*args, **kwargs))

        if not wait:
            scene.play()

    def actors(self, values=None, **kwargs):
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
            wires = graphics.Wires(self.points, self.pairs, values, **kwargs)
        return [wires]

    def plot(self, *values):
        rotation = itertools.cycle(['Blues', 'hot', 'summer', 'copper', 'rainbow'])

        if 0 not in self.bbox:
            raise NotImplementedError('Only usable by 1D or 2D networks')
        canvas = 'xyz'[self.bbox.tolist().index(0)]
        self[canvas] *= 0

        scene = graphics.Scene()
        self.render(scene=scene)
        for arr, cmap in zip(values, rotation):
            self[canvas][:] = arr
            self.render(scene=scene, cmap=cmap)
        scene.play()

class Cubic(Network):

    @classmethod
    def from_source(cls, im):
        network = cls(im.shape)
        network['source'] = im.ravel()
        return network
    
    def __init__(self, shape, spacing=None, bbox=None):
        arr = np.atleast_3d(np.empty(shape))

        self.points = np.array([i for i,v in np.ndenumerate(arr)], dtype=float)
        if spacing is not None:
            self.points *= spacing
        elif bbox is not None:
            self.points *= bbox / self.points.max(axis=0)

        I = np.arange(arr.size).reshape(arr.shape)
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

    def asarray(self, values):
        spacing = map(np.diff, map(np.unique, self.coords))
        min_spacing = [min(a) if len(a) else 1.0 for a in spacing]
        points = (self.points / min_spacing).astype(int)
        points -= points.min(axis=0)
        bbox = (self.bbox / min_spacing + 1).astype(int)
        actual_indexes = np.ravel_multi_index(points.T, bbox)
        array = np.zeros(bbox)
        array.flat[actual_indexes] = values.ravel()
        return array.squeeze()


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
        noncoplanar = geometry.drop_coplanar(points)
        edges = np.vstack(spatial.Voronoi(noncoplanar).ridge_points)
        if mask:
            edges = mask[edges]
        if directed:
            edges = np.vstack([edges, np.fliplr(edges)])
        return edges


class Radial(Network):
    '''
    Takes in points, sphere radii, and returns a fleshed out network consisting
    of spheres and cylinders.

    If connectivity pairs aren't specified, default is Delaunay tessellation.
    Pruning follows to ensure there are no collisions regardless.
    '''
    spans = utils.property_from(['cylinder_length_x','cylinder_length_y','cylinder_length_z'])
    midpoints = utils.property_from(['cylinder_center_x','cylinder_center_y','cylinder_center_z'])

    def __init__(self, centers, radii, pairs=None, prune=True, f=2):
        self.points = np.array(centers, dtype=float)
        if pairs is None:
            pairs = Delaunay.edges_from_points(self.points)
        self.pairs = np.atleast_2d(pairs)

        self['sphere_radii'] = np.ones(self.order, dtype=float)*radii
        self['cylinder_radii'] = self['sphere_radii'][self.pairs].min(axis=1)/f
        self.spans, self.midpoints = geometry.cylinders(self.points, self['sphere_radii'], self.pairs)

        if prune:
            self.prune_colliding()

    @property
    def bbox(self):
        minima = (self.coords - self['sphere_radii']).min(axis=1)
        maxima = (self.coords + self['sphere_radii']).max(axis=1)
        return maxima - minima

    @property
    def areas(self):
        return self['sphere_radii']**2 * 4. * np.pi

    @property
    def volumes(self):
        return self['sphere_radii']**3 * 4./3. * np.pi

    def prune_colliding(self):
        for center, radius in zip(self.points, self['sphere_radii']):
            safe = ~geometry.intersecting(center, radius,
                        self.spans, self.midpoints, self['cylinder_radii'])

            # deletes and stuff
            self.pairs = self.pairs[safe]
            self.midpoints = self.midpoints[safe]
            self.spans = self.spans[safe]
            self['cylinder_radii'] = self['cylinder_radii'][safe]

    def rasterize(self, resolution=20):
        offset = (self.coords - self['sphere_radii']).min(axis=1)
        scale = np.true_divide(self.bbox.max(), resolution-1)
        dims = np.ceil(self.bbox/scale) + 1
        im = np.zeros(dims)-1

        raster = Cubic.from_source(im)
        raster.points *= scale
        raster.points += offset

        for i, center in enumerate(self.points):
            distance = np.linalg.norm(raster.points-center, axis=1)
            equivalent = distance < self['sphere_radii'][i]
            raster['source'][equivalent] = i

        return raster

    def actors(self, saturation_history=None):
        shells = graphics.Spheres(self.points, self['sphere_radii'], color=(1,1,1), alpha=0.4)
        tubes = graphics.Tubes(self.midpoints, self.spans, self['cylinder_radii'])
        if saturation_history is None:
            return [shells, tubes]

        capacity = 4./3. * np.pi * self['sphere_radii']**3
        fill_radii = (capacity * saturation_history * 3./4. / np.pi)**(1./3.)
        history = graphics.Spheres(self.points, fill_radii, color=(0,0,1))
        return [shells, tubes, history]

    def porosity(self):
        box = np.prod(self.bbox)
        spheres = (self['sphere_radii']**3 * 4./3. * np.pi)
        lengths = np.linalg.norm(self.spans, axis=1)
        lengths/= 2 # account for duplicity
        circles = self['cylinder_radii']**2 * np.pi
        return (spheres.sum() + (lengths*circles).sum()) / box
