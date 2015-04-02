import numpy as np
from . import utils

def drop_coplanar(points):
    '''
    useful function when using scipy tessellation packages, which break
    if given coplanar input
    '''
    not_coplanar = np.diff(points, axis=0).sum(axis=0).nonzero()
    return points.T[not_coplanar].T


class GeometryInterface(object):
    '''
    Geometry interface allows manipulation of dictionary entries
    as if we were dealing with a separate geometrical object, but ultimately
    still tapping into the main data store
    '''
    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, item):
        return self.parent[item]

    def __setitem__(self, item, value):
        self.parent[item] = value

    def get(self, item, default):
        try:
            return self[item]
        except KeyError:
            return default


class Spheres(GeometryInterface):
    centers = utils.property_from(['x','y','z'])
    radii = utils.property_from(['sphere_radii'])

    @property
    def surface_areas(self):
        return (4 * np.pi * self.radii**2)

    @property
    def volumes(self):
        return (4./3. * np.pi * self.radii**3)

    @property
    def cross_sectional_areas(self):
        return (np.pi * self.radii**2)


class Cylinders(GeometryInterface):
    midpoints = utils.property_from(['cylinder_center_x','cylinder_center_y','cylinder_center_z'])
    radii = utils.property_from(['cylinder_radii'])
    spans = utils.property_from(['cylinder_length_x','cylinder_length_y','cylinder_length_z'])

    def generate(self, f):
        '''
        generates geometrically correct cylinders to act as sphere connectors.
        '''
        pairs = self.parent.pairs
        sphere_centers = self.parent.spheres.centers
        sphere_radii = self.parent.spheres.radii.reshape(-1,1) # transpose
        # assign radii
        self.radii = sphere_radii[pairs].min(axis=1)/f

        point_to_point_mid = self.parent.midpoints + self.parent.spans / 2
        unit_vectors = (self.parent.spans.T / self.parent.lengths).T
        # shave endpoints
        self.spans = self.parent.spans - sphere_radii[pairs].sum(axis=1)*unit_vectors
        # shift midpoint
        dR = np.diff(sphere_radii[pairs], axis=1).T[0].T
        self.midpoints = self.parent.midpoints - dR*unit_vectors / 2

    def intersecting(self, center, radius):
        '''
        takes in a sphere (center, radius), and returns a mask indicating
        which cylinders intersected the sphere

        Distance from point to (infinite) line(s):
        d = (a - p) - ((a - p) dot n) n

           a
        ----*----*-----> n
             \   |
              \  |
           a-p \ | d
                \|
                 * p

        Additionally, account for distance-to-cap
        '''
        unit = (self.parent.spans.T / self.parent.lengths).T
        # a minus p
        amp = self.midpoints - center
        shift = ((amp * unit).sum(axis=1) * unit.T).T
        spans = self.spans
        closest_point = self.midpoints - shift.clip(-abs(spans/2), abs(spans/2))
        # different elimination condition depending on cap vs. radial collision
        closest_point_if_inf = self.midpoints - shift
        cap_condition = np.any(closest_point!=closest_point_if_inf, axis=1)
        distance = np.linalg.norm(center - closest_point, axis=1)
        rad2rad = radius + self.radii.T
        clear = (cap_condition & (distance > radius-0.01)) | (distance > rad2rad)
        return ~clear

    @property
    def heights(self):
        return np.linalg.norm(self.spans, axis=1)

    @property
    def cross_sectional_areas(self):
        return np.pi * self.radii**2

    @property
    def surface_areas(self):
        return 2 * np.pi * self.radii * self.heights

    @property
    def volumes(self):
        return self.cross_sectional_areas * self.heights
