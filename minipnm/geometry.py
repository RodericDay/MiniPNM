import numpy as np
import minipnm as mini


def cylinders(centers, radii, pairs=None):
    '''
    generates geometrically correct cylinders to act as sphere connectors.
    quantities can either be given as pairs, or as an array with pair indices
    supplied separately.

    if the cylinder radii is supplied, the calculations will be adjusted for
    best fit. default is point connection at sphere surface (minimalistic)

    output:
        spans, midpoints
    '''
    if pairs is not None:
        centers = centers[pairs]
        radii = radii[pairs]

    spans = np.diff(centers, axis=1).squeeze()
    midpoints = centers[ :, 0, :] + spans/2.
    unit = (spans.T / np.linalg.norm(spans, axis=1)).T
    # shave endpoints
    radsum = radii.sum(axis=1)
    spans -= (radsum.T * unit.T).T
    # shift midpoint
    radsub = np.diff(radii, axis=1)
    midpoints -= (radsub.T * unit.T).T / 2.
    return spans, midpoints

def intersecting(center, radius,
                 spans, midpoints, radii):
    '''
    takes in a sphere and multiple cylinders
    - sphere(center, radius)
    - cylinders(spans, midpoints, radii)
    returns a mask indicating which cylinders intersected the sphere
    
    Distance from point to (infinite) line(s):
    d = (a - p) - ((a - p) dot n) n

       a
    ----*----*-----> n
         \   |
          \  |
       a-p \ | d
            \|
             * p
    '''
    unit = (spans.T / np.linalg.norm(spans, axis=1)).T
    # a minus p
    amp = midpoints - center
    shift = ((amp * unit).sum(axis=1) * unit.T).T
    closest_point = midpoints - shift.clip(-abs(spans/2), abs(spans/2))
    # different elimination condition depending on cap vs. radial collision
    closest_point_if_inf = midpoints - shift
    cap_condition = np.any(closest_point!=closest_point_if_inf, axis=1)
    distance = np.linalg.norm(center - closest_point, axis=1)
    rad2rad = radius + radii
    clear = (cap_condition & (distance > radius-0.01)) | (distance > rad2rad)
    return ~clear