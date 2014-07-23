import numpy as np
np.set_printoptions(linewidth=160, precision=2)
import minipnm as mini

network = mini.Bridson([7,7,7], (i%5/5.+0.1 for i,_ in enumerate(iter(int,1))))
network.pairs = mini.Delaunay.edges_from_points(network.points, directed=False)
# network = mini.Delaunay.random(20)
# network['radii'] = np.random.rand(network.order)/5

# find dimensions of connecting cylinders
# given points, spherical radii, and pairings
network['radii_sphere'] = network['radii']
network['radii_cylinder'] = network['radii_sphere'][network.pairs].min(axis=1)/2
# unit vector
uvs = (network.spans.T / np.linalg.norm(network.spans, axis=1)).T
# span needs to be shrunk
radsum = network['radii_sphere'][network.pairs].sum(axis=1)
spans = network.spans - (radsum.T * uvs.T).T
# midpoint needs to be shifted
radsub = np.diff(network['radii_sphere'][network.pairs], axis=1)
midpoints = network.midpoints - (radsub.T * uvs.T).T/2.


# now we should do some cleanup
# spheres intersecting tubes
start_count = len(midpoints)
for center, radius in zip(network.points, network['radii_sphere']):
    '''
    Distance from point to (infinite) line:
    d = (a - p) - ((a - p) dot n) n

       a
    ----*----*-----> n
         \   |
          \  |
       a-p \ | d
            \|
             * p
    '''
    # a minus p
    amp = midpoints - center
    shift = ((amp * uvs).sum(axis=1) * uvs.T).T
    closest_point = midpoints - shift.clip(-abs(spans/2), abs(spans/2))
    # different elimination condition depending on cap vs. radial collision
    closest_point_if_inf = midpoints - shift
    cap_condition = np.any(closest_point!=closest_point_if_inf, axis=1)
    distance = np.linalg.norm(center - closest_point, axis=1)
    rad2rad = radius + network['radii_cylinder']

    safe = (cap_condition & (distance > radius-0.01)) | (distance > rad2rad)

    # deletes and stuff
    uvs = uvs[safe]
    midpoints = midpoints[safe]
    spans = spans[safe]
    network['radii_cylinder'] = network['radii_cylinder'][safe]
print start_count - len(midpoints)

# currently displaying bad tubes
scene = mini.Scene()
scene.add_spheres(network.points, network['radii_sphere'], color=(0,0,1), alpha=0.4)
scene.add_tubes(midpoints, spans, network['radii_cylinder'])
scene.play()