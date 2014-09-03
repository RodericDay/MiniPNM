import random, math
import numpy as np

def inside(point, bbox):
    return all(c>0 and c<d for c, d in zip(point, bbox))

def get_nearby_indexes(grid, gcoords, scope=None):
    if scope is None:
        scope = len(gcoords) + 2
    slcobj = [slice(max(0, c-scope+1), c+scope) for c in gcoords]
    return set(grid[slcobj].flat) - {-1}

def get_nearby_discs(disk_list, sampled):
    for other in disk_list:
        rsum = other[-1] + sampled[-1]
        if any(abs(a-b) > rsum for a,b in zip(other[:-1], sampled[:-1])):
            continue
        yield other

def intersecting(neighbor_list, sampled):
    for neighbor in neighbor_list:
        rsum = (neighbor[-1] + sampled[-1])**2
        dist = sum((a-b)**2 for a, b in zip(neighbor[:-1], sampled[:-1]))
        yield rsum > dist

def sample_around(disk, r, d=2):
    x0, y0, z0, r0 = disk
    rmin = r0 + r
    rmax = r0 + 2 * r

    angle_xy = random.random() * 2 * math.pi
    angle_z = random.random() * 2 * math.pi if d==3 else 0
    dist = ( random.random()*(rmax**d - rmin**d) + rmin**d ) ** (1./d)

    x = x0 + dist * math.sin(angle_xy) * math.cos(angle_z)
    y = y0 + dist * math.cos(angle_xy) * math.cos(angle_z)
    z = z0 + dist * math.sin(angle_z)

    return x, y, z, r

def poisson_disk_sampling(bbox, r, n_iter=30, p_max=10000):
    '''
    A 3D version of Robert Bridson's algorithm, perhaps best illustrated by
    Mike Bostock's following D3.js animation:
    http://bl.ocks.org/mbostock/dbb02448b0f93e4c82c3

    Takes in a virtual 'bounding box' and a generator from which to sample,
    and purports to pack the space as tightly and randomly as possible,
    outputting the coordinates and radii of the corresponding circles.
    '''
    cell_size = r / math.sqrt(len(bbox))
    grid = np.zeros([d//cell_size+1 for d in bbox], dtype=int) - 1
    gcoords = lambda xyz: tuple(c//cell_size for c in xyz[:len(bbox)])

    disk_list = [(0,0,0,r)]
    grid[gcoords(disk_list[0][:-1])] = 0
    available = [0]

    while available and len(disk_list) < p_max:
        i = random.choice(available)
        origin = disk_list[i]

        for j in range(n_iter):
            sampled = sample_around(origin, r, d=len(bbox))
            if not inside(sampled, bbox):
                continue

            # neighbor_list = get_nearby_discs(disk_list, sampled)
            neighbor_list = (disk_list[i] for i in get_nearby_indexes(grid, gcoords(sampled)))
            if any(intersecting(neighbor_list, sampled)):
                continue

            # if we got here the point is valid!
            new_index = len(disk_list)
            available.append( new_index )
            grid[gcoords(sampled)] = new_index
            disk_list.append( sampled )
            break
            
        else:
            # somewhat unknown python feature, think of it as "nobreak"
            # if we got here, it's because no new points were able to be
            # generated. this source is probably too crowded to have new
            # neighbors, so we stop considering it
            available.remove(i)

    x,y,z,r = zip(*disk_list)
    return zip(x,y,z), r
