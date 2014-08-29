import random, math

def poisson_disk_sampling(bbox, pdf, n_iter=30, p_max=10000):
    '''
    A 3D version of Robert Bridson's algorithm, perhaps best illustrated by
    Mike Bostock's following D3.js animation:
    http://bl.ocks.org/mbostock/dbb02448b0f93e4c82c3

    Takes in a virtual 'bounding box' and a generator from which to sample,
    and purports to pack the space as tightly and randomly as possible,
    outputting the coordinates and radii of the corresponding circles.
    '''
    # subfunctions that use local namespace
    def outside():
        ''' checks if point center is within bounds '''
        return any(c < 0 or c > d for c,d in zip([xj,yj,zj], bbox))

    def near():
        ''' checks if distance between two centers larger than radii'''
        for (xk, yk, zk), rk in zip(points, radii):
            radii_sum = r + rk
            # manhattan distance filter for distant points
            dist_mhtn = [abs(xj-xk), abs(yj-yk), abs(zj-zk)]
            if any(dm > radii_sum for dm in dist_mhtn):
                yield False
            else:
                dist_btwn = math.sqrt(sum(d**2 for d in dist_mhtn))
                yield radii_sum > dist_btwn

    points = [(0,0,0)]
    radii = [next(pdf)]
    available = [0]
    is3d = True if len(bbox)==3 else False
    e = 3. if is3d else 2.
    
    while available and len(points) <= p_max:
        r = float(next(pdf))
        source = i = random.choice(available)
        xi, yi, zi = points[i]
        inner_r = min_dist = radii[i] + r
        outer_r = min_dist * 2


        for j in range(n_iter):
            # try a random point in the sampling space
            aj = random.random() * 2 * math.pi
            bj = random.random() * 2 * math.pi if is3d else 0
            rj = ( random.random()*(outer_r**e - inner_r**e) + inner_r**e )**(1./e)

            xj = rj * math.cos(bj) * math.sin(aj) + xi
            yj = rj * math.cos(bj) * math.cos(aj) + yi
            zj = rj * math.sin(bj) + zi

            # bail of checks fail
            if outside() or any(near()):
                continue
    
            # if we got here the point is valid!
            available.append( len(points) )
            points.append( (xj, yj, zj) )
            radii.append( r )
            break
            
        else:
            # somewhat unknown python feature, think of it as "nobreak"
            # if we got here, it's because no new points were able to be
            # generated. this source is probably too crowded to have new
            # neighbors, so we stop considering it
            available.remove(i)

    return points, radii
