import numpy as np

'''
transformation houses processes that are deeply intertwined with 'Network'
class functionality, and cannot be simply broken down into modularized array
operations.

warning:
    these functions will not return data, emphasizing that they are
    mutating the objects that are sent as arguments. be careful!
'''

def convexify(self, f=10):
    ''' this method gives networks a slight convexity to allow
        the ConvexHull algorithm to detect the faces. The translation is
        proportional to distance, but pinched at corners.
         __       __
        |__| --> (__)

        it works with any network, but results may be strange for non-cuboids
    '''        
    x,y,z = self['coords'].T
    w,h,t = self.dims*f

    dx = np.sinh(x-x.mean())/w
    ddx = np.abs(dx).max()-np.abs(dx)
    dy = np.sinh(y-y.mean())/h
    ddy = np.abs(dy).max()-np.abs(dy)
    dz = np.sinh(z-z.mean())/t
    ddz = np.abs(dz).max()-np.abs(dz)

    x = x + dx*ddy*ddz
    y = y + dy*ddx*ddz
    z = z + dz*ddx*ddy

    self['coords'] = np.vstack([x,y,z]).T

def add_natural_boundaries(network):
    '''
    takes every boundary pore and obtains the distance to its neighbours, and
    then creates a virtual pore diametrically opposite
    '''
    pl = network.points
    tl, hl = network.pairs.T
    new_points, new_pairs = [], []
    i = network.order
    for t in network.boundary().nonzero()[0]:
        for h in hl[t==tl]:
            a,b = pl[[t,h]]
            new_points.append(a - (b-a))
            new_pairs.append((t,i))
            i+=1

    network.points = np.vstack([network.points, new_points])
    network.pairs = np.vstack([network.pairs, new_pairs])