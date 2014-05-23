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

def pad():
    '''
    '''