import logging
import numpy as np
from scipy import optimize
import minipnm as mini
import MKS
MKS.define(globals())

def butler_volmer(oxygen, overpotential):
    E1 = np.exp(   2 *   alf   * F / (R * T) * overpotential )
    E2 = np.exp( - 2 * (1-alf) * F / (R * T) * overpotential )
    AO2 = 1
    j_orr = -j0 * AO2**0.25 * ( E1 - E2 )
    return j_orr


class SimpleLatticedCatalystLayer(object):
    
    def __init__(self, grid_shape, thickness, porosity):
        '''
        init by generating geometrical stuff
        '''
        topology = mini.Cubic(grid_shape)
        scale = thickness(m) / topology.bbox[0]
        topology.points *= scale

        # quickfit for porosity
        def objf(r):
            geometry = mini.Radial(topology.points, abs(r), topology.pairs, prune=False)
            return geometry.porosity()
        def minf(r):
            return abs(porosity - objf(r))
        r = abs(optimize.minimize_scalar(minf).x)

        self.geometry = mini.Radial(topology.points, r, topology.pairs)

    @property
    def npores(self):
        return self.geometry.order

    @property
    def pore_surface_area(self):
        return self.geometry.spheres.areas

    @property
    def geometric_surface_area(self):
        t, h, w = self.geometry.bbox * m
        return h * w

    def react_to_input(self, current):
        print current
        # oxygen_molar_flux = current / ( 4 * F )
        # oxygen_molar_fraction = mini.bvp.spsolve(self.O, oxygen_molar_flux)
        # oxygen_concentration = oxygen_molar_fraction * c

    def render(self):
        self.geometry.render()

if __name__ == '__main__':
    t = 15*um
    I_guess = 100000 * mA / cm ** 2

    catalyst_layer = SimpleLatticedCatalystLayer([10, 2, 2], t, 0.4)
    catalyst_layer.react_to_input(2*A)
