import numpy as np
from scipy import optimize
import minipnm as mini
import MKS
MKS.define(globals())


class SimpleLatticedCatalystLayer(object):
    
    def __init__(self, grid_shape, thickness, porosity):
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

    def generate_agglomerate(self, specific_surface_area, nafion_thickness):
        self.specific_surface_area = specific_surface_area
        connection_radii = self.geometry.cylinders.radii.squeeze() * m
        self.carbon_area = np.pi * (connection_radii - nafion_thickness)**2
        self.nafion_area = np.pi * connection_radii**2 - self.carbon_area

    def generate_systems(self, pressure):
        self.pressure = pressure
        x,y,z = self.geometry.coords
        membrane = x == x.min()
        gdl = x == x.max()

        s_C = 1000 * S / m # electric conductivity of carbon
        e_conds = lambda self: (-s_C * self.carbon_area / self.lengths)(S)
        ebcs = { 0.68 : gdl }
        self.electron_transport = mini.bvp.System(self.geometry.pairs, ebcs, e_conds(self) )

        # s_N = 100 * np.exp( ( 15.036 * 1 - 15.811) * 1000*K/T + (-30.726 * 1 + 30.481) ) * S / m
        s_N = 10 * S / m # protonic conductivity of nafion
        p_conds = lambda self: (-s_N * self.nafion_area / self.lengths)(S)
        pbcs = { 0 : membrane }
        self.proton_transport = mini.bvp.System(self.geometry.pairs, pbcs, p_conds(self) )

        t_C = 0.5 * W / K / m * 100 # thermal conductivity of carbon
        t_conds = lambda self: (-t_C * self.carbon_area / self.lengths)(W/K)
        tbcs = { 353 : gdl }
        self.heat_transport = mini.bvp.System(self.geometry.pairs, tbcs, t_conds(self) )

        D_b = 2.02E-5 * m**2 / s * 1E14 # binary diffusion coefficient
        def d_conds(self):
            c = self.pressure / ( R * 353*K )
            g_half = np.pi * self.geometry.spheres.radii.squeeze() * c * D_b
            g_cyl = self.geometry.cylinders.areas / self.geometry.cylinders.heights.squeeze() * c * D_b

            gis, gjs = g_half.quantity[self.geometry.pairs.T] * g_half.units
            g_D = (1 / gis + 1 / g_cyl + 1 / gjs)**-1
            return g_D(mol/m/s)
        dbcs = { 0.21 : gdl }
        self.gas_transport = mini.bvp.System(self.geometry.pairs, dbcs, d_conds(self))

    @property
    def npores(self):
        return self.geometry.order

    @property
    def depth(self):
        return self.geometry['x'] * m

    @property
    def lengths(self):
        return self.geometry.lengths * m

    @property
    def pore_surface_area(self):
        return self.geometry.spheres.areas * m**2

    @property
    def total_agglomerate_area(self):
        return self.geometric_surface_area * self.specific_surface_area

    @property
    def pore_agglomerate_area(self):
        proportion = self.pore_surface_area / self.pore_surface_area.sum()
        return self.total_agglomerate_area * proportion

    @property
    def geometric_surface_area(self):
        t, h, w = self.geometry.bbox * m
        return h * w

    def measured_current_density(self, local_current_density):
        total_current_generated = (local_current_density * self.pore_agglomerate_area).sum()
        return total_current_generated / self.geometric_surface_area

    @property
    def orr(self):
        '''
        Butler-Volmer
        '''
        j0 = 1.8E-2 * A / m**2 
        alf = 0.5
        T = self.temperature
        n = self.overpotential
        E1 = np.exp(   2 *   alf   * F / (R * T) * n )
        E2 = np.exp( - 2 * (1-alf) * F / (R * T) * n )
        AO2 = self.oxygen_molar_fraction.clip(0, np.inf)
        j_orr = -j0 * AO2**0.25 * ( E1 - E2 )
        return j_orr / 1000000

    def solve_systems(self, input_current_density):
        local_current = input_current_density * self.pore_agglomerate_area

        self.electronic_potential = self.electron_transport.solve( -local_current(A) ) * V
        self.protonic_potential = self.proton_transport.solve( local_current(A) ) * V

        E0 = 1.223 * V
        self.overpotential = self.electronic_potential - self.protonic_potential - E0

        heat_generation = local_current * self.overpotential
        self.temperature = self.heat_transport.solve( heat_generation(W) ) * K

        oxygen_molar_consumption = input_current_density / (4 * F)
        self.oxygen_molar_fraction = self.gas_transport.solve( oxygen_molar_consumption(mol/s/m**2) )

        output_current_density = self.orr
        return output_current_density
