import numpy as np
from scipy import optimize
import minipnm as mini
import MKS
MKS.define(globals())


class SimpleLatticedCatalystLayer(object):
    
    def __init__(self, grid_shape, thickness, porosity, pressure):
        topology = mini.Cubic(grid_shape)
        scale = thickness(m) / topology.bbox[0]
        topology.points *= scale
        self.pressure = pressure

        # quickfit for porosity
        def objf(r):
            geometry = mini.Radial(topology.points, abs(r), topology.pairs, prune=False)
            return geometry.porosity()
        def minf(r):
            return abs(porosity - objf(r))
        r = abs(optimize.minimize_scalar(minf).x)

        self.geometry = mini.Radial(topology.points, r, topology.pairs)

        if any(self.geometry.spheres.radii < 10E-9):
            raise Exception("some radii are smaller than 10 nm")

    def generate_agglomerate(self, specific_surface_area, nafion_thickness):
        self.specific_surface_area = specific_surface_area
        connection_radii = self.geometry.cylinders.radii * m
        self.carbon_area = np.pi * (connection_radii - nafion_thickness)**2
        self.nafion_area = np.pi * connection_radii**2 - self.carbon_area

    def generate_systems(self):
        self.electron_transport = mini.bvp.System(self.geometry.pairs, self.electronic_conductances, A, V )
        self.proton_transport = mini.bvp.System(self.geometry.pairs, self.protonic_conductances, A, V )
        self.heat_transport = mini.bvp.System(self.geometry.pairs, self.thermal_conductances, W, K )
        self.gas_transport = mini.bvp.System(self.geometry.pairs, self.diffusive_conductances, mol/s, 1 )

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

    @property
    def electronic_conductances(self):
        s_C = 1000 * S / m # electric conductivity of carbon
        return -s_C * self.carbon_area / self.lengths

    @property
    def protonic_conductances(self):
        # s_N = 100 * np.exp( ( 15.036 * 1 - 15.811) * 1000*K/T + (-30.726 * 1 + 30.481) ) * S / m
        s_N = 10 * S / m # protonic conductivity of nafion
        return -s_N * self.nafion_area / self.lengths

    @property
    def thermal_conductances(self):
        t_C = 0.5 * W / K / m # thermal conductivity of carbon
        return -t_C * self.carbon_area / self.lengths

    @property
    def diffusive_conductances(self):
        D_b = 2.02E-5 * m**2 / s # binary diffusion coefficient
        c = self.pressure / ( R * 353*K )

        g_half = np.pi * self.geometry.spheres.radii*m * c * D_b
        g_cyl = self.geometry.cylinders.areas / self.geometry.cylinders.heights*m * c * D_b

        gis, gjs = g_half.quantity[self.geometry.pairs.T] * g_half.units
        g_D = (1 / gis + 1 / g_cyl + 1 / gjs)**-1
        return g_D

    def measured_current_density(self, local_current_density):
        total_current_generated = (local_current_density * self.pore_agglomerate_area).sum()
        return total_current_generated / self.geometric_surface_area

    @property
    def orr(self):
        return self.output_current_density

    def solve_systems(self, input_current_density, bval):
        x,y,z = self.geometry.coords
        membrane = x == x.min()
        gdl = x == x.max()

        local_current = input_current_density * self.pore_agglomerate_area

        self.electronic_potential = self.electron_transport.solve( { bval*V : gdl }, local_current )
        self.protonic_potential = self.proton_transport.solve( { 0*V : membrane }, -local_current )

        E0 = 1.223 * V
        self.overpotential = self.electronic_potential - self.protonic_potential - E0

        heat_generation = local_current * self.overpotential
        self.temperature = self.heat_transport.solve( { 353*K : gdl } )

        T = self.temperature
        n = self.overpotential
        j0 = 1.8E-2 * A / m**2
        alf = 0.5
        E1 = np.exp(   2 *   alf   * F / (R * T) * n )
        E2 = np.exp( - 2 * (1-alf) * F / (R * T) * n )
        k = j0 * (E1 + E2) * 1E-20
        self.oxygen_molar_fraction = self.gas_transport.solve( { 0.21 : gdl }, k=np.where(gdl, 0, k.quantity) * mol/s)

        self.output_current_density = np.where( membrane | gdl, 0, self.oxygen_molar_fraction * k.quantity) * 1E11 * A/cm**2
        return self.output_current_density
