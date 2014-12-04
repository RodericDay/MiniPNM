import logging
import numpy as np
from scipy import optimize
import minipnm as mini
import MKS
MKS.define(globals())


class SimpleLatticedCatalystLayer(object):

    @classmethod
    def test(cls):
        # voltages = np.arange(0, 1.3, 0.1) * V
        radii = 50*nm
        porosity = 0.4
        thickness = 5*um
        cl = cls(thickness, radii, porosity, flat=True)
        cl.generate_agglomerate(200*cm**2/cm**2, 10*nm)
        return cl

    def __init__(self, thickness, radii, porosity, N=200, flat=False):
        '''
        given parameters, aim for a 10,000 pore model
        '''
        scale = 3 * radii
        nx = int( thickness / ( 3 * radii ) )
        nz = ny = int(np.sqrt(N / nx))
        if flat is True:
            ny = N // nx
            nz = 1
        nx /= 2
        ny *= 2
        self.topology = mini.Cubic([nx, ny, nz], scale(m))
        logging.info("\nPore-phase topology generated"
                     "\n\t{0.order} pores, {0.size} throats"
                     "\n\tgoal thickness: {thickness.quantity}"
                     "\n\tbbox:           {0.bbox}"
                     "".format(self.topology, **locals()) )

        radii = np.random.uniform( 0.5*radii(m), 1.2*radii(m), self.topology.order) * m
        self.geometry = mini.Radial(self.topology.points, radii(m), self.topology.pairs, prune=False)
        # fit the porosity by tweaking the throat factor
        self.geometry.cylinders.radii *= 1.578
        porosity = self.geometry.porosity()
        # we should do a couple of checks here:
        # - no radial collisions
        # - no throats wider than pores
        avg_throat_length = self.geometry.cylinders.heights.mean() * (m / nm)
        avg_throat_radius = self.geometry.cylinders.radii.mean() * (m / nm)
        logging.info("\nPore-phase geometry generated"
                     "\n\tporosity:          {porosity}"
                     "\n\tavg throat len:    {avg_throat_length} nm"
                     "\n\tavg throat radius: {avg_throat_radius} nm"
                     "".format(self.geometry, **locals()) )

        self.electron_transport = mini.bvp.System(self.geometry.pairs, A, V )
        self.proton_transport = mini.bvp.System(self.geometry.pairs, A, V )
        self.heat_transport = mini.bvp.System(self.geometry.pairs, W, K )
        self.gas_transport = mini.bvp.System(self.geometry.pairs, mol/s, 1 )
        self.water_transport = np.zeros(self.npores, dtype=bool)
        logging.info("\nTransports set up")

        x, y, z = self.topology.coords
        self.membrane = x==x.min()
        self.gdl = x==x.max()

    @property
    def npores(self):
        return self.topology.order

    @property
    def depth(self):
        return self.geometry['x'] * m

    @property
    def node_to_node_lengths(self):
        return self.topology.lengths * m

    @property
    def pore_surface_area(self):
        return self.geometry.spheres.areas * m**2

    @property
    def geometric_surface_area(self):
        t, h, w = self.geometry.bbox * m
        return h * w

    # AGGLOM RELATED

    def generate_agglomerate(self, specific_surface_area, nafion_thickness):
        self.specific_surface_area = specific_surface_area
        connection_radii = self.geometry.cylinders.radii * m
        self.carbon_area = np.pi * (connection_radii - nafion_thickness)**2
        self.nafion_area = np.pi * connection_radii**2 - self.carbon_area
        max_ratio = (nafion_thickness / connection_radii).max()
        logging.info("\nAgglomerate generated"
                     "\n\tavg nafion thickness: {0.average_nafion_thickness}"
                     "\n\tmax N / C ratio: {max_ratio}"
                     "".format(self, **locals()))

    @property
    def average_nafion_thickness(self):
        outer_rad = self.geometry.cylinders.radii * m
        inner_rad = (self.carbon_area / np.pi)**0.5
        return (outer_rad - inner_rad)(nm).mean()

    @property
    def total_agglomerate_area(self):
        return self.geometric_surface_area * self.specific_surface_area

    @property
    def pore_agglomerate_area(self):
        proportion = self.pore_surface_area / self.pore_surface_area.sum()
        return self.total_agglomerate_area * proportion

    # USER DEFINED:

    @property
    def electronic_conductances(self):
        s_C = 1000 * S / m # electric conductivity of carbon
        return -s_C * self.carbon_area / self.node_to_node_lengths

    @property
    def protonic_conductances(self):
        # s_N = 100 * np.exp( ( 15.036 * 1 - 15.811) * 1000*K/T + (-30.726 * 1 + 30.481) ) * S / m
        s_N = 10 * S / m # protonic conductivity of nafion
        return -s_N * self.nafion_area / self.node_to_node_lengths

    @property
    def thermal_conductances(self):
        t_C = 0.5 * W / K / m # thermal conductivity of carbon
        return -t_C * self.carbon_area / self.node_to_node_lengths

    @property
    def diffusive_conductances(self):
        D_b = 2.02E-5 * m**2 / s # binary diffusion coefficient
        P = 1 * atm
        T = self.temperature
        c = P / ( R * T )

        g_half = np.pi * self.geometry.spheres.radii*m * c * D_b
        g_cyl = self.geometry.cylinders.areas / self.geometry.cylinders.heights*m * c * D_b

        gis, gjs = g_half.quantity[self.geometry.pairs.T] * g_half.units
        g_D = (1 / gis + 1 / g_cyl + 1 / gjs)**-1

        S = np.where(self.throat_saturation, 1, 0)
        return g_D * (1 - S)

    @property
    def overpotential(self):
        E0 = 1.223 * V
        return self.electronic_potential - self.protonic_potential - E0

    def reaction_rate(self, T, n):
        '''
        Butler-Volmer
        '''
        f = self.pore_agglomerate_area / F
        j0 = 1.8E-2 * A / m**2
        alf = 0.5
        E1 = np.exp(   2 *   alf   * F / (R * T) * n )
        E2 = np.exp( - 2 * (1-alf) * F / (R * T) * n )
        return j0 * (E1 + E2) * f

    @property
    def throat_saturation(self):
        # saturated if either node is
        return self.water_transport[self.geometry.pairs].any(axis=1)

    # A BIT MORE ABSTRACTED

    @property
    def electronic_potential(self):
        self.electron_transport.conductances = self.electronic_conductances
        return self.electron_transport.solve(
            { self.measured_voltage : self.gdl }, self.local_current )

    @property
    def protonic_potential(self):
        self.proton_transport.conductances = self.protonic_conductances
        return self.proton_transport.solve(
            { 0*V : self.membrane }, -self.local_current )

    @property
    def temperature(self):
        return 333 * K

    def oxygen_molar_fraction(self, k):
        self.gas_transport.conductances = self.diffusive_conductances
        return self.gas_transport.solve( { 0.21 : self.gdl }, k=k)

    @property
    def measured_current_density(self):
        valid = ~(self.gdl | self.membrane) # axes tails
        total_current_generated = (valid*self.local_current).sum()
        return total_current_generated / self.geometric_surface_area

    # SOLVER METHODS

    def polarization_curve(self, voltage_range):
        current_densities = []
        for voltage in voltage_range:
            self.reach_steady_state(v=voltage)
            current_densities.append( self.measured_current_density(A/cm**2) )
        return np.array(current_densities) * A/cm**2

    def reach_steady_state(self, v, j=0*A/m**2, max_iter=100):
        self.measured_voltage = v
        self.local_current = j * self.pore_agglomerate_area
        for _ in range(max_iter):
            n = self.overpotential
            T = self.temperature
            k = self.reaction_rate(T, n)
            x = self.oxygen_molar_fraction(k)
            j = k*x * ( (A/m**2) / (mol/s) )
            new_local_current = j * self.pore_agglomerate_area
            # insufficient convergence criterion
            steady_state = np.allclose(new_local_current.quantity, self.local_current.quantity)
            self.local_current = new_local_current
            if steady_state:
                # return x cause it's a bit tricky to get otherwise, and
                # good for sanity checks
                return x
        else:
            raise Exception("Steady state not reached")
