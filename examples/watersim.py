import numpy as np
import minipnm as mini


class ArrayModel(object):

    def __init__(self, radii_array, node_spacing):
        '''
        Define what the network looks like
        based on an array of radii and distances
        '''
        self.topology = t = mini.Cubic.from_source(radii_array)
        self.geometry = g = mini.Radial(t.points, t['source'], t.pairs, prune=False)

        # some aliases for boundary conditions
        x, y, z = self.topology.coords
        self.membrane = x==x.min()
        self.gdl = x==x.max()

        self.setup_water_transport()
        self.setup_linear_systems()

        # make a collector for found states
        self.state_history = []

    def setup_water_transport(self):
        cmat = self.topology.adjacency_matrix
        # since algorithm is ony looking at relative ordering
        # just choose a quantity that represents conductivity well
        # here, we choose areas
        cmat.data = self.geometry.cylinders.areas
        self.water_transport = mini.simulations.Invasion(
            cmat=cmat,
            capacities=self.geometry.spheres.volumes)
        # we can update these conductivities
        # by modifying the Invasion._cmat attribute

    def setup_linear_systems(self):
        # we need to tie these to radius, rigidly or weakly
        self.oxygen_transport = mini.bvp.System(self.topology.pairs)
        self.oxygen_conductances = 3.769E-8 # base level
        self.oxygen_transport.conductances = self.oxygen_conductances

        self.proton_transport = mini.bvp.System(self.topology.pairs)
        self.proton_conductances = 0.00235592
        self.proton_transport.conductances = self.proton_conductances

        # heat transport
        cmat = self.topology.adjacency_matrix
        cmat.data[:] = 1
        self.heat_transport = mini.simulations.Diffusion(cmat=cmat)

    def resolve(self, cell_voltage, ambient_temperature,
                time_step=0.1, flood=True):
        '''
        given a certain voltage and temperature,
        slowly evolve the water transport of the network
        resolving all the relevant non-linear equations at every step
        and then return all the data generated
        '''
        # there exists an Invasion.steady_state method,
        # but here we use our own

        # water sim starts with an empty run, so we
        # even it out
        self.solve_coupled_nonlinear(cell_voltage, ambient_temperature)
        while not self.transient_steady():
            if not flood: break
            # generate water where reaction is taking place
            # according to Faraday's law and stoichiometric ratios
            generated = self.membrane * time_step
            # distribute it
            self.water_transport.distribute(generated)
            # have a condition for blocking
            waterlogged = self.water_transport.state > 0.9
            blocked = waterlogged * 0
            # convert to desired format
            open_throats = ~self.topology.cut(blocked, directed=False)
            self.oxygen_transport.conductances = self.oxygen_conductances * open_throats
            # runit
            self.solve_coupled_nonlinear(cell_voltage, ambient_temperature)

    def transient_steady(self):
        # goal: check history until changes stop happening
        # currently: stop when breakthrough to gdl
        full = self.water_transport.state > 0.9
        return any(full & self.gdl)

    def solve_coupled_nonlinear(self, electronic_potential, temperature):
        # here we update all the conductances as per the
        # latest changes in water content
        # and solve the systems
        open_current_voltage = 1.223 # V
        overpotential = -0.02 # V
        # damping factor
        df = damping_factor = 0.1 * electronic_potential
        # protonic potential keeps track of its old value
        protonic_potential = [electronic_potential - overpotential - open_current_voltage]
        for _ in range(500):

            k = self.reaction_rate_constant(overpotential, temperature)
            x = oxygen_fraction = self.oxygen_transport.solve({ 0.01 : self.gdl }, k=k)

            i = self.local_current(k, x)
            h = self.proton_transport.solve({ 1E-8 : self.membrane }, s=i)
            protonic_potential.append( (1-df)*protonic_potential[-1] + df*h )

            overpotential = electronic_potential - protonic_potential[-1] - open_current_voltage

            # check for conv
            ratio = protonic_potential[-2] / h
            cond1 = ratio.max() < 1.01
            cond2 = ratio.min() > 0.99
            if cond1:
                break

        # at every step, we also keep track of the chosen variable
        self.heat_transport.march( i )
        self.state_history.append( self.heat_transport.state )

    def reaction_rate_constant(self, overpotential, temperature):
        ''' Butler-Volmer '''
        n = overpotential
        T = temperature
        F = 96487
        R = 8.314
        SA = 1.936E-9 # m2
        I0 = 1.0e-11 # A/m2
        nO2 = 4
        ecsa = 20
        alpha = 0.5
        E1 = np.exp(  -alpha   * nO2 * F / ( R * T ) * n )
        E2 = np.exp( (1-alpha) * nO2 * F / ( R * T ) * n )
        k = I0 * ecsa * ( E1 - E2 )
        return k / ( nO2 * F )

    def local_current(self, reaction_rate_constant, oxygen_fraction):
        k = reaction_rate_constant
        x = oxygen_fraction
        F = 96487
        nO2 = 4
        return k * x * nO2 * F

    # visual methods
    def water_history_stack(self):
        saturations = self.water_transport.history
        frozen_pores = self.water_transport.expand_block_states()
        combined = np.where(frozen_pores, np.nan, saturations)
        W = np.dstack(self.topology.asarray(layer) for layer in combined).T
        return W

    def state_history_stack(self):
        S = np.dstack(self.topology.asarray(layer) for layer in self.state_history).T
        return S


# T-shape
R = 0.3 * np.ones([40, 30])
R[:3] = 0.2
R[:, 13:17] = 0.2
R = np.random.uniform(0.1, 0.5, [40, 30])
model = ArrayModel(R, 1)
model.resolve(1.1, 263, flood=True)

from minipnm.gui import floodview
W = model.water_history_stack()
S = model.state_history_stack()
G = model.topology.asarray(model.geometry.spheres.volumes).T
floodview(W, S, G)
