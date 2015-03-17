import numpy as np
import minipnm as mini

class ArrayModel(object):
    global F, R, nO2, nH
    F = 96487
    R = 8.314
    nO2 = 4
    nH = 2

    def __init__(self, radii_array, node_spacing):
        '''
        Define what the network looks like
        based on an array of radii and distances
        '''
        self.topology = t = mini.Cubic.from_source(radii_array)
        self.topology.points *= node_spacing
        self.geometry = g = mini.Radial(t.points, t['source'], t.pairs, prune=False)

        # some aliases for boundary conditions
        x, y, z = self.topology.coords
        self.distance_from_membrane = x
        self.membrane = x==x.min()
        self.gdl = x==x.max()

        # some basic facts
        w, h, d = self.geometry.bbox
        self.face_area = w * h

        self.setup_water_transport()
        self.setup_linear_systems()

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
        self.oxygen_transport = mini.bvp.System(self.topology.pairs)
        self.proton_transport = mini.bvp.System(self.topology.pairs)

    def update_conductances(self, T, P, s):
        l = self.topology.lengths
        A = self.geometry.cylinders.areas

        # the saturation of the throat is the max saturation of
        # either of its nodes
        s = s[self.topology.pairs].max(axis=1)
        # above a certain saturation, block (not really necessary)

        c = P / ( R * T )
        D = 2.02E-5
        self.oxygen_transport.conductances = c * D * A / l * (1.0001-s)

        S = 10000000 # S / m # conductivity of nafion
        Ap = l**2 - A # complement of duct area
        self.proton_transport.conductances = S * Ap / l

    def reaction_rate_constant(self, overpotential, temperature):
        ''' Butler-Volmer '''
        n = overpotential
        T = temperature
        SA = self.geometry.spheres.areas
        # go together
        x = self.distance_from_membrane
        ecsa = 0.1
        i0 = 1.0e-11 # A / m2
        alpha = 0.5
        # bv proper
        E1 = np.exp(  -alpha   * nO2 * F / ( R * T ) * n )
        E2 = np.exp( (1-alpha) * nO2 * F / ( R * T ) * n )
        k = i0 * ecsa * SA * ( E1 - E2 )
        return k # A

    def resolve(self, cell_voltage, ambient_temperature, pressure=201325,
                time_step=0.1, flood=True):
        '''
        given a certain voltage and temperature,
        slowly evolve the water transport of the network
        resolving all the relevant non-linear equations at every step
        '''
        # create containers for tracked states
        self.oxygen_history = []
        self.proton_history = []
        self.current_history = []
        # do a dry run to match the default empty water sim
        self.solve_coupled_nonlinear(cell_voltage, ambient_temperature, pressure)
        while flood and not self.transient_steady():
            # generate water where reaction is taking place
            # according to Faraday's law and stoichiometric ratios
            generated = self.membrane * time_step * 1E-18 * 10
            # distribute it
            self.water_transport.distribute(generated)
            # run it
            self.solve_coupled_nonlinear(cell_voltage, ambient_temperature, pressure)

    def transient_steady(self):
        # goal: check history until changes stop happening
        # currently: stop when breakthrough to gdl
        full = self.water_transport.state > 0.9
        return any(full & self.gdl)

    def solve_coupled_nonlinear(self, electronic_potential, temperature, pressure):
        # update all the conductances
        saturation = self.water_transport.state
        self.update_conductances(temperature, pressure, saturation)
        # initial guess & etc
        open_current_voltage = 1.223 # V
        overpotential = -0.02 # V
        protonic_potential = [electronic_potential - overpotential - open_current_voltage]
        # damping factor
        df = damping_factor = 0.1 * electronic_potential
        for _ in range(10000):

            k = self.reaction_rate_constant(overpotential, temperature)
            x = oxygen_fraction = self.oxygen_transport.solve(
                { 0.05 : self.gdl }, k=k / ( nO2 * F ) )
            i = k * x
            h = self.proton_transport.solve(
                { -i.sum() : self.membrane }, s=i)

            protonic_potential.append( (1-df)*protonic_potential[-1] + df*h )
            overpotential = electronic_potential - protonic_potential[-1] - open_current_voltage

            # check for conv
            ratio = protonic_potential[-2] / h
            cond1 = ratio.max() < 1.01
            cond2 = ratio.min() > 0.99
            if cond1:
                print '.'*(_//10)
                break
        else:
            raise Exception("no convergence after {}. {} {}"
                            "".format(_, ratio.mean(), ratio.ptp()))

        # at every step, we also keep track of any cool vars
        self.oxygen_history.append( x )
        self.proton_history.append( h ) # or [-1]?
        self.current_history.append( i )

    # visual methods
    def water_history_stack(self):
        saturations = self.water_transport.history
        frozen_pores = self.water_transport.expand_block_states()
        combined = np.where(frozen_pores, np.nan, saturations)
        W = np.dstack(self.topology.asarray(layer) for layer in combined).T
        return W

    def stack(self, history):
        S = np.dstack(self.topology.asarray(layer) for layer in history).T
        return S


if __name__ == '__main__':
    G = np.random.uniform(400E-9, 900E-9, [20, 3, 3])
    model = ArrayModel(G, 2 * 1000E-9)
    mini.gui.profileview(model)
