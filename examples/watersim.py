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
        self.oxygen_transport = mini.bvp.System(self.topology.pairs)
        self.proton_transport = mini.bvp.System(self.topology.pairs)

        # heat transport is a different underlying mechanism
        cmat = self.topology.adjacency_matrix
        cmat.data[:] = 1
        self.heat_transport = mini.simulations.Diffusion(cmat=cmat, nCFL=0.0001)

    def update_conductances(self, T, P):
        l = self.topology.lengths
        A = self.geometry.cylinders.areas

        c = P / ( R * T )
        D = 2.02E-5
        self.oxygen_conductances = c * D * A / l
        self.oxygen_transport.conductances = self.oxygen_conductances

        s = 1 # S / m # conductivity of nafion
        Ap = l**2 - A # complement
        self.proton_transport.conductances = s * Ap / l

    def resolve(self, cell_voltage, ambient_temperature, pressure=201325,
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
        self.solve_coupled_nonlinear(cell_voltage, ambient_temperature, pressure)
        while flood and not self.transient_steady():
            # generate water where reaction is taking place
            # according to Faraday's law and stoichiometric ratios
            generated = self.membrane * time_step
            # distribute it
            self.water_transport.distribute(generated)

            # have a condition for blocking
            waterlogged = self.water_transport.state > 0.9
            blocked = waterlogged
            # convert to desired format
            open_throats = ~self.topology.cut(blocked, directed=False)
            self.oxygen_transport.conductances = self.oxygen_conductances * open_throats

            # run it
            self.solve_coupled_nonlinear(cell_voltage, ambient_temperature, pressure)

    def transient_steady(self):
        # goal: check history until changes stop happening
        # currently: stop when breakthrough to gdl
        full = self.water_transport.state > 0.9
        return any(full & self.gdl)

    def solve_coupled_nonlinear(self, electronic_potential, temperature, pressure):
        # here we update all the conductances as per the
        # latest changes in water content
        # and solve the systems
        self.update_conductances(temperature, pressure)
        # initial guess & etc
        open_current_voltage = 1.223 # V
        overpotential = -0.02 # V
        # damping factor
        df = damping_factor = 0.1 * electronic_potential
        # protonic potential keeps track of its old value
        protonic_potential = [electronic_potential - overpotential - open_current_voltage]
        for _ in range(1000):

            k = self.reaction_rate_constant(overpotential, temperature)
            x = oxygen_fraction = self.oxygen_transport.solve(
                { 0.01 : self.gdl }, k=k / ( nO2 * F ) )

            i = self.local_current(k, x)
            h = self.proton_transport.solve(
                { 1E-16 : self.membrane }, s=i)
            protonic_potential.append( (1-df)*protonic_potential[-1] + df*h )

            overpotential = electronic_potential - protonic_potential[-1] - open_current_voltage

            # check for conv
            ratio = protonic_potential[-2] / h
            cond1 = ratio.max() < 1.01
            cond2 = ratio.min() > 0.99
            if cond1:
                break
        else:
            raise Exception("no convergence after {}".format(_))

        # at every step, we also keep track of the chosen variable
        self.heat_transport.march( i )
        self.state_history.append( i )

    def reaction_rate_constant(self, overpotential, temperature):
        ''' Butler-Volmer '''
        n = overpotential
        T = temperature
        SA = self.geometry.spheres.areas
        # go together
        ecsa = 0.1
        i0 = 1.0e-11 # A / m2
        alpha = 0.5
        # bv proper
        E1 = np.exp(  -alpha   * nO2 * F / ( R * T ) * n )
        E2 = np.exp( (1-alpha) * nO2 * F / ( R * T ) * n )
        k = i0 * ecsa * SA * ( E1 - E2 )
        return k # A

    def local_current(self, reaction_rate_constant, oxygen_fraction):
        k = reaction_rate_constant
        x = oxygen_fraction
        return k * x

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
G = 0.3 * np.ones([40, 30])
G[:3] = 0.2
G[:, 13:17] = 0.2
G = np.random.uniform(400E-9, 900E-9, [40, 10])
model = ArrayModel(G, 2 * 1000E-9)


def see_profile():
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.autoscale_view(True, True, True)
    polcurve, = ax1.plot([],[], 'kx')

    def update(V):
        model.resolve(V, 263, flood=False)

        mask = ~(model.gdl | model.membrane)

        x = model.distance_from_membrane[mask] * 1E6
        y = model.state_history[-1][mask]

        I = y.sum() / model.face_area
        ax1.plot(I, V, 'kx')

        ax2.plot(x, y, 'k')
        fig.canvas.draw()

    slider = Slider(label='V',
        ax=fig.add_axes([0.3,0.01,0.4,0.03]),
        valmin=0,
        valmax=1.2,
    )

    slider.on_changed(update)
    for V in np.linspace(0.1, 1.1, 20):
        slider.set_val(V)
    plt.show()

def see_2d():
    model.resolve(1, 263, flood=False)
    from minipnm.gui import floodview
    W = model.water_history_stack()
    S = model.state_history_stack()
    G = model.topology.asarray(model.geometry.spheres.volumes).T
    floodview(W, S, G)

see_profile()
# see_2d()
