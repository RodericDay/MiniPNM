from collections import OrderedDict, deque
import numpy as np
import minipnm as mini
import MKS
MKS.define(globals())


class ConvergenceError(Exception):
    def __iter__(self):
        '''
        memory between instances is currently shared
        '''
        for entry in Model.memory:
            yield entry


class Model():
    memory = deque(maxlen=4)

    properties = [
        'distance from membrane (um)',
        'protonic potential (V)',
        'electronic potential (V)',
        'reaction rate (m/s)',
        'oxygen molar fraction',
        'current density distribution (A/cm**2)',
    ]

    def __getitem__(self, key):
        # very sensitive to correct naming
        if '(' in key:
            name, units = key.rsplit(' ', 1)
        else: # handle dimension-less
            name, units = key, '1'
        attribute = name.replace(' ', '_')
        units = eval(units)
        values = getattr(self, attribute) / units
        return values

    # geometric stuff
    spacing = 150 * nm
    number_of_nodes = 100
    cross_sectional_area = np.pi * (15 * nm)**2
    surface_area = 4. * np.pi * (20 * nm)**2
    electrochemically_active_surface_area_ratio = 200
    # transport
    protonic_conductivity = 0.1 * S / m
    binary_diffusion_coefficient = 2.02E-5 * ( m**3 / s ) / m
    # environmental properties
    pressure = 2.27 * atm
    empirical_rate_constant = 3.44E-6 * 1 / s * spacing
    temperature = 353 * K
    # initial conditions
    reaction_rate = 0 * m / s
    oxygen_molar_fraction = 0.21

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            assert hasattr(self, key)
            setattr(self, key, value)
        n = self.number_of_nodes
        cubic = mini.Cubic([n,1,1], bbox=[n*self.spacing(m),1,1])
        x,y,z = cubic.coords
        self.gdl = x==x.max()
        self.membrane = x==x.min()
        self.distance_from_membrane = x * m

        self.solid_phase = mini.bvp.System(cubic.pairs, flux=A, potential=V)
        self.solid_phase.conductances = self.protonic_conductance
        self.gas_phase = mini.bvp.System(cubic.pairs, flux=mol/s, potential=1)
        self.gas_phase.conductances = self.diffusive_conductance

    @property
    def gas_concentration(self):
        return self.pressure / ( R * self.temperature )

    @property
    def protonic_conductance(self):
        s = self.protonic_conductivity
        A = self.cross_sectional_area
        l = self.spacing
        return s * A / l

    @property
    def diffusive_conductance(self):
        D = self.binary_diffusion_coefficient
        c = self.gas_concentration
        A = self.cross_sectional_area
        l = self.spacing
        return D * c * A / l

    @property
    def protonic_potential(self):
        i = self.current_density_distribution
        A = self.surface_area * self.electrochemically_active_surface_area_ratio
        return self.solid_phase.solve( { 0*V : self.membrane }, i*A)

    @property
    def electronic_potential(self):
        return self.gdl_voltage

    @property
    def overpotential(self):
        return self.electronic_potential - self.protonic_potential - 1.22*V

    @property
    def oxygen_concentration(self):
        return self.oxygen_molar_fraction * self.gas_concentration

    def update_reaction_rate(self):
        k0 = self.empirical_rate_constant
        n = self.overpotential
        T = self.temperature
        e = np.exp( - n * F / ( R * T ) ) #- np.exp( n * F / ( R * T ) )
        self.reaction_rate = k0 * e

    def update_oxygen_fraction(self):
        k = self.reaction_rate
        C = self.gas_concentration
        A = self.surface_area * self.electrochemically_active_surface_area_ratio
        x = self.gas_phase.solve( { 0.21 : self.gdl }, k=k*C*A )
        self.oxygen_molar_fraction = x

    @property
    def current_density_distribution(self):
        k = self.reaction_rate
        c = self.oxygen_concentration
        return k * c * F * ~(self.gdl | self.membrane)

    @property
    def current_density(self):
        return self.current_density_distribution.sum()

    def state(self):
        state = OrderedDict()
        for label in self.properties:
            state[label] = np.ones(self.number_of_nodes) * self[label]
        return state

    def steady_state(self):
        '''
        verify whether we are at steady state by iterating and comparing all
        relevant data series
        '''
        keys = ['current density distribution (A/cm**2)']

        if len(self.memory) < 2:
            raise ConvergenceError('Too few samples')
        elif all(np.allclose(self.memory[0][k], self.memory[1][k]) for k in keys):
            state = self.memory.popleft()
            self.memory.clear()
            return state
        elif len(self.memory) == 4 and \
            all(np.allclose(self.memory[0][k], self.memory[2][k]) for k in keys) and \
            all(np.allclose(self.memory[1][k], self.memory[3][k]) for k in keys):
            raise ConvergenceError('Oscillatory instability')
        else:
            raise ConvergenceError('No stability found')

    def converge(self, gdl_voltage, N=30):
        self.gdl_voltage = gdl_voltage * V
        self.memory.clear()
        for i in range(N):
            self.update_reaction_rate()
            self.update_oxygen_fraction()
            self.memory.appendleft( self.state() )
            try:
                return self.steady_state()
            except ConvergenceError as error:
                continue
        raise error

    def polarization_curve(self, span=(0, 1), samples=20, N=30):
        error = None
        currents = []
        voltages = []
        vmin, vmax = span

        for gdl_voltage in np.linspace(vmax, vmin, samples):
            try:
                state = self.converge(gdl_voltage, N)
                voltages.append( gdl_voltage )
                currents.append( state['current density distribution (A/cm**2)'].sum() )
            except ConvergenceError as error:
                # print error, 'for', gdl_voltage
                for state in error:
                    voltages.append( gdl_voltage )
                    currents.append( state['current density distribution (A/cm**2)'].sum() )

        if error is not None:
            return currents, voltages, 'x'
        else:
            return currents, voltages


if __name__ == '__main__':
    model = Model()
    mini.gui.GangedPlot(model)
