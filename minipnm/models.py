from collections import OrderedDict, deque
import numpy as np
import minipnm as mini
import MKS
MKS.define(globals())


class ConvergenceError(RuntimeError):
    pass # this guy should offer some easily examinable insight
    # into what went wrong

class OscillationError(ConvergenceError):
    pass

class AbstractModel():
    # save last four states in a deque to check for oscillations
    memory = deque(maxlen=4)

    def __init__(self):
        raise NotImplementedError()

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

    @property
    def state(self):
        state = OrderedDict()
        for label in self.properties:
            state[label] = self[label]
        return state


class Model(AbstractModel):

    # first property assumed to be x-axis in profiles
    properties = [
        'distance from membrane (um)',
        'protonic potential (V)',
        # 'electronic potential (V)',
        # 'reaction rate constant (m/s)',
        'oxygen molar fraction',
        'current density distribution (A/cm**2)',
    ]

    # geometric stuff
    spacing = 150 * nm
    number_of_nodes = 100
    cross_sectional_area = np.pi * (15 * nm)**2
    surface_area = 4. * np.pi * (20 * nm)**2
    electrochemically_active_surface_area_ratio = 20
    # transport
    protonic_conductivity = 0.1 * S / m
    binary_diffusion_coefficient = 2.02E-5 * ( m**3 / s ) / m
    # environmental properties
    pressure = 2.27 * atm
    empirical_rate_constant = 3.44E-6 / s * spacing
    temperature = 353 * K
    # initial conditions
    reaction_rate_constant = 0 * m / s
    oxygen_molar_fraction = 0.21

    def __init__(self, **kwargs):
        # only allow changing variables if they have been defined
        for key, value in kwargs.items():
            assert hasattr(self, key)
            setattr(self, key, value)

        # create geometry
        n = self.number_of_nodes
        l = n * self.spacing / m
        w = self.spacing / m
        h = self.spacing / m
        cubic = mini.Cubic([n,1,1], bbox=[l,h,w])
        x,y,z = cubic.coords
        self.gdl = x==x.max()
        self.membrane = x==x.min()
        self.distance_from_membrane = x * m

        # create transport mechanisms
        self.solid_phase = mini.bvp.System(cubic.pairs, flux_units=A, potential_units=V)
        self.solid_phase.conductances = self.protonic_conductance
        self.gas_phase = mini.bvp.System(cubic.pairs, flux_units=mol/s, potential_units=1)
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
        return self.solid_phase.solve( { 0*V : self.membrane }, s=2E-14 * (i*A).units )
        # return self.solid_phase.solve( { 0*V : self.membrane }, s=i*A )

    @property
    def electronic_potential(self):
        return self.gdl_voltage

    @property
    def overpotential(self):
        return self.electronic_potential - self.protonic_potential - 1.223*V

    @property
    def oxygen_concentration(self):
        return self.oxygen_molar_fraction * self.gas_concentration

    def update_reaction_rate(self):
        k0 = self.empirical_rate_constant
        n = self.overpotential
        T = self.temperature
        z = 4
        a = 0.5
        e = np.exp( - a*z*F*n / ( R*T ) ) - np.exp( (1-a)*z*F*n / ( R*T ) )
        self.reaction_rate_constant = k0 * e

    def update_oxygen_fraction(self):
        k = self.reaction_rate_constant
        C = self.gas_concentration
        A = self.surface_area * self.electrochemically_active_surface_area_ratio
        x = self.gas_phase.solve( { 0.21 : self.gdl }, k=k*C*A )
        self.oxygen_molar_fraction = x

    @property
    def current_density_distribution(self):
        k = self.reaction_rate_constant
        c = self.oxygen_concentration
        return k * c * F * ~(self.gdl | self.membrane)

    @property
    def current_density(self):
        return self.current_density_distribution.sum()

    @property
    def steady_state(self):
        '''
        verify whether we are at steady state by iterating and comparing all
        relevant data series in memory

        there are three possible outcomes
        - convergence
        - no solution
        - oscillation
        '''
        keys = ['current density distribution (A/cm**2)']

        if len(self.memory) < 2:
            return False

        elif all(np.allclose(self.memory[0][k], self.memory[1][k]) for k in keys):
            state = self.memory.popleft()
            self.memory.clear()
            return True

        elif len(self.memory) == 4 and \
            all(np.allclose(self.memory[0][k], self.memory[2][k]) for k in keys) and \
            all(np.allclose(self.memory[1][k], self.memory[3][k]) for k in keys):
            raise OscillationError()

    def resolve(self, gdl_voltage, N=30):
        self.gdl_voltage = gdl_voltage * V
        self.memory.clear()
        for self.state_resolution_iteration_number in range(N):
            self.update_reaction_rate()
            self.update_oxygen_fraction()
            self.memory.appendleft( self.state )
            if self.steady_state:
                return self.state
        raise ConvergenceError("No stability found")

    def polarization_curve(self, span=(0, 1), samples=20, N=30):
        error = None
        currents = []
        voltages = []
        vmin, vmax = span

        for gdl_voltage in np.linspace(vmax, vmin, samples):
            try:
                state = self.resolve(gdl_voltage, N)
                voltages.append( gdl_voltage )
                current_output = state['current density distribution (A/cm**2)']
                currents.append( current_output.sum() )
            except ConvergenceError:
                for state in self.memory:
                    voltages.append( gdl_voltage )
                    current_output = state['current density distribution (A/cm**2)']
                    currents.append( current_output.sum() )

        return currents, voltages
