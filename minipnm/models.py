import numpy as np
import minipnm as mini
import MKS
MKS.define(globals())


class ConvergenceError(Exception):
    msg = "Failed to converge at {:.3f} V"
    def __init__(self, voltage):
        self.voltage = voltage.quantity
    def __str__(self):
        return self.msg.format(self.voltage)


class Model():
    # geometric stuff
    spacing = 150 * nm
    n = 100
    cubic = mini.Cubic([n,1,1], bbox=[spacing(m),1,1])
    x,y,z = cubic.coords
    gdl = x==x.max()
    membrane = x==x.min()
    area = np.pi * (20 * nm)**2
    
    # transport
    sC = 1E3 * S
    solid_phase = mini.bvp.System(cubic.pairs, flux=A, potential=V, conductances=sC)
    gC = 8E-12 * mol / s
    gas_phase = mini.bvp.System(cubic.pairs, flux=mol/s, potential=1, conductances=gC)

    # environmental properties
    pressure = 1 * atm
    k0 = 3.44E-11 * m / s

    # initial conditions
    reaction_rate = k = 0 * m / s
    oxygen_molar_fraction = 0.21

    @property
    def gas_concentration(self):
        return self.pressure / ( R * self.temperature )

    @property
    def temperature(self):
        return 333*K

    @property
    def protonic_potential(self):
        i = self.current_density_distribution
        A = self.area
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
        n = self.overpotential
        T = self.temperature
        e = np.exp( - n * F / ( R * T ) ) #- np.exp( n * F / ( R * T ) )
        self.reaction_rate = self.k0 * e

    def update_oxygen_fraction(self):
        k = self.reaction_rate
        C = self.gas_concentration
        A = self.area
        x = self.gas_phase.solve( { 0.21 : self.gdl }, k=k*C*A )
        self.oxygen_molar_fraction = x

    @property
    def current_density_distribution(self):
        k = self.reaction_rate
        c = self.oxygen_concentration
        return k * c * F

    @property
    def current_density(self):
        return (~self.gdl * self.current_density_distribution).sum()/10000

    def steady_state(self):
        old_distribution = self.current_density_distribution.quantity
        self.update_reaction_rate()
        self.update_oxygen_fraction()
        new_distribution = self.current_density_distribution.quantity
        return np.allclose(old_distribution, new_distribution)

    def converge(self, N):
        for i in range(N):
            if self.steady_state():
                break
        else:
            raise ConvergenceError(self.gdl_voltage)

    def polarization_curve(self, span=(0, 1), samples=10, N=30):
        error = None
        currents = []
        voltages = []
        vmin, vmax = span

        for gdl_voltage in np.linspace(vmax, vmin, samples) * V:
            self.gdl_voltage = gdl_voltage
            try:
                self.converge(N)
            except ConvergenceError as error:
                print( error )
                currents.append( self.current_density(A/cm**2) )
                voltages.append( gdl_voltage(V) )
                self.steady_state()
            finally:
                currents.append( self.current_density(A/cm**2) )
                voltages.append( gdl_voltage(V) )

        if error is None:
            return currents, voltages
        else:
            error.points = [currents, voltages]
            raise error

    def profile(self, attrib, gdl_voltage, N=30):
        self.gdl_voltage = gdl_voltage * V
        self.converge(N)

        x = self.d
        y = getattr(self, attrib)
        y = y.quantity if hasattr(y, 'units') else y
        return x, y


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    model = Model()

    try:
        x, y = model.polarization_curve(samples=20)
        plt.plot(x, y, 'x-')
    except ConvergenceError as error:
        x, y = error.points
        plt.plot(x, y, 'bo')
    

    plt.show()
