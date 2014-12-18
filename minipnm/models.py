import numpy as np
import minipnm as mini
import MKS
MKS.define(globals())


class ConvergenceError(Exception):
    msg = "Failed to converge. Collected states." 


class Model():
    # geometric stuff
    spacing = 150 * nm
    number_of_nodes = 100
    area = np.pi * (20 * nm)**2
    # transport
    protonic_conductance = 1E-6 * S
    diffusive_conductance = 8E-12 * mol / s
    # environmental properties
    pressure = 1 * atm
    empirical_rate_constant = 3.44E-11 * m / s
    temperature = 333 * K
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
        k0 = self.empirical_rate_constant
        n = self.overpotential
        T = self.temperature
        e = np.exp( - n * F / ( R * T ) ) #- np.exp( n * F / ( R * T ) )
        self.reaction_rate = 0.5 * (k0 * e + self.reaction_rate)

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
        return k * c * F * ~(self.gdl | self.membrane)

    @property
    def current_density(self):
        return self.current_density_distribution.sum()/10000

    @property
    def labels(self):
        return [
            'distance from membrane (um)',
            'protonic potential (V)',
            'electronic potential (V)',
            'reaction rate (m/s)',
            'oxygen molar fraction',
            'current density distribution (A/cm**2)',
        ]

    def state(self):
        '''
        build a numpy recarray to contain all of our data
        http://docs.scipy.org/doc/numpy/user/basics.rec.html

        function as a very compact dict of arrays. obtain 'keys' via
        >>> recarray.dtype.names
        '''
        # very sensitive to correct naming
        dtype = [(l, float) for l in self.labels]
        shape = (self.number_of_nodes,)
        recarray = np.zeros(shape, dtype=dtype)
        for label in self.labels:
            if '(' in label:
                name, units = label.rsplit(' ', 1)
            else: # handle dimension-less
                name, units = label, '1'
            attribute = name.replace(' ', '_')
            units = eval(units)
            values = getattr(self, attribute) / units
            recarray[label] = values
        return recarray

    def steady_state(self):
        '''
        verify whether we are at steady state by iterating and comparing
        '''
        old_distribution = self.current_density_distribution.quantity
        self.update_reaction_rate()
        self.update_oxygen_fraction()
        new_distribution = self.current_density_distribution.quantity
        return np.allclose(old_distribution, new_distribution)

    def converge(self, gdl_voltage, N=100):
        self.gdl_voltage = gdl_voltage * V
        for i in range(N):
            if self.steady_state():
                break
        else:
            raise ConvergenceError()
        return self.state()

    def polarization_curve(self, span=(0, 1), samples=20, N=30):
        error = None
        currents = []
        voltages = []
        vmin, vmax = span

        for gdl_voltage in np.linspace(vmax, vmin, samples):
            try:
                self.converge(gdl_voltage, N)
            except ConvergenceError as error:
                currents.append( self.current_density(A/cm**2) )
                voltages.append( gdl_voltage )
                self.steady_state()
            finally:
                currents.append( self.current_density(A/cm**2) )
                voltages.append( gdl_voltage )

        if error is None:
            return currents, voltages
        else:
            error.points = [currents, voltages]
            # raise error
            return currents, voltages


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.rc('font', size=8)

    class GangedPlot(dict):
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.001)
        fig.canvas.mpl_connect('key_press_event',
            lambda event: exit() if event.key=='q' else None)

        def __init__(self, label_list):
            # profile plots
            n = len(label_list)-1
            for i, ylabel in enumerate(label_list[1:], 1):
                ax = self[ylabel] = self.fig.add_subplot(n, 2, 2*i-1)
                plt.setp( ax.get_xticklabels(), visible=False )
                ax.set_ylabel(ylabel, rotation=0)
                ax.yaxis.set_label_coords(0.5, 0.76)
                ax.set_yticks([])
            ax.set_xlabel(label_list[0])
            plt.setp( ax.get_xticklabels(), visible=True )

            self.pax = self.fig.add_subplot(1, 2, 2)
            self.pax.yaxis.tick_right()
            self.pax.yaxis.set_label_position("right")
            self.pax.set_xlabel('geometric current density (A/cm**2)')
            self.pax.set_ylabel('voltage at gdl (V)')
            self.fig.canvas.mpl_connect('button_press_event',
                lambda event: self.click(event) if self.pax is event.inaxes else None)
            self.fig.canvas.mpl_connect('motion_notify_event',
                lambda event: self.click(event) if self.pax is event.inaxes and event.button is 1 else None)
            # polarization curve

        def update(self, recarray):
            labels = recarray.dtype.names
            xlabel = labels[0]
            x = recarray[xlabel]
            for ylabel in labels[1:]:
                data = recarray[ylabel]
                ax = self[ylabel]
                ax.set_xlim(x.min(), x.max())
                if ax.lines:
                    line = ax.lines[0]
                    line.set_data(x, data)
                else:
                    ax.plot(x, data)

                # fix yticks
                yminold, ymaxold = ax.get_ylim()
                if data.ptp():
                    ymin, ymax = data.min(), data.max()
                else:
                    ymean = data.mean()
                    ymin, ymax = ymean-0.5, ymean+0.5
                ax.set_ylim(min(yminold, ymin), max(ymaxold, ymax))
                yticks = np.linspace(ymin, ymax, 7)[1:-1]
                ax.set_yticks(yticks)
            self.fig.canvas.draw()

        def click(self, event):
            voltage = event.ydata
            self.update(model.converge(voltage))

    model = Model()
    plot = GangedPlot(model.labels)

    x,y = model.polarization_curve()
    plot.pax.plot(x, y, 'x--')
    plt.show()
