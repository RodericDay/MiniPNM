import numpy as np
import minipnm as mini

cubic = mini.Cubic([20,1,20], scaling=8E-6)
x,y,z = cubic.coords

F = 96487
R = 8.314

class Model(object):

    properties = [
        'protonic_potential',
        'oxygen_mole_fraction',
        'local_current',
    ]

    distance_from_membrane = z
    open_current_voltage = 1.223 # V
    oxygen_ratio = 4
    ecsa_ratio = 20
    surface_area = 1.936E-9

    noise = np.random.rand(cubic.size) * 0
    blocked_pores = np.random.rand(cubic.order) < 0.1
    open_throats = ~cubic.cut(blocked_pores, directed=False)

    oxygen_transport = mini.bvp.System(cubic.pairs)
    oxygen_transport.conductances = 3.769E-8 * (1+noise) * open_throats

    proton_transport = mini.bvp.System(cubic.pairs)
    proton_transport.conductances = 0.00235592 * (1+noise)

    @property
    def reaction_rate_constant(self):
        T = self.temperature
        I0_cat = 1.0e-11 #A/m2
        As = self.ecsa_ratio
        A = self.surface_area # m2
        n = self.overpotential
        z = self.oxygen_ratio
        # Butler-Volmer
        alpha = 0.5
        k_cat = I0_cat*As*(np.exp(-alpha*z*F/(R*T)*n) - np.exp((1-alpha)*z*F/(R*T)*n))
        return k_cat/(z*F)

    def update_oxygen(self):
        dbcs = { 0.01 : z==z.max() }
        k = self.reaction_rate_constant
        self._x = self.oxygen_transport.solve(dbcs, k=k)

    def update_overpotential(self):
        dbcs = { 0 : z==z.min() }
        s = self.local_current
        self._h = self.proton_transport.solve(dbcs, s=s)

    def check_convergence(self):
        # Find new eta
        self.protonic_potential_old = self.electronic_potential - self.overpotential - self.open_current_voltage
        self.protonic_potential_new = self._h

        # Damped evolution
        df = 0.1*self.cell_voltage
        self.protonic_potential = self.protonic_potential_old + df*(self.protonic_potential_new - self.protonic_potential_old)
        self.overpotential = self.electronic_potential - self.protonic_potential - self.open_current_voltage

        self.ratio = self.protonic_potential_old/self.protonic_potential_new
        cond1 = self.ratio.max() < 1.01
        cond2 = self.ratio.min() > 0.99
        convergence = cond1 # and cond2
        return convergence

    @property
    def temperature(self):
        return 353

    @property
    def electronic_potential(self):
        return np.copy(self.cell_voltage)

    @property
    def oxygen_mole_fraction(self):
        return self._x

    @property
    def local_current(self):
        k = self.reaction_rate_constant
        x = self.oxygen_mole_fraction
        r = self.oxygen_ratio
        return k*x*r*F * ~(z==z.max())

    def resolve(self, cell_voltage, overpotential=-0.02):
        self.overpotential = overpotential
        self.cell_voltage = cell_voltage
        for outer_loop_count in range(0,500):
            if outer_loop_count%10==0: print '.',
            self.update_oxygen()
            self.update_overpotential()
            if self.check_convergence():
                print
                return
        print
        # raise RuntimeError("No convergence")

    def polarization_curve(self, voltage_range=[0.5]):
        V = []
        I = []
        for V_cell in reversed(voltage_range):
            V.append(V_cell)
            self.resolve(V_cell)
            I.append( self.local_current.sum() / self.surface_area )
        return I, V

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

model = Model()
fig, ax = plt.subplots(len(model.properties)+1)

X, Y = model.polarization_curve(np.linspace(0.4,0.9,5))
fig.subplots_adjust(right=0.5)
pax = fig.add_axes([0.6, 0.1, 0.3, 0.8])
pax.plot(X, Y)

def update(V_cell):
    model.resolve(V_cell)
    x = model.distance_from_membrane
    for i, attribute in enumerate(model.properties):
        y = getattr(model, attribute)
        # ax[i].matshow(cubic.asarray(y))
        ax[i].clear()
        ax[i].plot(x, y, 'bgrcmk'[i]+'x')
    fig.canvas.draw()
sld = Slider(ax[-1], 'V_cell', 0.1, 1.1, 0.9)
sld.on_changed(update)
update(0.9)

plt.show(); exit()
