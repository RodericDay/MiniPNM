import numpy as np
import minipnm as mini
import matplotlib.pyplot as plt
import MKS
MKS.define(globals())

T = 333*K
P = 1 * atm
C = P / ( R * T )
area = np.pi * (20 * nm)**2
k0 = 3.44E-11 * m / s

cubic = mini.Cubic([100,1,1], bbox=[1,1,1])
x,y,z = cubic.coords
gdl = x==x.max()
membrane = x==x.min()
# for protons
sC = 1E-12 * S / cubic.lengths**2
solid_phase = mini.bvp.System(cubic.pairs, flux=A, potential=V, conductances=sC)
# for gas
gC = 1E-19 * mol/s / cubic.lengths**2
gas_phase = mini.bvp.System(cubic.pairs, flux=mol/s, potential=1, conductances=gC)

class ConvergenceError(Exception):
    pass

def solve(v, N=100):
    i = 0 * A / m**2
    for _ in range(N):
        p = solid_phase.solve( { 0*V : membrane }, f*i*area)
        o = v - p - 1.22*V
        k = k0 * np.exp( - o * F / ( R * T ) )
        x = gas_phase.solve( { 0.21 : gdl }, k=k*C*area )
        c = x * C
        i = c * k * F

        if _ > N-3:
            yield (~gdl*i)(A/cm**2)

def polcurve():
    currents, voltages = [], []
    for v in np.linspace(0.4, 0.95, 30) * V:
        i1, i2 = solve(v)
        currents.extend( [i1.sum(), i2.sum()] )
        voltages.extend( [v(V), v(V)] )

    plt.xlabel('A/cm2')
    plt.ylabel(V.label)
    plt.plot(currents, voltages, 'bo' if f else 'g-')

def profile(v, s, N=30, q='i'):
    try:
        i = solve(v, q, N).quantity
        plt.plot(x, i, s)
        print i.sum()
    except ConvergenceError as e:
        global plot_polcurve
        plot_polcurve = False
        print 'failed at', v
        plt.plot(x, e.y1, s+'x--')
        print e.y1.sum()
        plt.plot(x, e.y2, s+'x--')
        print e.y2.sum()

f=0
polcurve()
f=1
polcurve()

# rob data
i0 = np.array([0.0001, 0.0302, 0.0502, 0.0601, 0.0996, 0.2002, 0.5994, 0.9989, 1.1990, 1.4985, 1.6978])
v0 = np.array([0.9478, 0.8704, 0.8551, 0.8502, 0.8315, 0.7865, 0.7418, 0.7116, 0.7007, 0.6839, 0.6550])
plt.plot(i0, v0, 'm.-')
plt.show()
