import numpy as np
import matplotlib.pyplot as plt
import minipnm as mini
import MKS
MKS.define(globals())

# network-building
thickness = 15 * um
SSA = 200 * m**2 / m**2
r = 25E-9 * m # assume an average pore radii
eN = 10 * nm

topology = mini.Cubic([10,1,1], r(m)*3)
network = mini.Radial(topology.points, r(m), topology.pairs)
t, h, w = network.bbox * m
x, y, z = topology.coords
left = x==x.min()
right = x==x.max()
r_c = network['cylinder_radii'] * m
carbon_area = np.pi * r_c**2
nafion_area = np.pi * ((r_c+eN)**2 - r_c**2)

# simple derivations
depth = x * m
wire_lengths = topology.lengths * m
areas = network.spheres.areas * m**2
volumes = network.spheres.volumes * m**3
total_agg_area = (h * w) * SSA
agg_areas = total_agg_area * areas / areas.sum()

# lit defs
E0 = 1.223 * V
E_ch = 0.68 * V
P_ch = -0.165 * V
alf = 0.5
j0 = 1.8E-2 * A / m**2 / 100
O2_x = 0.21

# env defs
T = 353 * K
P = 1.5E5 * Pa
c = P / ( R * T )
O2_c = O2_x * c

# material defs
E_s = 1000 * S / m
# P_s = 100 * np.exp( ( 15.036 * 1 - 15.811) * 1000*K/T + (-30.726 * 1 + 30.481) ) * S / m
P_s = 10 * S / m
D_b = 2.02E-5 * m**2 / s * 0.01

o_cmat = topology.system(-c * D_b / wire_lengths, units= mol/m**2/s)
e_cmat = topology.system(carbon_area * E_s / wire_lengths, units=S)
p_cmat = topology.system(nafion_area * P_s / wire_lengths, units=S)

# some current stuff
I_guess = 100000 * mA / cm ** 2
I_total = I_guess * h * w
I_perpore = I_total * network.spheres.areas / network.spheres.areas.sum()
I_dens_local = I_perpore / agg_areas

def butler_volmer(oxygen, overpotential):
    E1 = np.exp(   2 *   alf   * F / (R * T) * overpotential )
    E2 = np.exp( - 2 * (1-alf) * F / (R * T) * overpotential )
    AO2 = 1
    j_orr = -j0 * AO2**0.25 * ( E1 - E2 )
    return j_orr

def resolve(i):
    oxygen_molar_flux = i / ( 4 * F )
    oxygen_molar_fraction = mini.bvp.solve(o_cmat, {O2_x: right}, -oxygen_molar_flux, units=1)
    oxygen_concentration = oxygen_molar_fraction * c

    electron_potential = mini.bvp.solve(e_cmat, { E_ch : right }, agg_areas*i, units=V)

    proton_potential = mini.bvp.solve(p_cmat, { 0*V : left }, agg_areas*i, units=V)

    overpotential = electron_potential - proton_potential - E0
    iout = butler_volmer(oxygen_concentration, overpotential)

    globals().update(locals())
    return iout

def plot():
    fig, axs = plt.subplots(3, sharex=True)
    axs[0].set_title( topology.asarray(x).shape )

    axs[0].axhline(O2_c(mol/m**3), c='g')
    axs[0].plot(depth(um), oxygen_concentration(mol/m**3), 'g.--')
    axs[0].set_ylabel(O2_c.label)
    axs[0].set_ylim(0, 20)

    axs[1].axhline(E_ch(V), c='y')
    axs[1].plot(depth(um), electron_potential(V), 'y.--')
    axs[1].axhline(P_ch(V), c='b')
    axs[1].plot(depth(um), proton_potential(V), 'b.--')
    axs[1].set_ylabel(E_ch.label)

    axs[2].plot(depth(um), i(A/m**2), 'rx')
    axs[2].plot(depth(um), iout(A/m**2), 'm')
    axs[2].set_ylabel(i.label)

    plt.xlabel( 'distance along thickness from MEA to GDL [um]' )
    plt.show()

# find fixed point
from scipy import optimize

x0 = I_dens_local(A/m**2)
objf = lambda x: resolve(x * A/m**2)(A/m**2)
minf = lambda x: np.sum( (x - objf(x))**2 )

ans = optimize.minimize(minf, x0).x * A/m**2
resolve(ans)
plot()
exit()

# plolcurve
for E_ch in np.linspace(0.5, 0.6, 20) * V:
    x0 = optimize.minimize(minf, x0).x

    sol = (x0 * A/m**2 * agg_areas).sum()
    sol = sol / (w * h)
    plt.scatter( sol(A/m**2), E_ch(V) )

plt.show()
