import numpy as np
import matplotlib.pyplot as plt
import minipnm as mini
import MKS
MKS.define(globals())

# definitions
SSA = 5E5 * m**2 / m**3
korr = 3.44E18 / s
E0 = 1.22 * V
O2_c = 7.35 * mol / m**3
E_ch = 0.68 * V
P_ch = -0.165 * V
E_s = 1000 * S / m * 1E-18
P_s = 1000 * S / m * 1E-19
T = 353 * K
P = 1.5E5 * Pa
alf = 0.5
j0 = 1.8E-2 * A / m**2
D_b = 2.02E-5 * m**2 / s * 0.01
r = 25E-9 * m # assume an average pore radii

# network-building
topology = mini.Cubic([31,1,1], r(m)*3)
network = mini.Radial(topology.points, r(m), topology.pairs)
t, h, w = network.bbox * m

x, y, z = topology.coords
left = x==x.min()
right = x==x.max()

# simple derivations and definitions
c = P / ( R * T )
wire_lengths = topology.lengths * m
areas = network.areas * m**2
volumes = network.volumes * m**3
agg_areas = (t * h * w) * SSA * areas / areas.sum()

def butler_volmer(oxygen, overpotential):
    E1 = np.exp(   2 *   alf   * F / (R * T) * overpotential )
    E2 = np.exp( - 2 * (1-alf) * F / (R * T) * overpotential )
    AO2 = oxygen.quantity
    j_orr = j0 * AO2**0.25 * ( E1 - E2 )
    return j_orr

def resolve(i):
    oxygen_molar_flux = i / ( 4 * F )
    o_cmat = topology.system(-c * D_b / wire_lengths, units= mol / m**2 / s)
    oxygen_molar_fraction = mini.bvp.solve(o_cmat, { O2_c / c : right }, -oxygen_molar_flux, units=1)
    oxygen_concentration = oxygen_molar_fraction * c

    e_cmat = topology.system(m**2 * E_s / wire_lengths, units=S)
    electron_potential = mini.bvp.solve(e_cmat, { E_ch : right }, agg_areas*i, units=V)

    p_cmat = topology.system(m**2 * P_s / wire_lengths, units=S)
    proton_potential = mini.bvp.solve(p_cmat, {P_ch : right }, -agg_areas*i, units=V)

    overpotential = electron_potential - proton_potential - E0
    iout = butler_volmer(oxygen_concentration, overpotential)

    plot(**locals())

def plot(**caught):
    globals().update(caught)
    fig, axs = plt.subplots(3)
    axs[0].axhline(O2_c(mol/m**3), c='g')
    axs[0].plot(oxygen_concentration(mol/m**3), 'g.--')
    axs[0].set_ylabel(O2_c.label)

    axs[1].axhline(E_ch(V), c='y')
    axs[1].plot(electron_potential(V), 'y.--')
    axs[1].axhline(P_ch(V), c='b')
    axs[1].plot(proton_potential(V), 'b.--')
    axs[1].set_ylabel(E_ch.label)

    axs[2].plot(i(A/m**2), 'r')
    axs[2].plot(iout(A/m**2), 'm')
    axs[2].set_ylabel(i.label)

    plt.xlabel( 'along thickness from MEA to GDL' )
    plt.show()

mA = 1E-3 * A
cm = 1E-2 * m
I_guess = 890 * mA / cm ** 2
I_total = I_guess * h * w
I_perpore = I_total * network.areas / network.areas.sum()
I_dens_local = I_perpore / agg_areas

resolve(I_dens_local)
