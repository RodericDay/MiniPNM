import numpy as np
import minipnm as mini

g = (0.1+0.5*np.random.random() for _ in iter(int, 1))
centers, radii = mini.algorithms.poisson_disk_sampling(0.4, bbox=[9,9])
network = mini.Radial(centers, radii)
cmat = network.adjacency_matrix
cmat.data = network['sphere_radii'][cmat.row]
capacities = network['sphere_radii']**3 * 4./3. * 3.14159
source = network.indexes==0

gas = mini.simulations.Diffusion(cmat, insulated=True)
water = mini.simulations.Invasion(cmat, capacities)

def sim():
    # fill up a random amount
    for t in range(100):
        gas.march(source)
        try: water.distribute(0.2*source)
        except water.NeighborsSaturated: break

    # freeze everything at the frontier
    frontiers = water.cmat.col[water.find_frontier_throats(0)]
    targets = np.in1d(network.indexes, frontiers)
    water.block(targets)
    gas.block(targets)

    # continue
    for t in range(100):
        gas.march(source)
        water.distribute()
sim()

gui = mini.GUI()
scene = gui.scene
network.render(scene=scene)
water.render(network.points, scene=scene)
gas.render(network.points+[10,0,0], scene=scene)
gui.plot(gas.history.mean(axis=1))
gui.run()
