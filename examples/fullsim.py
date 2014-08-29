import numpy as np
import minipnm as mini

network = mini.Cubic([30,60])
x,y,z = network.coords
left = x==x.min()
right = x==x.max()
tophalf = y>y.mean()
dbcs = {0: left, 1:right&tophalf}

time = np.linspace(0,1,100)
oxygen, protons, activity, vapor, water, ice = [], [], [], [], [], []
for t in time:
    if len(water) == 0:
        water.append(np.ones(network.order, dtype=bool))

    flooded = ~water[-1]
    pruned = network - flooded
    sol = mini.algorithms.bvp.solve(pruned.laplacian, dbcs)
    oxygen.append(sol)

    # activity is flux
    adj = pruned.adjacency_matrix
    adj.data = abs(sol[adj.col] - sol[adj.row])
    rate = 1E5*adj.mean(axis=1).A1
    activity.append(rate)

    # temperature depends on rate

    # generate vapor proportional to activity
    vapor.append(vapor[-1]+rate if vapor else rate)

    # water fill step- currently req'd for interesting results
    candidates = (~flooded).nonzero()[0]
    if any(candidates):
        chosen = np.random.choice(candidates, 10)
        flooded[chosen] = True
    water.append(~flooded)

    # ice freeze step
    ice.append( flooded*rate )

gui = mini.GUI()
network.render(gui.scene, oxygen, cmap='Greens_r')
network['x'] += 40
network.render(gui.scene, activity, cmap='copper')
# network['x'] += 40
# network.render(gui.scene, temperature, cmap='hot')
network['x'] += 40
network.render(gui.scene, vapor)
network['x'] += 40
network.render(gui.scene, water)
network['x'] += 40
network.render(gui.scene, ice)
gui.plotXY(time, np.max(activity, axis=1))
gui.run()
