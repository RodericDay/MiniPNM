import numpy as np
import minipnm as mini

output = 200.

network = mini.Cubic([50,20])

# pts = np.random.rand(1000, 3)
# pts.T[0] *= 50
# pts.T[1] *= 20
# pts.T[2] *= 2
# network = mini.Delaunay(pts)

max_out = output / network.order * 5
x,y,z = network.coords
left = x < np.percentile(x,5)
right = x > np.percentile(x,95)
oxygen_dirichlet = {1 : right}

proton = (x.max() - x)
floodable = network.copy()
flood_history, activity, oxygen = [], [], []
is_water = np.zeros(network.order, dtype=bool)
while not all(is_water):
    floodable.prune(is_water)

    C = floodable.laplacian
    C[range(floodable.order), range(floodable.order)] += 0.00001 * proton
    A, b = mini.algorithms.bvp.build(C, oxygen_dirichlet)
    diffused = mini.algorithms.bvp.spsolve(A, b).round(5)

    activity_rate = proton * diffused
    activity_rate*= (200. / sum(activity_rate))
    activity_rate = activity_rate.clip(0,max_out)

    flood_history.append(is_water)
    activity.append(activity_rate)
    oxygen.append(diffused)

    new = np.random.choice((~is_water).nonzero()[0], min(1,network.order//100))
    is_water = flood_history[-1].copy()
    is_water[new] = True



activity_layer = network.copy()
activity_layer['z'] = z + 20

oxygen_layer = network.copy()
oxygen_layer['z'] = z + 40

gui = mini.GUI()
scene = gui.scene
network.render(scene, values=flood_history)
activity_layer.render(scene, values=activity, cmap='hot')
oxygen_layer.render(scene, values=oxygen, cmap='Greens_r')
gui.plotXY(range(len(activity)), [sum(a) for a in activity])
gui.run()