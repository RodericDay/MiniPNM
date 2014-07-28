import numpy as np
import minipnm as mini

noise = mini.image.gaussian_noise([50,20])
network = mini.Cubic(noise, noise.shape)
x,y,z = network.coords
left = x < np.percentile(x,10)
right = x > np.percentile(x,90)
oxygen_dirichlet = {0 : left, 1 : right}

proton = x.max() - x
floodable = network.copy()
flood_history, activity, oxygen = [], [], []
for i in range(300):
    if i == 0:
        is_water = np.zeros(network.order, dtype=bool)
    else:
        is_water = flood_history[-1].copy()
        new = np.random.choice((~is_water).nonzero()[0], 3)
        is_water[new] = True

    floodable.prune(is_water)
    diffused = mini.algorithms.bvp.solve(floodable.laplacian, oxygen_dirichlet)
    activity_rate = (proton * diffused)**3

    flood_history.append(is_water)
    activity.append(activity_rate)
    oxygen.append(diffused)


activity_layer = network.copy()
activity_layer['z'] = z + 20

oxygen_layer = network.copy()
oxygen_layer['z'] = z + 40

gui = mini.GUI()
scene = gui.scene
network.render(scene, values=flood_history)
activity_layer.render(scene, values=activity, cmap='hot')
oxygen_layer.render(scene, values=oxygen, cmap='Greens_r')
gui.plotXY(range(300), [sum(a) for a in activity])
gui.run()