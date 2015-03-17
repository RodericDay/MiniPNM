import numpy as np
import minipnm as mini

G = np.random.uniform(400E-9, 900E-9, [20, 10])
model = mini.models.ArrayModel(G, 2 * 1000E-9)

def see_2d():
    model.resolve(0.76, 263, flood=True)
    from minipnm.gui import floodview
    W = model.water_history_stack()
    S = model.stack(model.current_history)
    G = model.topology.asarray(model.geometry.spheres.volumes).T
    floodview(W, S, G)

# model.geometry.render()
see_2d()
