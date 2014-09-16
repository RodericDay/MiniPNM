from collections import defaultdict
import numpy as np
import minipnm as mini

network = mini.Cubic([10,10])
x,y,z = network.coords

RHS = network.system(-0.1, 1+4*0.1)
LHS = network.system( 0.1, 1-4*0.1)
def diffuse_oxygen(sinks=None, obstacles=None):
    A = RHS
    b = LHS * history.get('oxygen_concentration',[0])[-1]
    s = mini.algorithms.bvp.spsolve(A, b)
    return s

proton_concentration = x.max()-x

def diffuse_heat(sources):
    return x

def distribute_water(additions):
    return np.ones(network.order)

def form_ice(water, temperature):
    return np.in1d(network.indexes, np.random.choice(network.indexes, 10))

k1 = 1
k2 = 1
dt = 0.001
history = defaultdict(list)
def march(time):
    oxygen_concentration = diffuse_oxygen()
    activity = proton_concentration * oxygen_concentration
    temperature = diffuse_heat(k1*activity)
    water = distribute_water(k2*activity)
    ice = form_ice(water, temperature)
    for key, value in locals().items():
        history[key].append(value)

for t in (round(i*dt,3) for i in range(10)):
    march(t)

for key, value in history.items():
    history[key] = np.array(value)

gui = mini.GUI()

gas_actor = mini.graphics.Spheres(network.points, history['oxygen_concentration']*0.5, alpha=0.25)
gas_actor.mapper.ScalarVisibilityOn()
gas_actor.glyph3D.ScalingOff()
temp_actor = mini.graphics.Wires(network.points, network.pairs, history['temperature'])
water_actor = mini.graphics.Spheres(network.points, history['water']*0.3, color=(0,0,1), alpha=0.5)
ice_actor = mini.graphics.Spheres(network.points, history['ice']*0.4)
t, y = history['time'], np.sum(history['oxygen_concentration'], axis=1)

gui.scene.add_actors([gas_actor, water_actor, ice_actor])
gui.plot(t, y)
gui.run()
