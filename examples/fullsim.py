import numpy as np
import minipnm as mini

## Creation of a network ####
cubic = mini.Cubic([10,10])
rmin = 0.1
radii = np.random.rand(cubic.order)*(0.5-rmin)+rmin
capacities = 4./3. * np.pi * radii**3
radial = mini.Radial(cubic.points, radii, cubic.pairs)
conductance_matrix = radial.adjacency_matrix
conductance_matrix.data = 1./radial.lengths
###

tgrid = radial.rasterize(20)
correspondence = mini.match(radial.points, tgrid.points)
x,y,z = tgrid.coords
top = y==y.max()
bottom = y==y.min()
right = x==x.min()
left = x==x.max()

water_generator = (0.01*(radial['x']==0) for _ in iter(int, 1))
heat_generator = (20*(tgrid['x']==tgrid['x'].min())*1./(t+1) for _ in iter(int, 1))

# oxygen = mini.simulations.Diffusion()
water = mini.simulations.Invasion(capacities, conductance_matrix)
temperature = mini.simulations.Diffusion(tgrid.adjacency_matrix, u=0.1, insulated=top|bottom|left|right, T0=-10)
# high res latticed

def freeze(saturation, temperature):
    full = saturation>0.999
    cold = temperature<0
    return full & cold & (radial['x']!=0)

for t in range(200):
    try:
        water.distribute(next(water_generator))
    except mini.simulations.NeighborsSaturated as e:
        print(e)
        break
    temperature.march(next(heat_generator))
    frozen = freeze(water.state, temperature.state[correspondence])
    water.block(frozen)

gui = mini.GUI()
radial.render(scene=gui.scene)
water.render(radial.points, scene=gui.scene)
tgrid.render(temperature.history, alpha=0.5, scene=gui.scene)
gui.plot(np.mean(temperature.history, axis=1))
gui.run()
