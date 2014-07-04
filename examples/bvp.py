import minipnm as mini


network = mini.Delaunay.random(100)
o, i = network.split(network.boundary())
network = o | i
# network.render()

x,y,z = network.coords
dirichlet = { 0 : x == x.min(), 1 : x == x.max() }
x = mini.solve_bvp(network.laplacian, dirichlet)
network.render(x)