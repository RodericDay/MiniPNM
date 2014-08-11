import minipnm as mini

def pdf():
  i = 0
  while True:
    i = ( i + 1 ) % 2
    if    i==0: yield 1
    else: yield 0.5

network = mini.Bridson([20,20], pdf())
x,y,z = network.coords
conn = network.adjacency_matrix
lengths = (network.spans**2).sum(axis=1)**0.5
areas = network['cylinder_radii']**2 * 3.14159
conn.data = areas
history = mini.algorithms.invasion(conn, x==x.min())

gui = mini.GUI()
network.render(gui.scene, saturation_history=history)
gui.plotXY(range(len(history)), (network['sphere_radii']**2*history).sum(axis=1))
gui.run()
