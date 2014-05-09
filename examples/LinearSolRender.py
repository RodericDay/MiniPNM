import minipnm as mini

N = mini.gaussian_noise([100,100,5])
network = mini.Cubic(N, N.shape)
network = network - (N > N.mean())
x,y,z = network.coords
dbcs = 3*(x==x.min()) + 1*(x==x.max())
sol = mini.linear_solve(network, dbcs)
network.render(sol, alpha=1)