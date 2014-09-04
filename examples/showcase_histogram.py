import time
import numpy as np
import matplotlib.pyplot as plt
import minipnm as mini

arr1 = np.random.normal(size=10000)+2
arr2 = np.random.normal(size=5000)+7
arr = np.hstack([arr1, arr2])
arr = arr[arr>3]/30

g = (np.random.choice(arr) for _ in iter(int,1))
t0 = time.time()
centers, radii = mini.algorithms.poisson_disk_sampling(r=g, bbox=[10,10])
tt = time.time()-t0
print tt

network = mini.Radial(centers, radii)
print network
x,y,z = network.coords
hist = mini.algorithms.invasion(network.adjacency_matrix, x==x.min())
gui = mini.GUI()
network.render(hist, scene=gui.scene)
gui.run()

plt.hist(arr, bins=50, alpha=0.5, normed=True)
plt.hist(network['sphere_radii'], bins=50, alpha=0.5, normed=True)
plt.title("{} pores generated in {:.1f}s".format(network.order, tt))
# plt.show()
