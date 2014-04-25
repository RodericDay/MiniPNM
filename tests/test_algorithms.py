import os
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'binary'

import minipnm as mini

def test_shortest_paths():
    dirname = os.path.dirname(__file__) + '/resources/'
    filename= 'matrix.txt'

    R = np.loadtxt(dirname+filename, delimiter=',')
    R = R.reshape(R.shape+(1,))
    network = mini.Cubic(R)
    network = network - (R.flatten() < R.mean())

def test_linear_solve():
    # R = np.random.rand(30, 30, 10)
    R = np.random.rand(100,150,1)
    N = np.zeros_like(R)
    for i in 2**np.arange(6):
        N += ndimage.filters.gaussian_filter(R, i)
    N -= N.min()
    N /= N.max()

    network = mini.Cubic(N, N.shape)
    network = network - (N.ravel() > 0.45)

    x,y,z = network.coords
    left = x==x.min()
    right = x==x.max()

    ics = 1*left + 0.01*right
    sol = mini.solve_linear(network, ics)
    network.render(sol)
