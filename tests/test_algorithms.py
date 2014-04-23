import os
import numpy as np

import minipnm as mini

dirname = os.path.dirname(__file__) + '/resources/'
filename= 'matrix.txt'

R = np.loadtxt(dirname+filename, delimiter=',')
R = R.reshape(R.shape+(1,))
network = mini.Cubic(R)
network = network - (R.flatten() < R.mean())
network.preview()