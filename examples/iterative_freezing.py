import numpy as np
from scipy import spatial
import minipnm as mini

points = np.random.rand(2000,3)
points.T[2] = 0
network = mini.Delaunay(points)
network = network - network.boundary()
x,y,z = network.coords
# fix some "physical" properties
network['indexes'] = network.indexes # to recover positions
network['source'] = y < np.percentile(y,5)
network['sink'] = y > np.percentile(y,98)
network['thresholds'] = np.random.rand(x.size)

# start freezing
sat_history = []
ghost = np.zeros_like(x)
frozen = [np.zeros_like(x, dtype=bool)]
clone = network.copy()
for i in range(5):
    source = clone['source']
    thresholds = clone['thresholds']
    sink = clone['sink']
    saturation = mini.invasion(clone, source, thresholds, sink)

    # for vis
    coloured = saturation*(i+3)
    # set ghost as backdrop ie: if the value is zero but ghost can replace it, do it
    coloured[coloured==0] = np.vstack([ghost]*len(coloured))[coloured==0]

    sat_history.append(coloured)
    ghost = np.vstack(sat_history).max(axis=0)*0.7

    # freeze the smallest 20% of the filled pores
    filled = saturation[-1]
    frozen.append( filled & (thresholds < np.percentile(thresholds[filled], 20)) | frozen[-1])
    clone.prune(frozen[-1], remove_pores=False)

frozen = np.vstack([[f]*len(s) for s,f in zip(sat_history, frozen)])[::10]
sat_history = np.vstack(sat_history)[::10]

scene = mini.Scene()
scene.add_wires(network.points, network.pairs, sat_history)
scene.add_spheres(network.points, frozen*0.01, color=(1,1,1))
scene.play()