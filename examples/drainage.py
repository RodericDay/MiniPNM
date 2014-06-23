import numpy as np
import matplotlib.pyplot as plt
import minipnm as mini

''' 
known in the field as "imbibition"...
'''

np.random.seed(44)
points = np.random.rand(200,3)
points.T[2] = 0
points = np.vstack(sorted(points, key=lambda arr: arr.T[0], reverse=True))
network = mini.Delaunay(points)
dtfn = mini.distances_to_furthest_neighbors(network)
network = network - (dtfn > np.percentile(dtfn, 90))
# network = network - (mini.distances_to_nearest_neighbors(network) < 0.05)
radii = mini.distances_to_nearest_neighbors(network)/2

x,y,z = network.coords
sources = x<np.percentile(x, 5)
sinks = x>np.percentile(x, 95)

def flood(network, sources, radii, initial_state=None):
    cmat = network.connectivity_matrix # some directed graphing
    area = np.pi*radii**2 # accessibility depends on HEAD only
    length = np.linalg.norm(network.points[cmat.row]-network.points[cmat.col])
    cmat.data = (area/length)[cmat.col]

    if initial_state is None:
        initial_state = np.zeros(network.order, dtype=bool)
    saturations = [initial_state | sources]

    while not all(saturations[-1]==1):
        flooded = saturations[-1].copy()
        supply = np.in1d(cmat.row, flooded.nonzero()) # & in same group as sources!
        demand = np.in1d(cmat.col, (~flooded).nonzero())
        unstable = (supply & demand).nonzero()[0]
        if not any(unstable):
            break
        easiest = cmat.col[max(unstable, key=cmat.data.item)]
        flooded[easiest] = 1

        saturations.append(flooded)
        
    return np.vstack(saturations)

def drain(network, sinks, cmat, initial_state=None):
    '''
    dumb method: remove largest so long as you don't generate islands
    '''
    cmat = network.connectivity_matrix # some directed graphing
    area = np.pi*radii**2 # accessibility depends on TAIL only
    length = np.linalg.norm(network.points[cmat.row]-network.points[cmat.col])
    cmat.data = (area/length)[cmat.col]

    if initial_state is None:
        initial_state = np.ones(network.order, dtype=bool)        
    saturations = [initial_state]
    while not all(saturations[-1]==0):
        flooded = saturations[-1].copy()
        supply = np.in1d(cmat.row, flooded.nonzero())
        demand = np.in1d(cmat.col, flooded.nonzero())
        unstable = (supply & demand).nonzero()[0]
        if not any(unstable):
            break
        
        for path in sorted(unstable, key=cmat.data.item):
            candidate = cmat.row[path]
            
            flooded[candidate] = False
            if any(sinks & flooded) and ((network - ~flooded).clusters == 1):
                break
            else:
                flooded[candidate] = True
                continue

        saturations.append(flooded)

    return np.vstack(saturations)


flood_history = flood(network, sources, radii)
drain_history = drain(network, sinks, radii)
history = np.vstack([
    flood_history,
    drain_history,
    ])

do = 2
if do == 1:
    scene = mini.Scene()
    # scene.add_spheres(network.points, radii, alpha=0.1, color=(1,1,1))
    scene.add_spheres(network.points, radii*history, color=(0,0,1))
    scene.add_wires(network.points, network.pairs, alpha=0.2)
    scene.play()

elif do == 2:
    plt.plot( (flood_history/radii).max(axis=1), (radii*flood_history).sum(axis=1), 'x')
    plt.plot( (drain_history[::-1]/radii).max(axis=1), (radii*drain_history[::-1]).sum(axis=1), 'x')
    plt.semilogx()
    plt.show()