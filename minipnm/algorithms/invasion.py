import numpy as np
from scipy import sparse

def invasion(adjacency_matrix, sources, sinks=None):
    '''
    Most basic textbook example of sequential invasion
    '''
    tails = adjacency_matrix.row
    heads = adjacency_matrix.col
    conductance = adjacency_matrix.data

    # Assign a list of pores that are initially filled with liquid water
    state = sources
    history = [state.copy()]

    # Repeat until percolation or predefined stopping point
    while not all(state):
        # Identify interfacial throats (between unsaturated and saturated pores)
        saturated = state.nonzero()[0]
        interfacial_throats = np.in1d(tails, saturated) & ~np.in1d(heads, saturated)

        # Check for breakthrough conditions
        if sinks is not None and any(state[sinks]): break
        elif not any(interfacial_throats): break

        # Identify the interfacial throat, thmin, with lowest entry pressure
        thmin = max(interfacial_throats.nonzero()[0], key=conductance.item)
        next_fill = heads[thmin]
        conductance[next_fill] = 0
        state[next_fill] = True
        history.append(state.copy())

    return np.vstack(history)


class InvasionSimulation(object):
    '''
    This class simulates alop invasion with fractional generation in arbitrary
    pores simultaneously
    '''
    def __init__(self, capacities, conductance_matrix):
        self.capacities = capacities
        self.con = conductance_matrix
        self.saturation = np.zeros_like(capacities)

    def distribute(self, generation):
        content = self.capacities*self.saturation + generation
        excess = content.clip(self.capacities, content) - self.capacities
        content -= excess
        self.saturation = content / self.capacities
        if any(excess):
            for node in excess.nonzero()[0]:
                recipient = self.find_unsaturated_neighbor(node)
                if node == recipient:
                    raise Exception("Infinite loop")
                    return self.saturation
                excess[recipient] += excess[node]
                excess[node] = 0
            return self.distribute(excess)
        return self.saturation

    def find_unsaturated_neighbor(self, node):
        self.update_pressurized_clusters()
        full_sources = self.labels[self.con.col]==self.labels[node]
        non_full_sinks = self.saturation[self.con.row] < 0.999
        viable_throats = full_sources & non_full_sinks
        viable_throats = viable_throats.nonzero()[0] # bool -> idxs
        if len(viable_throats) == 0:
            return node
        best_throat = max(viable_throats, key=self.con.data.item)
        neighbor = self.con.row[best_throat]
        return neighbor

    def update_pressurized_clusters(self):
        '''
        this only needs to be checked when pores are newly saturated
        '''
        full_nodes = (self.saturation > 0.999).nonzero()[0]
        # superclusters require pressurized throats
        pressurized_throats = np.in1d(self.con.row, full_nodes) \
                            & np.in1d(self.con.col, full_nodes)
        i = self.con.row[pressurized_throats]
        j = self.con.col[pressurized_throats]
        v = np.ones_like(i)
        s = self.capacities.size
        coo = sparse.coo_matrix((v, (i, j)), shape=(s,s))
        self.labels = sparse.csgraph.connected_components(coo)[1]
