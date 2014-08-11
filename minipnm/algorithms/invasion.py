import numpy as np

def invasion(adjacency_matrix, sources, sinks=None):
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
