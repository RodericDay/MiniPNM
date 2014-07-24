import numpy as np

def invasion(network, sources, thresholds, sinks=None):
    network = network.copy()

    # assume throat access is bound by minimal pore access
    throat_thresholds = thresholds[network.pairs].max(axis=1)

    # Assign a list of pores that are initially filled with liquid water
    saturation = [sources]

    # Repeat until percolation or predefined stopping point
    while True:
        # Identify interfacial throats (between unsaturated and saturated pores)
        interfacial_throats = network.cut(saturation[-1])

        # Check for breakthrough conditions
        if sinks is not None and any(saturation[-1][sinks]): break
        elif len(interfacial_throats) == 0: break

        # Identify the interfacial throat, thmin, with lowest entry pressure
        entry_pressures = throat_thresholds[interfacial_throats]
        th_min = interfacial_throats[entry_pressures.argmin()]

        # Invade any air-filled pore adjacent to thmin with liquid water
        new_saturation = saturation[-1].copy()
        new_saturation[network.pairs[th_min]] = 1
        saturation.append(new_saturation)

    return np.vstack(saturation)