import random
import pytest
import numpy as np
import minipnm as mini

network = mini.Cubic([10,10])
sources = network.indexes==0
sim = mini.simulations.Simulation(network.adjacency_matrix)
dif = mini.simulations.Diffusion(network.adjacency_matrix)
inv = mini.simulations.Invasion(network.adjacency_matrix, 1)

def test_base_history_generation():
    n = random.randint(0, 6)
    for i in range(n):
        sim.state = sim.state
    assert sim.history.shape == (n+1, network.order)

def test_base_blocking_cmat():
    sim.block(True) # blocks everything
    assert sim.cmat.nnz == 0
    sim.block(None) # blocks nothing
    assert sim.cmat.nnz == network.size

def test_base_blocked_change():
    sim.block(True)
    with pytest.raises(sim.BlockedStateChange):
        sim.state = sim.indexes

def test_diffusion():
    for t in range(9):
        dif.march(sources)
    np.testing.assert_almost_equal(dif.history.sum(axis=1), np.arange(10))

if __name__ == '__main__':
    pytest.main(__file__)
