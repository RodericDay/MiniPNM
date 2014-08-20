import numpy as np
import minipnm as mini

def test_rectilinear_integrity():
    R = np.random.rand(10,20,30)
    # prune the mini way
    network = mini.Cubic.from_source(R)
    network = network - (R <= R.mean())
    O = network.asarray(network['source'])
    # what it would look like normally
    M = np.where(R > R.mean(), R, 0)
    assert np.allclose(M, O)

def test_rectilinear_integrity_2d():
    R = np.random.rand(15,30)
    # prune the mini way
    network = mini.Cubic.from_source(R)
    network = network - (R <= R.mean())
    O = network.asarray(network['source'])
    # what it would look like normally
    M = np.where(R > R.mean(), R, 0)
    assert np.allclose(M, O)

if __name__ == '__main__':
    import pytest
    pytest.main(__file__)
