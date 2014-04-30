import os

import pytest
import numpy as np

import minipnm as mini

# tests
def test_save_load_of_vtp_file():
    network = mini.Cubic(np.ones([3,3,3]))
    mini.save_vtp(network, 'test.vtp')
    for key, loaded_array in mini.load_vtp('test.vtp').items():
        original_array = network.pop(key)
        assert (original_array - loaded_array<1E6).all()
    assert( len(network) == 0 )
    os.system("find . -name 'test.vtp' -delete")

def test_prune():
    delaunay = mini.Delaunay(np.random.rand(100,3))
    original_size = delaunay.size
    changed = delaunay - ~delaunay.boundary()
    new_size = changed.size
    assert type(delaunay) is type(changed)
    assert np.greater(original_size, new_size).all()

def test_rectilinear_integrity():
    R = np.random.rand(50,40,30)
    # prune the mini way
    network = mini.Cubic(R)
    network = network - (R.ravel()<= R.mean())
    O = network.asarray()
    # what it would look like normally
    M = np.where(R > R.mean(), R, 0)
    assert np.allclose(M, O)

def test_linear_solver():
    R = mini.gaussian_noise([50, 50, 1])
    network = mini.Cubic(R)
    network = network - (R < np.percentile(R, 20)).ravel()
    x,y,z = network.coords
    l = x == x.min()
    r = x == x.max()
    ics = 2*l + 1*r
    sol = mini.linear_solve(network, ics)

if __name__ == '__main__':
    errors = pytest.main([__file__])
    os.system("find . -name '*.pyc' -delete")
