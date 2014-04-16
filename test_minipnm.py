import minipnm as mini

import pytest
import os
import numpy as np

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
    delaunay = delaunay - ~delaunay.boundary()
    new_size = delaunay.size
    assert np.greater(original_size, new_size).all()

def visual():
    c = mini.Cubic(np.random.rand(30,30,30))
    c.preview(c['intensity'])

    d = mini.Delaunay(np.random.rand(300,3))
    d.preview(d.boundary())
    d = d - ~d.boundary()
    d.preview(d.boundary())

if __name__ == '__main__':
    errors = pytest.main([__file__])
    # os.system("find . -name '*.pyc' -delete")
    # visual()