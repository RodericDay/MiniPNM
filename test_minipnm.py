#!/usr/bin/env python
from __future__ import division
import os
import itertools as it

import pytest
import numpy as np

import minipnm as mini

def disabled_test_save_load_of_vtp_file():
    network = mini.Cubic(np.ones([3,3,3]))
    mini.save_vtp(network, 'test.vtp')
    for key, loaded_array in mini.load_vtp('test.vtp').items():
        original_array = network.pop(key)
        assert (original_array - loaded_array<1E6).all()
    assert( len(network) == 0 )
    os.system("find . -name 'test.vtp' -delete")

def test_print():
    network = mini.Delaunay.random(100)
    print( network )

def test_prune():
    delaunay = mini.Delaunay(np.random.rand(100,3))
    original_size = delaunay.size
    changed = delaunay - ~delaunay.boundary()
    new_size = changed.size
    assert type(delaunay) is type(changed)
    assert np.greater(original_size, new_size).all()

def test_subtract_all():
    network = mini.Cubic([3,3,3])
    reduced = network.copy()
    reduced.prune(network.indexes!=-1)
    assert set(network.keys()) == set(reduced.keys())
    assert reduced.size == 0
    assert all([value.size==0 for value in reduced.values()])
    rereduced = reduced.copy()
    rereduced.prune(network.indexes!=-1)
    assert set(network.keys()) == set(rereduced.keys())
    assert rereduced.size == 0
    assert all(value.size==0 for value in rereduced.values())

def test_render():
    try:
        import vtk
    except ImportError:
        return
    network = mini.Delaunay.random(100)
    scene = mini.Scene()
    network.render(scene=scene)

def test_handling_of_pseudo_array_input():
    network = mini.Network()
    with pytest.raises(TypeError):
        network.points = None, None, None
    network.points = [(1,1,1), [2,2,2], np.array([3,3,3])]
    network.pairs = (0,1)
    network.pairs = [(1,2), [2,0]]

def test_merge():
    network = mini.Delaunay.random(100)
    inside, outside = network.split(network.boundary())
    (inside | outside)

def test_qhull_coplanar():
    points = np.random.rand(100,3)
    points.T[2] = 0
    network = mini.Delaunay(points)
    network.boundary()

def test_lengths():
    # create a voxelized sphere. black (1s) is void.
    N = 13
    im = np.ones([N,N,N])
    for i in [i for i, c in np.ndenumerate(im) if np.linalg.norm(np.subtract(i, N/2-0.5))>N/2.5]:
        im[i] = 0

def test_save_and_load():
    original = mini.Cubic([20,20,20])
    mini.save(original)
    copy = mini.load('Cubic.npz')
    assert type(original) is type(copy)
    for key, value in original.items():
        np.testing.assert_allclose(copy[key], value)
    os.system("rm Cubic.npz")

def test_clone():
    original = mini.Cubic([5,5,5])
    copy = original.copy()
    assert type(original) is type(copy)
    unmatched = set(original.keys()) ^ set(copy.keys())
    assert not unmatched
    for key, value in original.items():
        np.testing.assert_allclose(value, copy[key])

if __name__ == '__main__':
    errors = pytest.main()
    os.system("find . -name '*.pyc' -delete")
