#!/usr/bin/env python
import os

import pytest
import numpy as np

import minipnm as mini

# tests
def disabled_test_save_load_of_vtp_file():
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
    R = np.random.rand(10,20,30)
    # prune the mini way
    network = mini.Cubic(R)
    network = network - (R.ravel()<= R.mean())
    O = network.asarray()
    # what it would look like normally
    M = np.where(R > R.mean(), R, 0)
    assert np.allclose(M, O)

def test_linear_solver():
    R = mini.gaussian_noise([20, 20, 20])
    network = mini.Cubic(R)
    network = network - (R<np.percentile(R, 10))
    x,y,z = network.coords
    l = x == x.min()
    r = x == x.max()
    dbcs = 2*l + 1*r
    sol = mini.linear_solve(network, dbcs)

    l_flux = network.flux(sol, l)
    r_flux = network.flux(sol, r)
    assert np.allclose(l_flux.sum(), r_flux.sum())

def test_subtract_all():
    network = mini.Cubic.empty([3,3,3])
    reduced = network - np.ones(network.size[0]).astype(bool)
    assert set(network.keys()) == set(reduced.keys())
    assert reduced.size == (0,0)
    assert all(value.size==0 for value in reduced.values())
    rereduced = reduced - np.ones(reduced.size[0]).astype(bool)
    assert set(network.keys()) == set(rereduced.keys())
    assert rereduced.size == (0,0)
    assert all(value.size==0 for value in rereduced.values())

def test_percolation():
    network = mini.Cubic.empty([5,1,1])
    x,y,z = network.coords
    sources = np.array([1,0,0,0,0], dtype=bool)
    thresholds = np.array([1,2,3,4,1])
    conditions = [1,2,3,4]

    output = mini.percolation(network, sources, thresholds, conditions, rate=1)

    target = np.array([
        [1./2, 0,    0,    0,    0   ],
        [3./4, 1./2, 0,    0,    0   ],
        [5./6, 2./3, 1./2, 0,    0   ],
        [7./8, 3./4, 5./8, 1./2, 7./8],
    ])
    
    assert np.allclose(output, target)

def test_render():
    try:
        import vtk
    except ImportError:
        return
    network = mini.Delaunay.random(100)
    scene = mini.Scene()
    scene.add_wires(network.points, network.pairs)

if __name__ == '__main__':
    errors = pytest.main([__file__])
    os.system("find . -name '*.pyc' -delete")
