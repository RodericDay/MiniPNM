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

def test_basic_cubic():
    '''
    this is actually wrong
    '''
    R = np.zeros([2,2,2])
    network = mini.Cubic(R, R.shape)
    expected = [    [0,0,0], [0,0,2], [0,2,0], [0,2,2],
                    [2,0,0], [2,0,2], [2,2,0], [2,2,2], ]
    assert np.allclose(expected, network.points)

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

def test_rectilinear_integrity_2d():
    R = np.random.rand(15,30)
    # prune the mini way
    network = mini.Cubic(R)
    network = network - (R.ravel()<= R.mean())
    O = network.asarray()
    # what it would look like normally
    M = np.where(R > R.mean(), R, 0)
    assert np.allclose(M, O)

def test_linear_solver():
    R = mini.gaussian_noise([10, 10, 10])
    network = mini.Cubic(R)
    network = network - (R<np.percentile(R, 10))
    x,y,z = network.coords
    l = x == x.min()
    r = x == x.max()
    dbcs = { 2 : l, 1 : r }
    sol = mini.solve_bvp(network.laplacian, dbcs)

    l_flux = np.subtract(*network.cut(l, sol)).sum()
    r_flux = -np.subtract(*network.cut(r, sol)).sum()
    assert np.allclose(l_flux, r_flux)

def test_subtract_all():
    network = mini.Cubic.empty([3,3,3])
    reduced = network - np.ones(network.order).astype(bool)
    assert set(network.keys()) == set(reduced.keys())
    assert reduced.size == 0
    assert all(value.size==0 for value in reduced.values())
    rereduced = reduced - np.ones(reduced.order).astype(bool)
    assert set(network.keys()) == set(rereduced.keys())
    assert rereduced.size == 0
    assert all(value.size==0 for value in rereduced.values())

def test_percolation():
    network = mini.Cubic.empty([5,1,1])
    x,y,z = network.coords
    sources = np.array([1,0,0,0,0], dtype=bool)
    thresholds = np.array([1,2,3,4,1])
    conditions = [1,2,3,4]

    output = mini.percolation(network, sources, thresholds, conditions, rate=1)

    target = np.array([
        [1/2,   0,   0,   0,   0],
        [3/4, 1/2,   0,   0,   0],
        [5/6, 2/3, 1/2,   0,   0],
        [7/8, 3/4, 5/8, 1/2, 7/8],
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

def test_sphere_stuff():
    N = 20
    x = np.arange(-N,N)
    y = np.arange(-N,N)
    X,Y = np.meshgrid(x,y)
    Z = np.sqrt(X**2 + Y**2)
    im = Z < Z.mean()

    network = mini.Cubic(im, im.shape)
    network = network - im

    centers, radii = mini.extract_spheres(im)
    sphere = mini.PackedSpheres(centers, radii)

    stitched = mini.binary.radial_join(network, sphere)

def test_lengths():
    # create a voxelized sphere. black (1s) is void.
    N = 13
    im = np.ones([N,N,N])
    for i in [i for i, c in np.ndenumerate(im) if np.linalg.norm(np.subtract(i, N/2-0.5))>N/2.5]:
        im[i] = 0

def test_bridson():
    pdf = (np.random.normal(3) if i%3!=2 else np.random.normal(8) for i in it.count())
    network = mini.Bridson([30,30,10], pdf)

def test_shortest_path():
    string = '''
    131 673 234 103  18
    201  96 342 965 150
    630 803 746 422 111
    537 699 497 121 956
    805 732 524  37 331
    '''
    matrix = np.matrix([[int(w) for w in row.split(' ') if w]
                       for row in string.strip().split('\n')])
    matrix = np.rot90(matrix, 3)
    network = mini.Cubic(matrix)
    cmat = network.adjacency_matrix
    cmat.data = matrix.A1[cmat.col]
    # some geometric definitions
    x,y,z = network.coords
    top_left = network.indexes[(x==x.min()) & (y==y.max())]
    bottom_right = network.indexes[(x==x.max()) & (y==y.min())]
    left = network.indexes[x==x.min()]
    right = network.indexes[x==x.max()]
    # top-left to bottom-right;           right and down:           2427
    blocked = (x[cmat.row] > x[cmat.col]) | (y[cmat.row] < y[cmat.col])
    np.set_printoptions(linewidth=200)
    cmat1 = cmat.copy()
    cmat1.row = cmat.row[~blocked]
    cmat1.col = cmat.col[~blocked]
    cmat1.data = cmat.data[~blocked]
    path1 = mini.algorithms.shortest_path(cmat1, top_left, bottom_right)
    assert matrix.A1[path1].sum() == 2427
    # any left start to any right end;    left, right, up, down:     994
    path2 = mini.algorithms.shortest_path(cmat, left, right)
    assert matrix.A1[path2].sum() == 994
    # top-left to bottom-right;           left, right, up, down:    2297
    path3 = mini.algorithms.shortest_path(cmat, top_left, bottom_right)
    assert matrix.A1[path3].sum() == 2297

if __name__ == '__main__':
    errors = pytest.main([__file__])
    os.system("find . -name '*.pyc' -delete")
