import numpy as np
import minipnm as mini

centers, radii = mini.algorithms.poisson_disk_sampling(2)
radial = mini.Radial(centers, radii)

def test_sphere_props():
    radial.spheres.centers
    radial.spheres.radii
    radial.spheres.areas
    radial.spheres.volumes

def test_cylinder_props():
    radial.cylinders.midpoints
    radial.cylinders.radii
    radial.cylinders.heights
    radial.cylinders.areas
    radial.cylinders.volumes

def test_raster():
    ''' sometimes fails '''
    raster = radial.rasterize(40)
    im = raster.asarray(raster['source']>=0)
    centers2, radii2 = mini.image.extract_spheres(im)
    np.testing.assert_equal(len(centers), len(centers2))

if __name__ == '__main__':
    import pytest
    pytest.main(__file__)
