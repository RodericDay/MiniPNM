import numpy as np
import minipnm as mini

def test_raster():
    ''' sometimes fails '''
    centers, radii = mini.algorithms.poisson_disk_sampling()
    radial = mini.Radial(centers, radii)
    raster = radial.rasterize(40)
    im = raster.asarray(raster['source']>=0)
    centers2, radii2 = mini.image.extract_spheres(im)
    np.testing.assert_equal(len(centers), len(centers2))

if __name__ == '__main__':
    import pytest
    pytest.main(__file__)
