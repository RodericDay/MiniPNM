import minipnm as mini

def test_squeeze_not_needed():
    '''
    due to the way the property generator works, composite arrays returned as
    verticals. since python does not yet support the matrix multiplication as
    standard, this causes ambiguity. this ensures that the ambiguity isn't
    surfacing at the user level, by calling squeeze on those arrays within
    the property generator itself.
    '''
    topology = mini.Cubic([4,4,4])
    geometry = mini.Radial(topology.points, 0.3, topology.pairs)
    assert geometry.spheres.areas.ndim == 1
    assert geometry.cylinders.areas.ndim == 1

if __name__ == '__main__':
    import pytest
    pytest.main(__file__)
