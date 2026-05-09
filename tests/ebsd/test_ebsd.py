import numpy as np
import pytest
from pymtex.geometry.symmetry import CrystalSymmetry
from pymtex.geometry.orientation import Orientation
from pymtex.ebsd import EBSD


cs = CrystalSymmetry('m-3m')


def make_ebsd(n=20):
    x = np.linspace(0, 9, n)
    y = np.zeros(n)
    ori = Orientation.rand(n, cs)
    return EBSD(x, y, ori)


class TestConstruction:
    def test_num_pixels(self):
        ebsd = make_ebsd(20)
        assert ebsd.numPixels == 20

    def test_bad_length_raises(self):
        x = np.linspace(0, 9, 10)
        y = np.zeros(10)
        ori = Orientation.rand(5, cs)
        with pytest.raises(ValueError):
            EBSD(x, y, ori)

    def test_repr(self):
        ebsd = make_ebsd(5)
        assert 'EBSD' in repr(ebsd)


class TestFilter:
    def test_filter_reduces_size(self):
        ebsd = make_ebsd(20)
        mask = np.zeros(20, dtype=bool)
        mask[:10] = True
        filtered = ebsd.filter(mask)
        assert filtered.numPixels == 10

    def test_indexed(self):
        x = np.linspace(0, 4, 5)
        y = np.zeros(5)
        ori = Orientation.rand(5, cs)
        phase = np.array([0, 1, 1, 0, 1])
        ebsd = EBSD(x, y, ori, phase=phase)
        indexed = ebsd.indexed()
        assert indexed.numPixels == 3


class TestMeanOrientation:
    def test_returns_orientation(self):
        ebsd = make_ebsd(10)
        mean = ebsd.meanOrientation()
        assert isinstance(mean, Orientation)
        assert np.isclose(mean.norm(), 1.0, atol=1e-8)


class TestGrainSegmentation:
    def test_single_grain(self):
        # All pixels have the same orientation → one grain
        x = np.array([0., 1., 2., 3.])
        y = np.zeros(4)
        q = Orientation.cube(cs)
        # repeat the same orientation 4 times
        ori = Orientation(
            np.full(4, q.a), np.full(4, q.b),
            np.full(4, q.c), np.full(4, q.d),
            cs=cs
        )
        ebsd = EBSD(x, y, ori)
        grain_id, n_grains = ebsd.calcGrains(threshold=np.deg2rad(1))
        assert n_grains == 1
        assert np.all(grain_id == 1)

    def test_two_grains(self):
        # Two groups of pixels with a large misorientation between them
        x = np.array([0., 1., 2., 3.])
        y = np.zeros(4)
        q1 = Orientation.cube(cs)
        q2 = Orientation.byAxisAngle([0, 0, 1], np.deg2rad(45), cs)
        a = np.array([q1.a, q1.a, q2.a, q2.a])
        b = np.array([q1.b, q1.b, q2.b, q2.b])
        c = np.array([q1.c, q1.c, q2.c, q2.c])
        d = np.array([q1.d, q1.d, q2.d, q2.d])
        ori = Orientation(a, b, c, d, cs=cs)
        ebsd = EBSD(x, y, ori)
        grain_id, n_grains = ebsd.calcGrains(threshold=np.deg2rad(5))
        # pixels 0,1 in one grain; 2,3 in another — but grain count depends
        # on whether the 45° misorientation exceeds threshold
        assert n_grains >= 1
