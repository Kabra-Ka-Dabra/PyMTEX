import numpy as np
import pytest
from pymtex.geometry.symmetry import CrystalSymmetry
from pymtex.geometry.miller import Miller


cs_cubic = CrystalSymmetry('m-3m')
cs_hex   = CrystalSymmetry('6/mmm')


class TestConstruction:
    def test_hkl(self):
        m = Miller(hkl=[1, 0, 0], cs=cs_cubic)
        assert m.type == 'hkl'
        assert np.allclose(m.hkl, [1, 0, 0])

    def test_uvw(self):
        m = Miller(uvw=[1, 1, 0], cs=cs_cubic)
        assert m.type == 'uvw'

    def test_both_raises(self):
        with pytest.raises(ValueError):
            Miller(hkl=[1,0,0], uvw=[1,0,0], cs=cs_cubic)

    def test_neither_raises(self):
        with pytest.raises(ValueError):
            Miller(cs=cs_cubic)

    def test_no_cs_raises(self):
        with pytest.raises(ValueError):
            Miller(hkl=[1,0,0])

    def test_array_hkl(self):
        m = Miller(hkl=[[1,0,0],[0,1,0],[0,0,1]], cs=cs_cubic)
        assert m.shape == (3,)


class TestAngle:
    def test_parallel(self):
        m = Miller(hkl=[1, 0, 0], cs=cs_cubic)
        assert np.isclose(m.angle(m, degrees=True), 0.0, atol=1e-8)

    def test_perpendicular(self):
        a = Miller(hkl=[1, 0, 0], cs=cs_cubic)
        b = Miller(hkl=[0, 1, 0], cs=cs_cubic)
        assert np.isclose(a.angle(b, degrees=True), 90.0, atol=1e-6)

    def test_110_vs_001(self):
        a = Miller(hkl=[1, 1, 0], cs=cs_cubic)
        b = Miller(hkl=[0, 0, 1], cs=cs_cubic)
        assert np.isclose(a.angle(b, degrees=True), 90.0, atol=1e-6)

    def test_100_vs_110(self):
        a = Miller(hkl=[1, 0, 0], cs=cs_cubic)
        b = Miller(hkl=[1, 1, 0], cs=cs_cubic)
        assert np.isclose(a.angle(b, degrees=True), 45.0, atol=1e-6)

    def test_111_vs_001_cubic(self):
        a = Miller(hkl=[1, 1, 1], cs=cs_cubic)
        b = Miller(hkl=[0, 0, 1], cs=cs_cubic)
        expected = np.degrees(np.arccos(1/np.sqrt(3)))
        assert np.isclose(a.angle(b, degrees=True), expected, atol=1e-5)


class TestSymmetricEquivalent:
    def test_cubic_100_multiplicity(self):
        # {100} family: ±x, ±y, ±z = 6 plane normals
        m = Miller(hkl=[1, 0, 0], cs=cs_cubic)
        equiv = m.symmetricEquivalent()
        assert equiv.size == 6

    def test_cubic_110_multiplicity(self):
        # {110} family: 12 equivalent normals
        m = Miller(hkl=[1, 1, 0], cs=cs_cubic)
        equiv = m.symmetricEquivalent()
        assert equiv.size == 12

    def test_cubic_111_multiplicity(self):
        # {111} family: 8 equivalent normals (all ±1 combinations)
        m = Miller(hkl=[1, 1, 1], cs=cs_cubic)
        equiv = m.symmetricEquivalent()
        assert equiv.size == 8

    def test_unit_vectors(self):
        m = Miller(hkl=[1, 2, 3], cs=cs_cubic)
        equiv = m.symmetricEquivalent()
        v = equiv.toVector3d()
        norms = np.linalg.norm(v, axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-8)


class TestMillerBravais:
    def test_hkil_to_hkl(self):
        m = Miller(hkl=[1, 0, -1, 0], cs=cs_hex)
        assert m.hkl.shape == (3,)
        assert np.allclose(m.hkl, [1, 0, 0])

    def test_UVTW_to_uvw(self):
        m = Miller(uvw=[1, 0, -1, 0], cs=cs_hex)
        assert m.uvw.shape == (3,)
