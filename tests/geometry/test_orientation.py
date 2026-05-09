import numpy as np
import pytest
from pymtex.geometry.symmetry import CrystalSymmetry
from pymtex.geometry.orientation import Orientation


cs_cubic = CrystalSymmetry('m-3m')
cs_hex   = CrystalSymmetry('6/mmm')


class TestConstruction:
    def test_by_euler_identity(self):
        o = Orientation.byEuler(0, 0, 0, cs_cubic)
        # q and -q represent the same rotation; angle == 0 is the real check
        assert np.isclose(o.angle(), 0.0, atol=1e-10)

    def test_by_euler_degrees(self):
        o1 = Orientation.byEuler(90, 45, 30, cs_cubic, degrees=True)
        o2 = Orientation.byEuler(
            np.deg2rad(90), np.deg2rad(45), np.deg2rad(30), cs_cubic
        )
        assert np.isclose(o1.a, o2.a, atol=1e-10)

    def test_by_axis_angle(self):
        o = Orientation.byAxisAngle([0, 0, 1], np.pi/2, cs_cubic)
        assert np.isclose(o.angle(), np.pi/2, atol=1e-10)

    def test_by_matrix_identity(self):
        o = Orientation.byMatrix(np.eye(3), cs_cubic)
        assert np.isclose(o.angle(), 0.0, atol=1e-10)

    def test_rand(self):
        orients = Orientation.rand(50, cs_cubic)
        assert orients.shape == (50,)
        assert np.allclose(orients.norm(), 1.0, atol=1e-10)

    def test_cs_stored(self):
        o = Orientation.cube(cs_cubic)
        assert o.cs.name == 'm-3m'


class TestStandardComponents:
    def test_cube(self):
        o = Orientation.cube(cs_cubic)
        phi1, Phi, phi2 = o.toEuler(degrees=True)
        assert np.isclose(Phi, 0.0, atol=1e-8)

    def test_goss(self):
        o = Orientation.goss(cs_cubic)
        _, Phi, _ = o.toEuler(degrees=True)
        assert np.isclose(Phi, 45.0, atol=1e-6)


class TestSymmetricEquivalent:
    def test_shape(self):
        o = Orientation.cube(cs_cubic)
        equiv = o.symmetricEquivalent()
        assert equiv.shape[0] == cs_cubic.nfold

    def test_all_unit_quaternions(self):
        o = Orientation.rand(1, cs_cubic)
        equiv = o.symmetricEquivalent()
        norms = equiv.norm()
        assert np.allclose(norms, 1.0, atol=1e-8)


class TestFundamentalRegion:
    def test_angle_minimised(self):
        o = Orientation.rand(20, cs_cubic)
        o_fr = o.project2FundamentalRegion()
        # Each angle in FR ≤ corresponding symmetry-equivalent max
        equiv = o.symmetricEquivalent()
        min_angles = np.min(equiv.angle(), axis=0)
        fr_angles  = o_fr.angle()
        assert np.allclose(fr_angles, min_angles, atol=1e-6)

    def test_idempotent(self):
        o = Orientation.rand(10, cs_cubic)
        o1 = o.project2FundamentalRegion()
        o2 = o1.project2FundamentalRegion()
        assert np.allclose(o1.angle(), o2.angle(), atol=1e-8)


class TestMisorientation:
    def test_same_orientation_zero(self):
        o = Orientation.byAxisAngle([0, 0, 1], np.pi/3, cs_cubic)
        assert np.isclose(o.calcMisorientation(o), 0.0, atol=1e-8)

    def test_known_misorientation(self):
        o1 = Orientation.cube(cs_cubic)
        o2 = Orientation.byAxisAngle([0, 0, 1], np.deg2rad(5), cs_cubic)
        mis = o1.calcMisorientation(o2)
        assert np.isclose(mis, np.deg2rad(5), atol=1e-6)

    def test_symmetry_reduces_angle(self):
        # A 45° rotation about [100] equals a 45° rotation, but after
        # applying cubic symmetry the fundamental zone misorientation should
        # be ≤ 62.8° (max for cubic)
        o1 = Orientation.cube(cs_cubic)
        o2 = Orientation.byAxisAngle([1, 0, 0], np.deg2rad(45), cs_cubic)
        mis = o1.calcMisorientation(o2)
        assert mis <= np.deg2rad(63)

    def test_hexagonal(self):
        o1 = Orientation.cube(cs_hex)
        o2 = Orientation.byAxisAngle([0, 0, 1], np.deg2rad(10), cs_hex)
        mis = o1.calcMisorientation(o2)
        assert np.isclose(mis, np.deg2rad(10), atol=1e-6)
