import numpy as np
import pytest
from pymtex.geometry.rotation import Rotation


class TestConstruction:
    def test_identity(self):
        r = Rotation.identity()
        assert r.a == 1.0
        assert not r.improper

    def test_inversion(self):
        r = Rotation.inversion()
        assert r.improper

    def test_proper_flag_default(self):
        r = Rotation(1.0, 0.0, 0.0, 0.0)
        assert not r.improper

    def test_from_axis_angle(self):
        r = Rotation.from_axis_angle([0, 0, 1], np.pi / 2)
        assert np.isclose(r.angle(), np.pi / 2)
        assert not r.improper

    def test_from_euler(self):
        r = Rotation.from_euler(0.3, 0.8, 1.2, "ZYZ")
        assert np.isclose(r.norm(), 1.0, atol=1e-10)

    def test_from_matrix_identity(self):
        r = Rotation.from_matrix(np.eye(3))
        assert np.isclose(r.angle(), 0.0, atol=1e-10)

    def test_from_rodrigues_zero(self):
        r = Rotation.from_rodrigues([0.0, 0.0, 0.0])
        assert np.isclose(r.angle(), 0.0, atol=1e-10)

    def test_from_rodrigues_round_trip(self):
        r1 = Rotation.from_axis_angle([1, 0, 0], np.pi / 4)
        rod = r1.to_rodrigues()
        r2 = Rotation.from_rodrigues(rod)
        assert np.allclose(r1.to_matrix(), r2.to_matrix(), atol=1e-10)


class TestInverse:
    def test_inv_proper(self):
        r = Rotation.from_axis_angle([0, 1, 0], np.pi / 3)
        prod = r * r.inv()
        assert np.isclose(prod.a, 1.0, atol=1e-10)

    def test_inv_preserves_improper(self):
        r = Rotation(1.0, 0.0, 0.0, 0.0, improper=True)
        ri = r.inv()
        assert ri.improper


class TestMultiplication:
    def test_proper_proper(self):
        r1 = Rotation.from_axis_angle([1, 0, 0], np.pi / 4)
        r2 = Rotation.from_axis_angle([0, 1, 0], np.pi / 4)
        r = r1 * r2
        assert isinstance(r, Rotation)
        assert not r.improper

    def test_proper_improper(self):
        r1 = Rotation.identity()
        r2 = Rotation.inversion()
        r = r1 * r2
        assert r.improper

    def test_improper_improper(self):
        r1 = Rotation.inversion()
        r2 = Rotation.inversion()
        r = r1 * r2
        assert not r.improper  # XOR: improper ^ improper = False


class TestRotateVector:
    def test_rotate_x_to_y(self):
        r = Rotation.from_axis_angle([0, 0, 1], np.pi / 2)
        v = np.array([1.0, 0.0, 0.0])
        result = r.rotate(v)
        assert np.allclose(result, [0.0, 1.0, 0.0], atol=1e-10)

    def test_rotate_identity(self):
        r = Rotation.identity()
        v = np.array([1.0, 2.0, 3.0])
        assert np.allclose(r.rotate(v), v, atol=1e-10)

    def test_rotate_180_about_x(self):
        r = Rotation.from_axis_angle([1, 0, 0], np.pi)
        v = np.array([0.0, 1.0, 0.0])
        result = r.rotate(v)
        assert np.allclose(result, [0.0, -1.0, 0.0], atol=1e-10)

    def test_rotate_batch_vectors(self):
        r = Rotation.from_axis_angle([0, 0, 1], np.pi / 2)
        vs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = r.rotate(vs)
        # single rotation applied to 2 vectors → (2, 3)
        assert result.shape == (2, 3)
        assert np.allclose(result[0], [0.0, 1.0, 0.0], atol=1e-10)
        assert np.allclose(result[1], [-1.0, 0.0, 0.0], atol=1e-10)

    def test_improper_negates(self):
        r = Rotation(1.0, 0.0, 0.0, 0.0, improper=True)
        v = np.array([1.0, 2.0, 3.0])
        result = r.rotate(v)
        assert np.allclose(result, -v, atol=1e-10)


class TestMisorientationAngle:
    def test_same_rotation(self):
        r = Rotation.from_axis_angle([0, 0, 1], np.pi / 3)
        assert np.isclose(r.misorientation_angle(r), 0.0, atol=1e-10)

    def test_known_angle(self):
        angle = np.pi / 4
        r1 = Rotation.identity()
        r2 = Rotation.from_axis_angle([0, 0, 1], angle)
        assert np.isclose(r1.misorientation_angle(r2), angle, atol=1e-10)


class TestRandom:
    def test_unit_norm(self):
        r = Rotation.rand(n=50)
        assert np.allclose(r.norm(), 1.0, atol=1e-10)

    def test_proper_by_default(self):
        r = Rotation.rand(n=50)
        assert not np.any(r.improper)
