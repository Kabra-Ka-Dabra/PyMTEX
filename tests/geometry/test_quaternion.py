import numpy as np
import pytest
from pymtex.geometry.quaternion import Quaternion


class TestConstruction:
    def test_scalar(self):
        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        assert q.a == 1.0 and q.b == q.c == q.d == 0.0

    def test_array(self):
        q = Quaternion([0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0])
        assert q.shape == (2,)

    def test_identity(self):
        q = Quaternion.identity()
        assert q.a == 1.0
        assert q.b == q.c == q.d == 0.0

    def test_nan(self):
        q = Quaternion.nan((3,))
        assert q.shape == (3,)
        assert np.all(np.isnan(q.a))


class TestNormalize:
    def test_already_unit(self):
        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        qn = q.normalize()
        assert np.isclose(qn.norm(), 1.0)

    def test_non_unit(self):
        q = Quaternion(2.0, 0.0, 0.0, 0.0)
        qn = q.normalize()
        assert np.isclose(qn.a, 1.0)

    def test_array(self):
        q = Quaternion([2.0, 0.0], [0.0, 3.0], [0.0, 0.0], [0.0, 0.0])
        qn = q.normalize()
        assert np.allclose(qn.norm(), 1.0)


class TestMultiplication:
    def test_identity_left(self):
        q = Quaternion(0.707, 0.707, 0.0, 0.0)
        e = Quaternion.identity()
        r = e * q
        assert np.isclose(r.a, q.a) and np.isclose(r.b, q.b)

    def test_identity_right(self):
        q = Quaternion(0.707, 0.707, 0.0, 0.0)
        e = Quaternion.identity()
        r = q * e
        assert np.isclose(r.a, q.a) and np.isclose(r.b, q.b)

    def test_inverse(self):
        # q * q^{-1} = identity
        q = Quaternion.from_axis_angle([0, 0, 1], np.pi / 3)
        r = q * q.inv()
        assert np.isclose(r.a, 1.0, atol=1e-10)
        assert np.isclose(r.b, 0.0, atol=1e-10)

    def test_non_commutative(self):
        q1 = Quaternion.from_axis_angle([1, 0, 0], np.pi / 4)
        q2 = Quaternion.from_axis_angle([0, 1, 0], np.pi / 4)
        r12 = q1 * q2
        r21 = q2 * q1
        # d component differs in sign for these specific rotations
        assert not np.isclose(r12.d, r21.d)

    def test_scalar_multiply(self):
        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        r = q * 2.0
        assert np.isclose(r.a, 2.0)


class TestAngleAxis:
    def test_round_trip_angle(self):
        angle = np.pi / 3
        q = Quaternion.from_axis_angle([0, 0, 1], angle)
        assert np.isclose(q.angle(), angle)

    def test_round_trip_axis(self):
        axis = np.array([1.0, 0.0, 0.0])
        q = Quaternion.from_axis_angle(axis, np.pi / 4)
        ax = q.axis()
        assert np.allclose(ax, axis, atol=1e-10)

    def test_identity_angle(self):
        q = Quaternion.identity()
        assert np.isclose(q.angle(), 0.0)


class TestMatrix:
    def test_identity(self):
        q = Quaternion.identity()
        M = q.to_matrix()
        assert np.allclose(M, np.eye(3))

    def test_round_trip(self):
        q_orig = Quaternion.from_axis_angle([1, 1, 0], np.pi / 5).normalize()
        M = q_orig.to_matrix()
        q_back = Quaternion.from_matrix(M)
        # q and -q represent the same rotation
        assert (np.allclose(q_back.components(), q_orig.components(), atol=1e-10) or
                np.allclose(q_back.components(), -q_orig.components(), atol=1e-10))

    def test_rotation_z90(self):
        # 90° about z: maps x→y
        q = Quaternion.from_axis_angle([0, 0, 1], np.pi / 2)
        M = q.to_matrix()
        x = np.array([1.0, 0.0, 0.0])
        y_expected = np.array([0.0, 1.0, 0.0])
        assert np.allclose(M @ x, y_expected, atol=1e-10)

    def test_batch(self):
        angles = np.linspace(0, np.pi, 5)
        axes = np.tile([0, 0, 1], (5, 1))
        q = Quaternion.from_axis_angle(axes, angles)
        M = q.to_matrix()
        assert M.shape == (5, 3, 3)
        # each should be a valid rotation matrix
        for i in range(5):
            assert np.allclose(M[i] @ M[i].T, np.eye(3), atol=1e-10)
            assert np.isclose(np.linalg.det(M[i]), 1.0, atol=1e-10)


class TestEuler:
    def test_identity_ZYZ(self):
        q = Quaternion.identity()
        alpha, beta, gamma = q.to_euler("ZYZ")
        assert np.isclose(beta, 0.0, atol=1e-10)

    def test_round_trip_ZYZ(self):
        alpha, beta, gamma = 0.3, 0.8, 1.2
        q = Quaternion.from_euler(alpha, beta, gamma, "ZYZ")
        a2, b2, g2 = q.to_euler("ZYZ")
        assert np.isclose(b2, beta, atol=1e-10)
        # alpha and gamma are only recoverable modulo 2π
        q2 = Quaternion.from_euler(a2, b2, g2, "ZYZ")
        # same rotation → same matrix
        assert np.allclose(q.to_matrix(), q2.to_matrix(), atol=1e-10)

    def test_round_trip_Bunge(self):
        phi1, Phi, phi2 = 0.5, 1.0, 1.5
        q = Quaternion.from_euler(phi1, Phi, phi2, "Bunge")
        p1, P, p2 = q.to_euler("Bunge")
        q2 = Quaternion.from_euler(p1, P, p2, "Bunge")
        assert np.allclose(q.to_matrix(), q2.to_matrix(), atol=1e-10)

    def test_ZYZ_Bunge_same_rotation(self):
        # Both conventions should produce the same rotation matrix from their native angles
        phi1, Phi, phi2 = 1.0, 0.5, 0.8
        q_b = Quaternion.from_euler(phi1, Phi, phi2, "Bunge")
        alpha, beta, gamma = q_b.to_euler("ZYZ")
        q_z = Quaternion.from_euler(alpha, beta, gamma, "ZYZ")
        assert np.allclose(q_b.to_matrix(), q_z.to_matrix(), atol=1e-10)


class TestDot:
    def test_self_dot(self):
        q = Quaternion(0.5, 0.5, 0.5, 0.5)
        assert np.isclose(q.dot(q), 1.0)

    def test_orthogonal(self):
        q1 = Quaternion(1.0, 0.0, 0.0, 0.0)
        q2 = Quaternion(0.0, 1.0, 0.0, 0.0)
        assert np.isclose(q1.dot(q2), 0.0)

    def test_dot_outer_shape(self):
        q1 = Quaternion.rand(n=4)
        q2 = Quaternion.rand(n=6)
        D = q1.dot_outer(q2)
        assert D.shape == (4, 6)


class TestRandom:
    def test_unit_norm(self):
        q = Quaternion.rand(n=100)
        assert np.allclose(q.norm(), 1.0, atol=1e-10)

    def test_max_angle(self):
        max_a = np.pi / 6
        q = Quaternion.rand(n=200, max_angle=max_a)
        assert np.all(q.angle() <= max_a + 1e-10)
