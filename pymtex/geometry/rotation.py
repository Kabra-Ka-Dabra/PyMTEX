"""
Rotation class – extends Quaternion with rotation-specific operations.

Follows MTEX's @rotation class: a rotation is a quaternion that may also
be improper (i.e. combined with inversion).  Improper rotations are used
to represent reflections, roto-inversions, etc.
"""

from __future__ import annotations

import numpy as np

from pymtex.geometry.quaternion import Quaternion


class Rotation(Quaternion):
    """
    Proper or improper rotation represented as a unit quaternion.

    Parameters
    ----------
    a, b, c, d : array-like – quaternion components
    improper   : bool or array of bool – True for improper (inversion) rotations
    """

    def __init__(self, a, b, c, d, improper=False):
        super().__init__(a, b, c, d)
        imp = np.asarray(improper, dtype=bool)
        self.improper = np.broadcast_to(imp, self.shape).copy()

    # ------------------------------------------------------------------
    # Factory methods  (override Quaternion where needed)
    # ------------------------------------------------------------------

    @classmethod
    def identity(cls):
        """Identity rotation (no rotation, proper)."""
        return cls(1.0, 0.0, 0.0, 0.0)

    @classmethod
    def inversion(cls):
        """Pure inversion (improper identity)."""
        return cls(1.0, 0.0, 0.0, 0.0, improper=True)

    @classmethod
    def nan(cls, shape=()):
        z = np.full(shape, np.nan)
        return cls(z, z.copy(), z.copy(), z.copy())

    @classmethod
    def rand(cls, n=1, max_angle=None):
        """Uniform random proper rotation(s)."""
        q = Quaternion.rand(n=n, max_angle=max_angle)
        return cls(q.a, q.b, q.c, q.d)

    @classmethod
    def from_axis_angle(cls, axis, angle):
        """Create rotation from axis (unit vector) and angle (radians)."""
        q = Quaternion.from_axis_angle(axis, angle)
        return cls(q.a, q.b, q.c, q.d)

    @classmethod
    def from_euler(cls, alpha, beta, gamma, convention="ZYZ"):
        """
        Create rotation from Euler angles (radians).

        Conventions: 'ZYZ'/'Matthies'/'ABG' (default), 'Bunge'/'ZXZ'
        """
        q = Quaternion.from_euler(alpha, beta, gamma, convention=convention)
        return cls(q.a, q.b, q.c, q.d)

    @classmethod
    def from_matrix(cls, M):
        """Create rotation from 3×3 (or N×3×3) rotation matrix."""
        q = Quaternion.from_matrix(M)
        return cls(q.a, q.b, q.c, q.d)

    @classmethod
    def from_rodrigues(cls, r):
        """
        Create rotation from Rodrigues–Frank vector r = tan(θ/2) * n̂.

        Parameters
        ----------
        r : array-like, shape (3,) or (N, 3)
        """
        r = np.asarray(r, dtype=float)
        scalar_input = r.ndim == 1
        r = r.reshape(-1, 3)
        tan_half = np.linalg.norm(r, axis=-1)
        angle = 2.0 * np.arctan(tan_half)
        safe_tan = np.where(tan_half < 1e-15, 1.0, tan_half)
        axis = r / safe_tan[:, None]
        q = Quaternion.from_axis_angle(axis, angle)
        rot = cls(q.a, q.b, q.c, q.d)
        if scalar_input:
            return cls(q.a[0], q.b[0], q.c[0], q.d[0])
        return rot

    # ------------------------------------------------------------------
    # Array interface  (preserve improper flag)
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        return Rotation(self.a[idx], self.b[idx], self.c[idx], self.d[idx],
                        improper=self.improper[idx])

    def __repr__(self):
        if self.a.ndim == 0:
            kind = "Improper rotation" if self.improper else "Rotation"
            return (f"{kind}({self.a:.6f} + {self.b:.6f}i "
                    f"+ {self.c:.6f}j + {self.d:.6f}k)")
        return f"Rotation array, shape={self.shape}, improper={self.improper.any()}"

    # ------------------------------------------------------------------
    # Arithmetic  (preserve improper flag via XOR, matching MTEX)
    # ------------------------------------------------------------------

    def __mul__(self, other):
        if isinstance(other, Rotation):
            q = Quaternion.__mul__(self, other)
            imp = self.improper ^ other.improper
            return Rotation(q.a, q.b, q.c, q.d, improper=imp)
        return Quaternion.__mul__(self, other)

    def __neg__(self):
        return Rotation(-self.a, -self.b, -self.c, -self.d,
                        improper=self.improper.copy())

    # ------------------------------------------------------------------
    # Rotation-specific operations
    # ------------------------------------------------------------------

    def inv(self):
        """Inverse rotation: conjugate quaternion, same improper flag."""
        return Rotation(self.a, -self.b, -self.c, -self.d,
                        improper=self.improper.copy())

    def rotate(self, v):
        """
        Apply rotation to 3-D vector(s) using q v q*.

        Parameters
        ----------
        v : array-like, shape (3,) or (N, 3)

        Returns
        -------
        np.ndarray, same shape as v
        """
        v = np.asarray(v, dtype=float)
        scalar_v = v.ndim == 1
        v = v.reshape(-1, 3)

        mat = self.to_matrix()
        if mat.ndim == 2:
            # single rotation
            result = (mat @ v.T).T
        else:
            # broadcast: (N_rot, 3, 3) x (N_v, 3)
            result = np.einsum("nij,mj->nmi", mat, v)

        if self.improper.any():
            result = np.where(self.improper[..., None], -result, result)

        return result.squeeze(0) if (mat.ndim == 2 and scalar_v) else result

    def misorientation_angle(self, other):
        """
        Rotation angle of the misorientation self^{-1} * other.

        Parameters
        ----------
        other : Rotation

        Returns
        -------
        float or np.ndarray – angle in radians
        """
        mis = self.inv() * other
        return mis.angle()

    def angle_between(self, other):
        """Alias for misorientation_angle."""
        return self.misorientation_angle(other)

    def is_proper(self):
        """Return True where rotation is proper (not improper)."""
        return ~self.improper

    def is_improper(self):
        """Return True where rotation is improper."""
        return self.improper.copy()
