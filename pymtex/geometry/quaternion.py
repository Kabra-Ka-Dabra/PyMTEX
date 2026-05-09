"""
Quaternion class following MTEX conventions.

A quaternion is stored as q = a + b*i + c*j + d*k where:
  a  - real (scalar) part
  b,c,d - imaginary (vector) part

For a rotation by angle θ about unit axis n = (n1, n2, n3):
  a = cos(θ/2),  b = n1*sin(θ/2),  c = n2*sin(θ/2),  d = n3*sin(θ/2)

Note: q and -q represent the same rotation; MTEX treats them as distinct objects.
"""

import numpy as np


class Quaternion:
    """Unit quaternion q = a + bi + cj + dk (MTEX convention)."""

    def __init__(self, a, b, c, d):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        c = np.asarray(c, dtype=float)
        d = np.asarray(d, dtype=float)
        shape = np.broadcast_shapes(a.shape, b.shape, c.shape, d.shape)
        self.a = np.broadcast_to(a, shape).copy()
        self.b = np.broadcast_to(b, shape).copy()
        self.c = np.broadcast_to(c, shape).copy()
        self.d = np.broadcast_to(d, shape).copy()

    # ------------------------------------------------------------------
    # Factory / class methods
    # ------------------------------------------------------------------

    @classmethod
    def identity(cls):
        """Identity quaternion (no rotation)."""
        return cls(1.0, 0.0, 0.0, 0.0)

    @classmethod
    def nan(cls, shape=()):
        """Quaternion filled with NaN."""
        z = np.full(shape, np.nan)
        return cls(z, z.copy(), z.copy(), z.copy())

    @classmethod
    def rand(cls, n=1, max_angle=None):
        """
        Uniform random unit quaternions on SO(3) via Shoemake's method.

        Parameters
        ----------
        n : int
            Number of quaternions to generate.
        max_angle : float, optional
            If given, restrict rotations to angles <= max_angle (radians).
        """
        def _sample(count):
            u1 = np.random.uniform(0, 1, count)
            u2 = np.random.uniform(0, 1, count)
            u3 = np.random.uniform(0, 1, count)
            a = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
            b = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
            c = np.sqrt(u1) * np.sin(2 * np.pi * u3)
            d = np.sqrt(u1) * np.cos(2 * np.pi * u3)
            return cls(a, b, c, d)

        q = _sample(n)
        if max_angle is not None:
            mask = q.angle() > max_angle
            while np.any(mask):
                replacement = _sample(int(np.sum(mask)))
                q.a[mask] = replacement.a
                q.b[mask] = replacement.b
                q.c[mask] = replacement.c
                q.d[mask] = replacement.d
                mask = q.angle() > max_angle
        if n == 1:
            return cls(q.a[0], q.b[0], q.c[0], q.d[0])
        return q

    @classmethod
    def from_axis_angle(cls, axis, angle):
        """
        Create from rotation axis and angle.

        Parameters
        ----------
        axis : array-like, shape (3,) or (N, 3) – unit vector(s)
        angle : float or (N,) – rotation angle in radians
        """
        axis = np.asarray(axis, dtype=float)
        angle = np.asarray(angle, dtype=float)
        norms = np.linalg.norm(axis, axis=-1, keepdims=True)
        axis = axis / np.where(norms < 1e-15, 1.0, norms)
        half = angle / 2.0
        s = np.sin(half)
        return cls(
            np.cos(half),
            axis[..., 0] * s,
            axis[..., 1] * s,
            axis[..., 2] * s,
        )

    @classmethod
    def from_euler(cls, alpha, beta, gamma, convention="ZYZ"):
        """
        Create from Euler angles (radians).

        Supported conventions
        ---------------------
        'ZYZ' / 'Matthies' / 'ABG'  – default (active, intrinsic ZYZ)
        'Bunge' / 'ZXZ'             – passive ZXZ used in materials science
        """
        alpha = np.asarray(alpha, dtype=float)
        beta  = np.asarray(beta,  dtype=float)
        gamma = np.asarray(gamma, dtype=float)

        conv = convention.upper()
        if conv in ("BUNGE", "ZXZ"):
            # Convert Bunge → Matthies offsets from MTEX Euler.m
            alpha = alpha - np.pi / 2
            gamma = gamma - 3 * np.pi / 2
        elif conv not in ("ZYZ", "MATTHIES", "ABG"):
            raise ValueError(f"Unknown Euler convention: {convention!r}")

        p = (alpha + gamma) / 2   # = atan2(d, a)
        q = (gamma - alpha) / 2   # = atan2(b, c)

        a = np.cos(beta / 2) * np.cos(p)
        b = np.sin(beta / 2) * np.sin(q)
        c = np.sin(beta / 2) * np.cos(q)
        d = np.cos(beta / 2) * np.sin(p)
        return cls(a, b, c, d)

    @classmethod
    def from_matrix(cls, M):
        """
        Create from a 3×3 (or N×3×3) rotation matrix using Shepperd's method.

        Parameters
        ----------
        M : array-like, shape (3, 3) or (N, 3, 3)
        """
        M = np.asarray(M, dtype=float)
        scalar_input = M.ndim == 2
        M = M.reshape(-1, 3, 3)
        n = M.shape[0]

        a = np.zeros(n)
        b = np.zeros(n)
        c = np.zeros(n)
        d = np.zeros(n)

        trace = M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]

        t0 = trace > 0
        if np.any(t0):
            s = 0.5 / np.sqrt(trace[t0] + 1.0)
            a[t0] = 0.25 / s
            b[t0] = (M[t0, 2, 1] - M[t0, 1, 2]) * s
            c[t0] = (M[t0, 0, 2] - M[t0, 2, 0]) * s
            d[t0] = (M[t0, 1, 0] - M[t0, 0, 1]) * s

        t1 = ~t0 & (M[:, 0, 0] > M[:, 1, 1]) & (M[:, 0, 0] > M[:, 2, 2])
        if np.any(t1):
            s = 2.0 * np.sqrt(1.0 + M[t1, 0, 0] - M[t1, 1, 1] - M[t1, 2, 2])
            a[t1] = (M[t1, 2, 1] - M[t1, 1, 2]) / s
            b[t1] = 0.25 * s
            c[t1] = (M[t1, 0, 1] + M[t1, 1, 0]) / s
            d[t1] = (M[t1, 0, 2] + M[t1, 2, 0]) / s

        t2 = ~t0 & ~t1 & (M[:, 1, 1] > M[:, 2, 2])
        if np.any(t2):
            s = 2.0 * np.sqrt(1.0 + M[t2, 1, 1] - M[t2, 0, 0] - M[t2, 2, 2])
            a[t2] = (M[t2, 0, 2] - M[t2, 2, 0]) / s
            b[t2] = (M[t2, 0, 1] + M[t2, 1, 0]) / s
            c[t2] = 0.25 * s
            d[t2] = (M[t2, 1, 2] + M[t2, 2, 1]) / s

        t3 = ~t0 & ~t1 & ~t2
        if np.any(t3):
            s = 2.0 * np.sqrt(1.0 + M[t3, 2, 2] - M[t3, 0, 0] - M[t3, 1, 1])
            a[t3] = (M[t3, 1, 0] - M[t3, 0, 1]) / s
            b[t3] = (M[t3, 0, 2] + M[t3, 2, 0]) / s
            c[t3] = (M[t3, 1, 2] + M[t3, 2, 1]) / s
            d[t3] = 0.25 * s

        q = cls(a, b, c, d)
        if scalar_input:
            return cls(a[0], b[0], c[0], d[0])
        return q

    # ------------------------------------------------------------------
    # Array interface
    # ------------------------------------------------------------------

    @property
    def shape(self):
        return self.a.shape

    @property
    def size(self):
        return self.a.size

    @property
    def ndim(self):
        return self.a.ndim

    def __len__(self):
        if self.a.ndim == 0:
            raise TypeError("scalar Quaternion has no len()")
        return self.a.shape[0]

    def __getitem__(self, idx):
        return Quaternion(self.a[idx], self.b[idx], self.c[idx], self.d[idx])

    def __repr__(self):
        if self.a.ndim == 0:
            return (f"Quaternion({self.a:.6f} + {self.b:.6f}i "
                    f"+ {self.c:.6f}j + {self.d:.6f}k)")
        return f"Quaternion array, shape={self.shape}"

    def components(self):
        """Return components as (..., 4) array [a, b, c, d]."""
        return np.stack([self.a, self.b, self.c, self.d], axis=-1)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __mul__(self, other):
        """Hamilton product q1 * q2, or scalar multiplication."""
        if isinstance(other, (int, float, np.ndarray)):
            return Quaternion(self.a * other, self.b * other,
                              self.c * other, self.d * other)
        if isinstance(other, Quaternion):
            a1, b1, c1, d1 = self.a, self.b, self.c, self.d
            a2, b2, c2, d2 = other.a, other.b, other.c, other.d
            return Quaternion(
                a1*a2 - b1*b2 - c1*c2 - d1*d2,
                a1*b2 + b1*a2 + c1*d2 - d1*c2,
                a1*c2 - b1*d2 + c1*a2 + d1*b2,
                a1*d2 + b1*c2 - c1*b2 + d1*a2,
            )
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            return self.__mul__(other)
        return NotImplemented

    def __neg__(self):
        return Quaternion(-self.a, -self.b, -self.c, -self.d)

    def __eq__(self, other):
        if not isinstance(other, Quaternion):
            return NotImplemented
        return (np.isclose(self.a, other.a) & np.isclose(self.b, other.b) &
                np.isclose(self.c, other.c) & np.isclose(self.d, other.d))

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def norm(self):
        """Quaternion magnitude."""
        return np.sqrt(self.a**2 + self.b**2 + self.c**2 + self.d**2)

    def normalize(self):
        """Return unit quaternion."""
        n = self.norm()
        return Quaternion(self.a / n, self.b / n, self.c / n, self.d / n)

    def conjugate(self):
        """Quaternion conjugate q* = a - bi - cj - dk."""
        return Quaternion(self.a, -self.b, -self.c, -self.d)

    def inv(self):
        """Quaternion inverse. For unit quaternions equals conjugate."""
        n2 = self.a**2 + self.b**2 + self.c**2 + self.d**2
        return Quaternion(self.a / n2, -self.b / n2, -self.c / n2, -self.d / n2)

    def dot(self, other):
        """Component-wise dot product (scalar)."""
        return (self.a * other.a + self.b * other.b +
                self.c * other.c + self.d * other.d)

    def dot_outer(self, other):
        """
        Outer dot product: result[i, j] = dot(self[i], other[j]).

        Returns
        -------
        np.ndarray, shape (len(self), len(other))
        """
        a1, b1 = self.a.ravel(), self.b.ravel()
        c1, d1 = self.c.ravel(), self.d.ravel()
        a2, b2 = other.a.ravel(), other.b.ravel()
        c2, d2 = other.c.ravel(), other.d.ravel()
        return (a1[:, None]*a2[None, :] + b1[:, None]*b2[None, :] +
                c1[:, None]*c2[None, :] + d1[:, None]*d2[None, :])

    # ------------------------------------------------------------------
    # Geometric properties
    # ------------------------------------------------------------------

    def angle(self):
        """Rotation angle in radians ∈ [0, π]."""
        return 2.0 * np.arccos(np.clip(np.abs(self.a), 0.0, 1.0))

    def axis(self):
        """
        Rotation axis as unit vector(s).

        Returns
        -------
        np.ndarray, shape (..., 3)
        """
        sin_half = np.sqrt(np.clip(1.0 - self.a**2, 0.0, None))
        safe = np.where(sin_half < 1e-10, 1.0, sin_half)
        return np.stack([self.b / safe, self.c / safe, self.d / safe], axis=-1)

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def to_matrix(self):
        """
        Convert to 3×3 rotation matrix (or N×3×3 for arrays).

        Uses the MTEX formula:
          R = 2*(v vᵀ) + 2a*skew(v) + (2a²−1)*I,  v = (b, c, d)
        """
        a, b, c, d = self.a, self.b, self.c, self.d
        scalar_input = a.ndim == 0
        a, b, c, d = (np.atleast_1d(x) for x in (a, b, c, d))
        n = a.size

        mat = np.empty((n, 3, 3))
        mat[:, 0, 0] = 2*(a*a + b*b) - 1
        mat[:, 0, 1] = 2*(b*c - a*d)
        mat[:, 0, 2] = 2*(b*d + a*c)
        mat[:, 1, 0] = 2*(c*b + a*d)
        mat[:, 1, 1] = 2*(a*a + c*c) - 1
        mat[:, 1, 2] = 2*(c*d - a*b)
        mat[:, 2, 0] = 2*(d*b - a*c)
        mat[:, 2, 1] = 2*(d*c + a*b)
        mat[:, 2, 2] = 2*(a*a + d*d) - 1

        return mat[0] if scalar_input else mat

    def to_euler(self, convention="ZYZ"):
        """
        Convert to Euler angles (radians).

        Conventions: 'ZYZ'/'Matthies'/'ABG' (default), 'Bunge'/'ZXZ'

        Returns
        -------
        tuple (angle1, angle2, angle3)
        """
        a, b, c, d = self.a, self.b, self.c, self.d

        # Matthies ZYZ
        p     = np.arctan2(d, a)   # = (alpha + gamma) / 2
        q_ang = np.arctan2(b, c)   # = (gamma - alpha) / 2

        alpha = p - q_ang
        beta  = 2.0 * np.arctan2(np.sqrt(b**2 + c**2), np.sqrt(a**2 + d**2))
        gamma = p + q_ang

        conv = convention.upper()
        if conv in ("ZYZ", "MATTHIES", "ABG"):
            return alpha, beta, gamma
        if conv in ("BUNGE", "ZXZ"):
            return alpha + np.pi / 2, beta, gamma + 3 * np.pi / 2
        raise ValueError(f"Unknown Euler convention: {convention!r}")

    def to_rodrigues(self):
        """
        Rodrigues–Frank vector r = tan(θ/2) * n̂.

        Returns
        -------
        np.ndarray, shape (..., 3)
        """
        sin_half = np.sqrt(np.clip(1.0 - self.a**2, 0.0, None))
        safe_a = np.where(np.abs(self.a) < 1e-10, 1e-10, self.a)
        t = sin_half / safe_a  # = tan(theta/2)
        safe_sin = np.where(sin_half < 1e-10, 1.0, sin_half)
        return np.stack([
            self.b / safe_sin * t,
            self.c / safe_sin * t,
            self.d / safe_sin * t,
        ], axis=-1)
