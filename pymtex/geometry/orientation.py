"""
Orientation – port of MTEX's @orientation.

An orientation is a rotation that carries a crystal symmetry (CS) and,
optionally, a specimen symmetry (SS).  The two key operations that make
orientations different from raw rotations are:

  * project2FundamentalRegion  – find the symmetry-equivalent representative
    with the smallest rotation angle.
  * calcMisorientation         – minimum rotation angle over all symmetry-
    equivalent misorientation representations.
"""

from __future__ import annotations

import numpy as np

from pymtex.geometry.rotation import Rotation
from pymtex.geometry.quaternion import Quaternion
from pymtex.geometry.symmetry import CrystalSymmetry


class Orientation(Rotation):
    """
    Crystallographic orientation = rotation + crystal symmetry.

    Parameters
    ----------
    a, b, c, d : array-like
        Quaternion components (MTEX convention: a is the scalar part).
    cs : CrystalSymmetry
        Crystal symmetry.
    ss : CrystalSymmetry, optional
        Specimen symmetry.  Defaults to triclinic ``'1'``.
    improper : bool or array-like, optional
        Improper flag(s) passed to :class:`Rotation`.

    Examples
    --------
    >>> cs = CrystalSymmetry('m-3m')
    >>> o = Orientation.byEuler(0, 0, 0, cs)          # cube component
    >>> o_cube = Orientation.cube(cs)
    >>> Orientation.byEuler(35.26, 45, 0, cs, degrees=True)  # Goss
    """

    def __init__(self, a, b, c, d, cs, ss=None, improper=False):
        super().__init__(a, b, c, d, improper=improper)
        if not isinstance(cs, CrystalSymmetry):
            raise TypeError("cs must be a CrystalSymmetry instance")
        self.cs = cs
        self.ss = ss if ss is not None else CrystalSymmetry('1')

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def byEuler(cls, phi1, Phi, phi2, cs, ss=None,
                convention='Bunge', degrees=False):
        """
        Create orientation(s) from Euler angles.

        Parameters
        ----------
        phi1, Phi, phi2 : float or array-like
            Euler angles in radians (or degrees if *degrees=True*).
        cs : CrystalSymmetry
        ss : CrystalSymmetry, optional
        convention : {'Bunge', 'ZYZ'}
            ``'Bunge'`` (ZXZ, default) is standard in materials science.
        degrees : bool
            If True, angles are in degrees.
        """
        if degrees:
            phi1 = np.deg2rad(phi1)
            Phi  = np.deg2rad(Phi)
            phi2 = np.deg2rad(phi2)
        q = Quaternion.from_euler(phi1, Phi, phi2, convention=convention)
        return cls(q.a, q.b, q.c, q.d, cs=cs, ss=ss)

    @classmethod
    def byAxisAngle(cls, axis, angle, cs, ss=None, degrees=False):
        """Create orientation from rotation axis and angle."""
        if degrees:
            angle = np.deg2rad(angle)
        q = Quaternion.from_axis_angle(axis, angle)
        return cls(q.a, q.b, q.c, q.d, cs=cs, ss=ss)

    @classmethod
    def byMatrix(cls, M, cs, ss=None):
        """Create orientation from a 3×3 (or N×3×3) rotation matrix."""
        q = Quaternion.from_matrix(M)
        return cls(q.a, q.b, q.c, q.d, cs=cs, ss=ss)

    @classmethod
    def rand(cls, n, cs, ss=None):
        """*n* uniformly random orientations."""
        q = Quaternion.rand(n=n)
        return cls(q.a, q.b, q.c, q.d, cs=cs, ss=ss)

    # --- standard texture components (MTEX naming) ----------------------

    @classmethod
    def cube(cls, cs, ss=None):
        """Cube component {001}<100>: φ1=Φ=φ2=0."""
        return cls.byEuler(0, 0, 0, cs, ss)

    @classmethod
    def goss(cls, cs, ss=None):
        """Goss component {110}<001>: φ1=0, Φ=45°, φ2=0."""
        return cls.byEuler(0, np.pi/4, 0, cs, ss)

    @classmethod
    def brass(cls, cs, ss=None):
        """Brass component {110}<112>: φ1=35.26°, Φ=45°, φ2=0."""
        return cls.byEuler(np.deg2rad(35.26), np.pi/4, 0, cs, ss)

    @classmethod
    def copper(cls, cs, ss=None):
        """Copper component {112}<111>: φ1=90°, Φ=35.26°, φ2=45°."""
        return cls.byEuler(np.pi/2, np.deg2rad(35.26), np.pi/4, cs, ss)

    @classmethod
    def rotationS(cls, cs, ss=None):
        """S component {123}<634>: φ1=59°, Φ=37°, φ2=63°."""
        return cls.byEuler(np.deg2rad(59), np.deg2rad(37), np.deg2rad(63),
                           cs, ss)

    # ------------------------------------------------------------------
    # Array interface (preserve cs/ss)
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        return Orientation(self.a[idx], self.b[idx], self.c[idx], self.d[idx],
                           cs=self.cs, ss=self.ss,
                           improper=self.improper[idx])

    def __repr__(self):
        if self.a.ndim == 0:
            phi1, Phi, phi2 = self.to_euler('Bunge')
            return (f"Orientation(φ1={np.degrees(phi1):.1f}°, "
                    f"Φ={np.degrees(Phi):.1f}°, "
                    f"φ2={np.degrees(phi2):.1f}°, cs={self.cs.name!r})")
        return (f"Orientation array, shape={self.shape}, "
                f"cs={self.cs.name!r}")

    # ------------------------------------------------------------------
    # Arithmetic (preserve cs/ss)
    # ------------------------------------------------------------------

    def __mul__(self, other):
        if isinstance(other, Orientation):
            q = Rotation.__mul__(self, other)
            return Orientation(q.a, q.b, q.c, q.d,
                               cs=self.cs, ss=self.ss,
                               improper=q.improper)
        return Rotation.__mul__(self, other)

    def inv(self):
        r = Rotation.inv(self)
        return Orientation(r.a, r.b, r.c, r.d,
                           cs=self.cs, ss=self.ss,
                           improper=r.improper)

    # ------------------------------------------------------------------
    # Symmetry operations
    # ------------------------------------------------------------------

    def symmetricEquivalent(self):
        """
        All crystal-symmetry-equivalent representations of self.

        Returns
        -------
        Orientation array of shape (nfold, *self.shape)
            Axis 0 indexes the symmetry operators.
        """
        sym = self.cs.rot  # Rotation array, shape (nfold,)
        nfold = self.cs.nfold

        # Broadcast: sym[i] * self
        sym_q = Quaternion(sym.a, sym.b, sym.c, sym.d)
        self_q = Quaternion(self.a, self.b, self.c, self.d)

        # For each sym s: q_result = s * self (Hamilton product)
        # Shapes: sym (nfold,), self (*shape) → result (nfold, *shape)
        a1 = sym.a.reshape(nfold, *([1]*self.a.ndim))
        b1 = sym.b.reshape(nfold, *([1]*self.b.ndim))
        c1 = sym.c.reshape(nfold, *([1]*self.c.ndim))
        d1 = sym.d.reshape(nfold, *([1]*self.d.ndim))

        a2, b2, c2, d2 = self.a, self.b, self.c, self.d

        ra = a1*a2 - b1*b2 - c1*c2 - d1*d2
        rb = a1*b2 + b1*a2 + c1*d2 - d1*c2
        rc = a1*c2 - b1*d2 + c1*a2 + d1*b2
        rd = a1*d2 + b1*c2 - c1*b2 + d1*a2

        imp = sym.improper.reshape(nfold, *([1]*self.improper.ndim)) \
              ^ self.improper

        return Orientation(ra, rb, rc, rd, cs=self.cs, ss=self.ss,
                           improper=imp)

    def project2FundamentalRegion(self):
        """
        Return the symmetry-equivalent orientation with the smallest rotation
        angle (i.e., |a| maximised ⟺ angle minimised).

        For scalar Orientations returns a scalar Orientation.
        For array Orientations returns an array of the same shape.
        """
        equiv = self.symmetricEquivalent()   # shape (nfold, *self.shape)
        # |a| is maximised at the minimum-angle representative
        abs_a = np.abs(equiv.a)              # shape (nfold, *self.shape)
        idx = np.argmax(abs_a, axis=0)       # shape (*self.shape)

        # Gather the selected elements
        def _gather(arr):
            # arr shape: (nfold, *shape)
            return np.take_along_axis(arr, idx[np.newaxis], axis=0)[0]

        return Orientation(
            _gather(equiv.a), _gather(equiv.b),
            _gather(equiv.c), _gather(equiv.d),
            cs=self.cs, ss=self.ss,
            improper=_gather(equiv.improper),
        )

    def calcMisorientation(self, other):
        """
        Minimum misorientation angle (radians) between *self* and *other*
        over all symmetry-equivalent representations.

        For each pair (self[i], other[i]) returns the minimum angle over
        { s1 * self[i]^{-1} * other[i] * s2 : s1, s2 ∈ CS }.

        Parameters
        ----------
        other : Orientation
            Must have the same crystal symmetry.

        Returns
        -------
        float or np.ndarray
            Misorientation angle(s) in radians.
        """
        # self^{-1} * other  (base misorientation)
        mis_base = self.inv() * other   # Rotation

        # All CS-equivalent: s * mis_base for s in CS
        # (only left-multiplication by CS needed for same-phase misorientation)
        sym = self.cs.rot
        nfold = self.cs.nfold

        a1 = sym.a.reshape(nfold, *([1]*mis_base.a.ndim))
        b1 = sym.b.reshape(nfold, *([1]*mis_base.b.ndim))
        c1 = sym.c.reshape(nfold, *([1]*mis_base.c.ndim))
        d1 = sym.d.reshape(nfold, *([1]*mis_base.d.ndim))

        a2, b2, c2, d2 = mis_base.a, mis_base.b, mis_base.c, mis_base.d

        ra = a1*a2 - b1*b2 - c1*c2 - d1*d2
        # Misorientation angle = 2 * arccos(|a|)
        max_abs_a = np.max(np.abs(ra), axis=0)
        return 2.0 * np.arccos(np.clip(max_abs_a, 0.0, 1.0))

    def angle_between(self, other):
        """Alias for :meth:`calcMisorientation`."""
        return self.calcMisorientation(other)

    # ------------------------------------------------------------------
    # Euler angle convenience
    # ------------------------------------------------------------------

    def toEuler(self, convention='Bunge', degrees=False):
        """
        Return Euler angles.

        Parameters
        ----------
        convention : {'Bunge', 'ZYZ'}
        degrees : bool

        Returns
        -------
        tuple of (phi1, Phi, phi2)
        """
        alpha, beta, gamma = self.to_euler(convention)
        if degrees:
            return np.degrees(alpha), np.degrees(beta), np.degrees(gamma)
        return alpha, beta, gamma
