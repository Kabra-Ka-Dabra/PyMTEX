"""
Miller – port of MTEX's @Miller.

Represents crystallographic directions [uvw] and plane normals (hkl) as
3-D vectors in the crystal reference frame.  Hexagonal Miller-Bravais
notation (hkil / [UVTW]) is also supported.
"""

from __future__ import annotations

import numpy as np
from pymtex.geometry.symmetry import CrystalSymmetry


class Miller:
    """
    Crystallographic Miller indices.

    Pass exactly one of *hkl* (plane normals) or *uvw* (directions).

    Parameters
    ----------
    hkl : array-like, shape (3,) or (N, 3), optional
        Plane-normal indices (h, k, l).
    uvw : array-like, shape (3,) or (N, 3), optional
        Direction indices [u, v, w].
    cs : CrystalSymmetry
        Crystal symmetry used for angle / symmetry calculations.

    Examples
    --------
    >>> cs = CrystalSymmetry('m-3m')
    >>> n = Miller(hkl=[1, 1, 0], cs=cs)
    >>> d = Miller(uvw=[1, 1, 1], cs=cs)
    >>> n.angle(Miller(hkl=[0, 0, 1], cs=cs), degrees=True)
    45.0
    """

    def __init__(self, hkl=None, uvw=None, cs=None):
        if (hkl is None) == (uvw is None):
            raise ValueError("Provide exactly one of hkl or uvw.")
        if cs is None:
            raise ValueError("cs (CrystalSymmetry) is required.")
        if not isinstance(cs, CrystalSymmetry):
            raise TypeError("cs must be a CrystalSymmetry instance.")

        self.cs = cs

        if hkl is not None:
            data = np.asarray(hkl, dtype=float)
            self._type = 'hkl'
        else:
            data = np.asarray(uvw, dtype=float)
            self._type = 'uvw'

        # Ensure shape (..., 3)
        if data.ndim == 1:
            if data.shape[0] == 4:
                # Miller-Bravais hkil → hkl
                data = self._hkil_to_hkl(data) if self._type == 'hkl' \
                    else self._UVTW_to_uvw(data)
            data = data.reshape(3)
        elif data.shape[-1] == 4:
            if self._type == 'hkl':
                data = np.apply_along_axis(self._hkil_to_hkl, -1, data)
            else:
                data = np.apply_along_axis(self._UVTW_to_uvw, -1, data)

        self._v = data  # shape (3,) or (N, 3)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def hkl(self):
        """Miller indices as (h, k, l) array.  Only valid for hkl type."""
        if self._type != 'hkl':
            raise AttributeError("This Miller object stores directions [uvw].")
        return self._v

    @property
    def uvw(self):
        """Direction indices as [u, v, w] array.  Only valid for uvw type."""
        if self._type != 'uvw':
            raise AttributeError("This Miller object stores plane normals (hkl).")
        return self._v

    @property
    def h(self): return self._v[..., 0]
    @property
    def k(self): return self._v[..., 1]
    @property
    def l(self): return self._v[..., 2]

    @property
    def type(self):
        """``'hkl'`` or ``'uvw'``."""
        return self._type

    @property
    def shape(self):
        return self._v.shape[:-1] if self._v.ndim > 1 else ()

    @property
    def size(self):
        return self._v.shape[0] if self._v.ndim > 1 else 1

    # ------------------------------------------------------------------
    # Cartesian vector
    # ------------------------------------------------------------------

    def toVector3d(self):
        """
        Return the corresponding unit vector(s) in the Cartesian crystal frame.

        For cubic crystals the crystal and Cartesian frames coincide, so this
        simply normalises the index triplet.  For non-cubic systems the
        lattice metric would be needed; this base implementation assumes
        orthonormal axes (valid for cubic/tetragonal/orthorhombic).
        """
        v = self._v.copy().astype(float)
        norms = np.linalg.norm(v, axis=-1, keepdims=True)
        norms = np.where(norms < 1e-15, 1.0, norms)
        return v / norms

    # ------------------------------------------------------------------
    # Angle between two Miller objects
    # ------------------------------------------------------------------

    def angle(self, other, degrees=False):
        """
        Angle between *self* and *other* (smallest positive value).

        Parameters
        ----------
        other : Miller
            Must have the same *cs* and same *type* (both hkl or both uvw).
        degrees : bool

        Returns
        -------
        float or np.ndarray
        """
        v1 = self.toVector3d()
        v2 = other.toVector3d()
        # dot product; shapes may broadcast
        cos_a = np.clip(np.sum(v1 * v2, axis=-1), -1.0, 1.0)
        ang = np.arccos(np.abs(cos_a))   # abs → smallest angle
        return np.degrees(ang) if degrees else ang

    # ------------------------------------------------------------------
    # Symmetrically equivalent indices
    # ------------------------------------------------------------------

    def symmetricEquivalent(self, unique=True):
        """
        All crystallographically equivalent Miller indices under *cs*.

        Parameters
        ----------
        unique : bool
            If True (default) return only distinct directions/normals.

        Returns
        -------
        Miller  – possibly with extra leading dimension for the equivalents.
        """
        sym_rot = self.cs.properRotations   # proper operators (det = +1)
        v = self.toVector3d()               # (..., 3)

        # Rotate v by each symmetry operator
        mats = sym_rot.to_matrix()          # (nfold, 3, 3)
        if v.ndim == 1:
            rotated = np.einsum('nij,j->ni', mats, v)   # (nfold, 3)
        else:
            rotated = np.einsum('nij,...j->n...i', mats, v)

        if unique:
            # De-duplicate by rounding and keeping unique rows
            rounded = np.round(rotated, 8)
            # Flatten to 2D, find unique
            orig_shape = rounded.shape
            flat = rounded.reshape(-1, 3)
            _, idx = np.unique(flat, axis=0, return_index=True)
            flat_unique = flat[np.sort(idx)]
            rotated = flat_unique

        if self._type == 'hkl':
            return Miller(hkl=rotated, cs=self.cs)
        return Miller(uvw=rotated, cs=self.cs)

    # ------------------------------------------------------------------
    # Multiplicities
    # ------------------------------------------------------------------

    @property
    def multiplicity(self):
        """Number of symmetrically equivalent indices."""
        return self.symmetricEquivalent(unique=True).size

    # ------------------------------------------------------------------
    # Repr / string output
    # ------------------------------------------------------------------

    def __repr__(self):
        v = self._v
        if self._type == 'hkl':
            if v.ndim == 1:
                return f"Miller(hkl=({v[0]:.0f} {v[1]:.0f} {v[2]:.0f}), cs={self.cs.name!r})"
            return f"Miller(hkl, shape={self.shape}, cs={self.cs.name!r})"
        else:
            if v.ndim == 1:
                return f"Miller(uvw=[{v[0]:.0f} {v[1]:.0f} {v[2]:.0f}], cs={self.cs.name!r})"
            return f"Miller(uvw, shape={self.shape}, cs={self.cs.name!r})"

    def __len__(self):
        if self._v.ndim == 1:
            raise TypeError("scalar Miller has no len()")
        return self._v.shape[0]

    def __getitem__(self, idx):
        v = self._v[idx]
        if self._type == 'hkl':
            return Miller(hkl=v, cs=self.cs)
        return Miller(uvw=v, cs=self.cs)

    # ------------------------------------------------------------------
    # Miller-Bravais conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hkil_to_hkl(hkil):
        """Convert 4-index (h k i l) to 3-index (h k l)."""
        h, k, i, l = hkil
        # i = -(h+k) by definition; drop i
        return np.array([h, k, l], dtype=float)

    @staticmethod
    def _UVTW_to_uvw(UVTW):
        """Convert 4-index hexagonal direction [U V T W] to [u v w]."""
        U, V, T, W = UVTW
        u = 2*U + V
        v = 2*V + U
        w = W
        return np.array([u, v, w], dtype=float)

    @staticmethod
    def _uvw_to_UVTW(uvw):
        """Convert [u v w] to hexagonal [U V T W]."""
        u, v, w = uvw
        U = (2*u - v) / 3
        V = (2*v - u) / 3
        T = -(U + V)
        W = w
        return np.array([U, V, T, W])

    # ------------------------------------------------------------------
    # Normalise
    # ------------------------------------------------------------------

    def normalize(self):
        """Return normalised (unit-length) copy."""
        v = self.toVector3d()
        if self._type == 'hkl':
            return Miller(hkl=v, cs=self.cs)
        return Miller(uvw=v, cs=self.cs)

    # ------------------------------------------------------------------
    # Dot product
    # ------------------------------------------------------------------

    def dot(self, other):
        """Dot product of the unit vectors."""
        return np.sum(self.toVector3d() * other.toVector3d(), axis=-1)
