"""
PoleFigure – port of MTEX's @PoleFigure.

Stores measured pole figure data: for each crystal direction h (a Miller index),
the measured intensities at a set of specimen directions r on the unit sphere.

The relationship between the ODF f(g) and a pole figure P_h(r) is:

    P_h(r) = ∫_{SO(3)} f(g) · δ(R(g)·ĥ − r̂) dg

i.e. P_h(r) is the marginal of the ODF over all orientations whose rotation
maps the crystal direction h onto the specimen direction r.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from pymtex.geometry.symmetry import CrystalSymmetry
from pymtex.geometry.miller import Miller


def _spherical_to_xyz(theta, phi):
    """Convert (inclination θ, azimuth φ) to unit Cartesian vector."""
    return np.stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ], axis=-1)


def _xyz_to_spherical(v):
    """Convert unit Cartesian vector to (θ, φ)."""
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    theta = np.arccos(np.clip(v[..., 2], -1.0, 1.0))
    phi   = np.arctan2(v[..., 1], v[..., 0])
    return theta, phi


class PoleFigure:
    """
    Measured pole figure dataset.

    Each pole figure associates a crystal direction h (given as a
    :class:`~pymtex.geometry.miller.Miller` index) with a set of specimen
    directions **r** and the measured diffraction intensity at each direction.

    Parameters
    ----------
    h : Miller or list of Miller
        Crystal pole(s) being measured, e.g. ``Miller(hkl=[1,0,0], cs=cs)``.
    r : array-like (M, 3) or list of array-like
        Unit vectors of specimen measurement directions.
        Pass a single array to reuse the same grid for all poles.
    intensities : array-like (M,) or list of array-like
        Measured intensities at each *r* direction.
    cs : CrystalSymmetry
    ss : CrystalSymmetry, optional
        Specimen symmetry (default: triclinic ``'1'``).

    Examples
    --------
    Build a synthetic pole figure from a known ODF and add noise:

    >>> import numpy as np
    >>> from pymtex import CrystalSymmetry, Orientation, Miller
    >>> from pymtex.texture import PoleFigure
    >>> cs = CrystalSymmetry('m-3m')
    >>> theta = np.linspace(0, np.pi/2, 19)
    >>> phi   = np.linspace(0, 2*np.pi, 37)[:-1]
    >>> TH, PH = np.meshgrid(theta, phi)
    >>> r = _spherical_to_xyz(TH.ravel(), PH.ravel())
    >>> h100 = Miller(hkl=[1,0,0], cs=cs)
    >>> I = np.ones(len(r))           # uniform texture
    >>> pf = PoleFigure(h100, r, I, cs)
    """

    def __init__(self, h, r, intensities, cs, ss=None):
        # Normalise to lists
        if isinstance(h, Miller):
            h = [h]
        if not isinstance(r, list):
            r = [np.asarray(r, dtype=float)] * len(h)
        if not isinstance(intensities, list):
            intensities = [np.asarray(intensities, dtype=float)] * len(h)

        if len(h) != len(r) or len(h) != len(intensities):
            raise ValueError(
                "h, r, and intensities must have the same length."
            )

        if not isinstance(cs, CrystalSymmetry):
            raise TypeError("cs must be a CrystalSymmetry instance.")

        self.h = h
        self.r = [np.asarray(ri, dtype=float) for ri in r]
        # Normalise each r to unit length
        for i, ri in enumerate(self.r):
            norms = np.linalg.norm(ri, axis=-1, keepdims=True)
            self.r[i] = ri / np.where(norms < 1e-15, 1.0, norms)

        self.intensities = [np.asarray(Ii, dtype=float).ravel()
                            for Ii in intensities]
        self.cs = cs
        self.ss = ss if ss is not None else CrystalSymmetry('1')

        # Validate
        for j, (rj, Ij) in enumerate(zip(self.r, self.intensities)):
            if rj.shape[0] != Ij.shape[0]:
                raise ValueError(
                    f"Pole {j}: r has {rj.shape[0]} rows but "
                    f"intensities has {Ij.shape[0]} entries."
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def numPF(self):
        """Number of pole figures."""
        return len(self.h)

    @property
    def numPoints(self):
        """List of measurement-point counts per pole figure."""
        return [len(Ij) for Ij in self.intensities]

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_spherical(cls, h, theta, phi, intensities, cs, ss=None):
        """
        Construct from (inclination, azimuth) angles.

        Parameters
        ----------
        theta : array-like (M,) – inclination from specimen Z in radians.
        phi   : array-like (M,) – azimuthal angle in radians.
        """
        theta = np.asarray(theta, dtype=float)
        phi   = np.asarray(phi,   dtype=float)
        r = _spherical_to_xyz(theta, phi)
        return cls(h, r, intensities, cs, ss)

    @classmethod
    def from_grid(cls, h, theta_max=np.pi/2,
                  n_theta=19, n_phi=36,
                  intensities=None, cs=None, ss=None):
        """
        Construct with a regular (θ, φ) grid on the upper hemisphere.

        Parameters
        ----------
        theta_max : float
            Maximum tilt angle (radians).  Default π/2 (full hemisphere).
        n_theta, n_phi : int
            Grid density.
        intensities : array-like (n_theta*n_phi,) or None
            Leave None to create a placeholder (for forward modelling).
        """
        theta = np.linspace(0, theta_max, n_theta)
        phi   = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        r = _spherical_to_xyz(TH.ravel(), PH.ravel())
        if intensities is None:
            intensities = np.ones(len(r))
        return cls(h, r, intensities, cs, ss)

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def normalize(self):
        """Return copy with each pole figure normalised to unit mean."""
        new_I = []
        for Ij in self.intensities:
            mean = Ij.mean()
            new_I.append(Ij / mean if mean > 0 else Ij.copy())
        return PoleFigure(self.h, [r.copy() for r in self.r], new_I,
                          self.cs, self.ss)

    def correct_background(self, background):
        """Subtract constant background per pole figure."""
        if not isinstance(background, (list, tuple, np.ndarray)):
            background = [background] * self.numPF
        new_I = [np.maximum(Ij - bg, 0)
                 for Ij, bg in zip(self.intensities, background)]
        return PoleFigure(self.h, [r.copy() for r in self.r], new_I,
                          self.cs, self.ss)

    # ------------------------------------------------------------------
    # Interpolation on the sphere
    # ------------------------------------------------------------------

    def interp(self, j, r_query):
        """
        Nearest-neighbour intensity lookup for pole figure *j* at
        query directions *r_query* (N, 3).

        Returns
        -------
        np.ndarray (N,)
        """
        r_j = self.r[j]       # (M, 3)
        I_j = self.intensities[j]   # (M,)
        r_q = np.asarray(r_query, dtype=float)

        # cos(angle) = dot product (both are unit vectors)
        cos_ang = r_j @ r_q.T   # (M, N)
        nearest = np.argmax(cos_ang, axis=0)  # (N,)
        return I_j[nearest]

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, cmap='jet', plot_type='scatter', levels=15,
             projection='equal_area', upper_hemisphere=True,
             suptitle='Measured pole figures'):
        """
        Plot all measured pole figures.

        Layout rules
        ------------
        * Max **3** pole figures per row.
        * Max **4** rows per figure.
        * If more than 12 pole figures, the overflow continues in a new figure.

        Parameters
        ----------
        cmap : str
            Colormap (default ``'jet'``).
        plot_type : 'scatter' (default, point-by-point) or 'contour'
            ``'scatter'`` draws each measured point as a coloured dot.
            ``'contour'`` interpolates onto a regular grid and draws filled
            contours with thin contour-line overlay.
        levels : int
            Number of contour levels (only for ``plot_type='contour'``).
        projection : 'equal_area' (default, Schmidt net) or 'stereo'
        upper_hemisphere : bool
            Only plot points with r_z ≥ 0 (default True).
        suptitle : str or None
            Super-title for each figure.

        Returns
        -------
        matplotlib.figure.Figure  – if everything fits on one figure
        list of matplotlib.figure.Figure  – if multiple figures are needed
        """
        from pymtex.texture._plot_utils import plot_pf_grid

        figs = plot_pf_grid(
            self.h, self.r, self.intensities,
            cmap=cmap, plot_type=plot_type, levels=levels,
            projection=projection, upper_hemisphere=upper_hemisphere,
            suptitle=suptitle,
        )
        return figs[0] if len(figs) == 1 else figs

    def __repr__(self):
        poles = ', '.join(str(h) for h in self.h)
        return (f"PoleFigure({self.numPF} poles: {poles}, "
                f"cs={self.cs.name!r})")
