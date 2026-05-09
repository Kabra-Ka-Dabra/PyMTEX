"""
EBSD – port of MTEX's @EBSD data container.

Stores spatially indexed orientation measurements (one per pixel/point)
along with phase labels and metadata.
"""

from __future__ import annotations

import numpy as np
from pymtex.geometry.orientation import Orientation
from pymtex.geometry.symmetry import CrystalSymmetry


class EBSD:
    """
    Electron Backscatter Diffraction dataset.

    Parameters
    ----------
    x, y : array-like, shape (N,)
        Spatial coordinates of measurement points.
    orientations : Orientation, shape (N,)
        Measured orientations.
    phase : array-like of int, shape (N,), optional
        Phase index per point (0 = not indexed).
    phase_names : list of str, optional
        Name for each phase index (index 0 = 'notIndexed').

    Examples
    --------
    >>> cs = CrystalSymmetry('m-3m')
    >>> x = np.linspace(0, 10, 20)
    >>> y = np.zeros(20)
    >>> ori = Orientation.rand(20, cs)
    >>> ebsd = EBSD(x, y, ori)
    >>> ebsd.numPixels
    20
    """

    def __init__(self, x, y, orientations, phase=None, phase_names=None):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        n = x.size

        if not isinstance(orientations, Orientation):
            raise TypeError("orientations must be an Orientation instance.")
        if orientations.size != n:
            raise ValueError(
                f"Length mismatch: x/y have {n} points, "
                f"orientations has {orientations.size}."
            )

        self.x = x
        self.y = y
        self.orientations = orientations
        self.cs = orientations.cs

        if phase is None:
            self.phase = np.ones(n, dtype=int)
        else:
            self.phase = np.asarray(phase, dtype=int).ravel()

        if phase_names is None:
            self.phase_names = {0: 'notIndexed', 1: self.cs.name}
        else:
            self.phase_names = {i: name for i, name in enumerate(phase_names)}

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def numPixels(self):
        """Total number of measurement points."""
        return self.x.size

    @property
    def stepSize(self):
        """Estimated step size (median nearest-neighbour distance in x)."""
        xs = np.sort(np.unique(self.x))
        if xs.size < 2:
            return 1.0
        return float(np.median(np.diff(xs)))

    @property
    def isIndexed(self):
        """Boolean mask: True where a valid orientation was measured."""
        return self.phase > 0

    @property
    def boundingBox(self):
        """(xmin, xmax, ymin, ymax)."""
        return self.x.min(), self.x.max(), self.y.min(), self.y.max()

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter(self, mask):
        """
        Return a new EBSD restricted to points where *mask* is True.

        Parameters
        ----------
        mask : array-like of bool, shape (N,)
        """
        mask = np.asarray(mask, dtype=bool).ravel()
        ori = self.orientations[mask]
        return EBSD(
            self.x[mask], self.y[mask], ori,
            phase=self.phase[mask],
            phase_names=list(self.phase_names.values()),
        )

    def indexed(self):
        """Return EBSD with only indexed points (phase > 0)."""
        return self.filter(self.isIndexed)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def meanOrientation(self):
        """
        Mean orientation computed as the orientation closest to the
        quaternion mean (a simple approximation valid for tight clusters).
        """
        # Quaternion mean (not rigorous on SO(3) but practical for small spreads)
        a = self.orientations.a
        b = self.orientations.b
        c = self.orientations.c
        d = self.orientations.d

        # Flip signs so all quats are in the same hemisphere as the first
        signs = np.sign(np.sum(
            np.stack([a, b, c, d], axis=-1) *
            np.array([a[0], b[0], c[0], d[0]]),
            axis=-1
        ))
        signs = np.where(signs == 0, 1.0, signs)

        a_m = np.mean(a * signs)
        b_m = np.mean(b * signs)
        c_m = np.mean(c * signs)
        d_m = np.mean(d * signs)
        norm = np.sqrt(a_m**2 + b_m**2 + c_m**2 + d_m**2)
        return Orientation(a_m/norm, b_m/norm, c_m/norm, d_m/norm,
                           cs=self.cs)

    def calcKAM(self, max_angle=np.deg2rad(5)):
        """
        Kernel Average Misorientation (KAM) in radians.

        Computes the mean misorientation angle between each point and its
        4-connected neighbours, ignoring pairs beyond *max_angle*.

        Returns
        -------
        np.ndarray, shape (N,)
            KAM value per point (NaN for boundary/isolated points).
        """
        # Build coordinate-to-index lookup
        step = self.stepSize
        xi = np.round(self.x / step).astype(int)
        yi = np.round(self.y / step).astype(int)

        lookup = {}
        for idx, (xi_, yi_) in enumerate(zip(xi, yi)):
            lookup[(xi_, yi_)] = idx

        ori = self.orientations
        kam = np.full(self.numPixels, np.nan)

        for i in range(self.numPixels):
            neighbours = []
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                j = lookup.get((xi[i]+dx, yi[i]+dy))
                if j is not None:
                    # scalar misorientation between point i and j
                    o_i = Orientation(
                        float(ori.a[i]), float(ori.b[i]),
                        float(ori.c[i]), float(ori.d[i]),
                        cs=self.cs
                    )
                    o_j = Orientation(
                        float(ori.a[j]), float(ori.b[j]),
                        float(ori.c[j]), float(ori.d[j]),
                        cs=self.cs
                    )
                    ang = float(o_i.calcMisorientation(o_j))
                    if ang <= max_angle:
                        neighbours.append(ang)
            if neighbours:
                kam[i] = float(np.mean(neighbours))

        return kam

    # ------------------------------------------------------------------
    # Grain segmentation (simple threshold)
    # ------------------------------------------------------------------

    def calcGrains(self, threshold=np.deg2rad(10)):
        """
        Simple flood-fill grain segmentation.

        Two neighbouring pixels belong to the same grain if their
        misorientation angle is below *threshold*.

        Parameters
        ----------
        threshold : float
            Misorientation threshold in radians.

        Returns
        -------
        grain_id : np.ndarray of int, shape (N,)
            Grain index per pixel (starting at 1; 0 = unassigned).
        num_grains : int
        """
        step = self.stepSize
        xi = np.round(self.x / step).astype(int)
        yi = np.round(self.y / step).astype(int)

        lookup = {}
        for idx, (xi_, yi_) in enumerate(zip(xi, yi)):
            lookup[(xi_, yi_)] = idx

        grain_id = np.zeros(self.numPixels, dtype=int)
        current_grain = 0
        ori = self.orientations

        for start in range(self.numPixels):
            if grain_id[start] != 0:
                continue
            current_grain += 1
            queue = [start]
            grain_id[start] = current_grain

            while queue:
                i = queue.pop()
                o_i = Orientation(
                    float(ori.a[i]), float(ori.b[i]),
                    float(ori.c[i]), float(ori.d[i]),
                    cs=self.cs
                )
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    j = lookup.get((xi[i]+dx, yi[i]+dy))
                    if j is None or grain_id[j] != 0:
                        continue
                    o_j = Orientation(
                        float(ori.a[j]), float(ori.b[j]),
                        float(ori.c[j]), float(ori.d[j]),
                        cs=self.cs
                    )
                    if float(o_i.calcMisorientation(o_j)) < threshold:
                        grain_id[j] = current_grain
                        queue.append(j)

        return grain_id, current_grain

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self):
        xmin, xmax, ymin, ymax = self.boundingBox
        return (
            f"EBSD({self.numPixels} points, "
            f"cs={self.cs.name!r}, "
            f"x=[{xmin:.2g}, {xmax:.2g}], y=[{ymin:.2g}, {ymax:.2g}])"
        )
