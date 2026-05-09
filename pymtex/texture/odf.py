"""
ODF – Orientation Distribution Function, port of MTEX's @ODF.

The ODF f: SO(3) → R⁺ describes the probability density of crystallite
orientations in a polycrystalline material.  This implementation stores the
ODF as a discrete set of values on a regular SO(3) grid (Euler angle grid)
together with the volume weights of each cell.

The ODF satisfies the normalisation condition:
    ∫_{SO(3)} f(g) dg = 1    (dg = 1/(8π²) sin Φ dφ₁ dΦ dφ₂)

Random texture corresponds to f(g) = 1 everywhere.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.interpolate import griddata
from scipy.spatial import KDTree

from pymtex.geometry.quaternion import Quaternion
from pymtex.geometry.rotation import Rotation
from pymtex.geometry.orientation import Orientation
from pymtex.geometry.symmetry import CrystalSymmetry
from pymtex.geometry.miller import Miller


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_hemisphere(a, b, c, d):
    """Project quaternions onto the hemisphere with a ≥ 0 (q ≡ −q)."""
    s = np.where(a < 0, -1.0, 1.0)
    return a * s, b * s, c * s, d * s


def _proj_equal_area(r):
    """Equal-area (Schmidt) projection of unit vectors r (N,3) → (x, y)."""
    theta = np.arccos(np.clip(r[:, 2], -1, 1))
    phi   = np.arctan2(r[:, 1], r[:, 0])
    rho   = np.sin(theta / 2) * np.sqrt(2)
    return rho * np.cos(phi), rho * np.sin(phi)


def _draw_circle(ax, lw=0.8):
    th = np.linspace(0, 2 * np.pi, 360)
    ax.plot(np.cos(th), np.sin(th), 'k-', lw=lw)


def _pf_axes(ax):
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')


def _phi2_fundamental(cs):
    """Maximum φ₂ in the fundamental Euler domain (degrees)."""
    s = cs.system
    return {'cubic': 90, 'hexagonal': 60, 'trigonal': 120,
            'tetragonal': 90, 'orthorhombic': 90,
            'monoclinic': 180}.get(s, 360)


# ---------------------------------------------------------------------------
# ODF
# ---------------------------------------------------------------------------

class ODF:
    """
    Orientation Distribution Function on a discrete SO(3) grid.

    Parameters
    ----------
    orientations : Orientation, shape (N,)
        Grid points in SO(3).
    f : np.ndarray, shape (N,)
        ODF values at each grid point (m.r.d., ≥ 0).
    weights : np.ndarray, shape (N,)
        Volume weight of each cell.  Should sum to 1.
    cs : CrystalSymmetry
    ss : CrystalSymmetry, optional
    """

    def __init__(self, orientations, f, weights, cs, ss=None):
        if not isinstance(orientations, Orientation):
            raise TypeError("orientations must be an Orientation instance.")
        self.orientations = orientations
        self.f       = np.asarray(f,       dtype=float).ravel()
        self.weights = np.asarray(weights, dtype=float).ravel()
        self.cs = cs
        self.ss = ss if ss is not None else CrystalSymmetry('1')
        self._kdtree = None

    # ------------------------------------------------------------------
    # KD-tree for fast nearest-neighbour eval
    # ------------------------------------------------------------------

    def _build_kdtree(self):
        a, b, c, d = _to_hemisphere(
            self.orientations.a, self.orientations.b,
            self.orientations.c, self.orientations.d,
        )
        self._kdtree = KDTree(np.stack([a, b, c, d], axis=-1))

    # ------------------------------------------------------------------
    # Statistical properties
    # ------------------------------------------------------------------

    @property
    def size(self):
        return self.f.size

    def texture_index(self):
        """J = ∫ f(g)² dg.  J = 1 for random texture."""
        return float(np.sum(self.f**2 * self.weights))

    def entropy(self):
        """H = −∫ f(g) log f(g) dg."""
        mask = self.f > 0
        return float(-np.sum(self.f[mask] * np.log(self.f[mask])
                             * self.weights[mask]))

    def _mean_orientation(self):
        a = np.sum(self.orientations.a * self.f * self.weights)
        b = np.sum(self.orientations.b * self.f * self.weights)
        c = np.sum(self.orientations.c * self.f * self.weights)
        d = np.sum(self.orientations.d * self.f * self.weights)
        n = np.sqrt(a**2 + b**2 + c**2 + d**2)
        return Orientation(a/n, b/n, c/n, d/n, cs=self.cs)

    # ------------------------------------------------------------------
    # Evaluation at arbitrary orientations
    # ------------------------------------------------------------------

    def eval(self, ori):
        """
        Evaluate the ODF by nearest-neighbour on the SO(3) grid.

        Parameters
        ----------
        ori : Orientation, shape (M,) or scalar

        Returns
        -------
        np.ndarray (M,)
        """
        if self._kdtree is None:
            self._build_kdtree()

        a, b, c, d = _to_hemisphere(
            np.atleast_1d(ori.a).ravel(),
            np.atleast_1d(ori.b).ravel(),
            np.atleast_1d(ori.c).ravel(),
            np.atleast_1d(ori.d).ravel(),
        )
        pts = np.stack([a, b, c, d], axis=-1)
        _, idx = self._kdtree.query(pts)
        return self.f[idx]

    # ------------------------------------------------------------------
    # Forward model: ODF → pole figure
    # ------------------------------------------------------------------

    def calcPF(self, h, r=None, n_theta=19, n_phi=72):
        """
        Compute the pole figure for crystal direction *h* from the ODF.

        Returns
        -------
        r  : (M, 3)  specimen unit vectors
        pf : (M,)    intensities (m.r.d.)
        """
        if r is None:
            theta = np.linspace(0, np.pi/2, n_theta)
            phi   = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
            TH, PH = np.meshgrid(theta, phi, indexing='ij')
            r = np.stack([
                np.sin(TH.ravel()) * np.cos(PH.ravel()),
                np.sin(TH.ravel()) * np.sin(PH.ravel()),
                np.cos(TH.ravel()),
            ], axis=-1)
        else:
            r = np.asarray(r, dtype=float)
            r = r / np.linalg.norm(r, axis=-1, keepdims=True)

        h_vec = h.toVector3d()
        mats  = self.orientations.to_matrix()     # (N, 3, 3)
        r_ij  = np.einsum('nij,j->ni', mats, h_vec)  # (N, 3)

        M = r.shape[0]
        pf_vals   = np.zeros(M)
        pf_weight = np.zeros(M)

        cos_sim = r_ij @ r.T                        # (N, M)
        nearest = np.argmax(cos_sim, axis=1)        # (N,)

        np.add.at(pf_vals,   nearest, self.f * self.weights)
        np.add.at(pf_weight, nearest, self.weights)

        with np.errstate(invalid='ignore', divide='ignore'):
            pf = np.where(pf_weight > 0, pf_vals / pf_weight, 0.0)

        mean_pf = pf.mean()
        if mean_pf > 0:
            pf /= mean_pf

        return r, pf

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _plot_pf_scatter(ax, r, I, title, vmin, vmax, cmap):
        mask  = r[:, 2] >= -0.05
        x, y  = _proj_equal_area(r[mask])
        sc    = ax.scatter(x, y, c=I[mask], cmap=cmap, s=4,
                           vmin=vmin, vmax=vmax)
        _draw_circle(ax)
        _pf_axes(ax)
        ax.set_title(title, fontsize=9)
        return sc

    @staticmethod
    def _plot_pf_contour(ax, r, I, title, vmin, vmax, cmap, levels):
        mask  = r[:, 2] >= -0.05
        x, y  = _proj_equal_area(r[mask])
        I_up  = I[mask]

        xi = np.linspace(-1, 1, 150)
        yi = np.linspace(-1, 1, 150)
        XX, YY = np.meshgrid(xi, yi)
        ZZ = griddata((x, y), I_up, (XX, YY), method='linear')
        ZZ[XX**2 + YY**2 > 1.01] = np.nan

        cf = ax.contourf(XX, YY, ZZ, levels=levels, cmap=cmap,
                         vmin=vmin, vmax=vmax)
        ax.contour(XX, YY, ZZ, levels=levels,
                   colors='k', linewidths=0.25, alpha=0.4)
        _draw_circle(ax)
        _pf_axes(ax)
        ax.set_title(title, fontsize=9)
        return cf

    # ------------------------------------------------------------------
    # Pole figure plots (single and multi)
    # ------------------------------------------------------------------

    def plotPF(self, h, ax=None, plot_type='contour', cmap='jet',
               n_theta=19, n_phi=72, levels=15):
        """
        Plot the computed pole figure for one crystal direction *h*.

        Parameters
        ----------
        h : Miller
        ax : matplotlib Axes, optional
        plot_type : 'contour' (default) or 'scatter'
        cmap : str  (default ``'jet'``)
        levels : int  (number of contour levels)
        """
        from pymtex.texture._plot_utils import render_pf

        r, pf = self.calcPF(h, n_theta=n_theta, n_phi=n_phi)

        if ax is None:
            _, ax = plt.subplots(figsize=(4, 4))

        hkl = '({}{}{})'.format(int(h.h), int(h.k), int(h.l))
        sc = render_pf(ax, r, pf, hkl,
                       cmap=cmap, plot_type=plot_type, levels=levels)
        plt.colorbar(sc, ax=ax, label='m.r.d.', fraction=0.046, pad=0.04)
        return ax

    def plotPFs(self, h_list, cmap='jet', plot_type='contour', levels=15,
                n_theta=19, n_phi=72,
                suptitle='Calculated pole figures'):
        """
        Plot computed pole figures for a list of crystal directions.

        Layout rules
        ------------
        * Max **3** pole figures per row.
        * Max **4** rows per figure.
        * Overflow goes into additional figures.

        Parameters
        ----------
        h_list : list of Miller
        cmap : str  (default ``'jet'``)
        plot_type : 'contour' (default) or 'scatter'
        levels : int
        n_theta, n_phi : int  – grid density for forward projection
        suptitle : str or None

        Returns
        -------
        matplotlib.figure.Figure  – if one figure is enough
        list of matplotlib.figure.Figure  – if multiple figures needed
        """
        from pymtex.texture._plot_utils import plot_pf_grid

        r_list, I_list = [], []
        for h in h_list:
            r, pf = self.calcPF(h, n_theta=n_theta, n_phi=n_phi)
            r_list.append(r)
            I_list.append(pf)

        figs = plot_pf_grid(
            h_list, r_list, I_list,
            cmap=cmap, plot_type=plot_type, levels=levels,
            suptitle=suptitle,
        )
        return figs[0] if len(figs) == 1 else figs

    # ------------------------------------------------------------------
    # Inverse pole figure
    # ------------------------------------------------------------------

    def plotIPF(self, r_specimen=None, ax=None, plot_type='scatter',
                cmap='jet'):
        """
        Inverse pole figure (IPF): distribution of a specimen direction
        *r_specimen* mapped into the crystal frame.

        Parameters
        ----------
        r_specimen : array-like (3,), optional – default [0,0,1] (ND)
        """
        if r_specimen is None:
            r_specimen = np.array([0.0, 0.0, 1.0])
        r_specimen = np.asarray(r_specimen, dtype=float)
        r_specimen /= np.linalg.norm(r_specimen)

        mats   = self.orientations.to_matrix()
        c_dirs = np.einsum('nij,j->ni', mats.transpose(0, 2, 1), r_specimen)

        theta = np.arccos(np.clip(c_dirs[:, 2], -1, 1))
        phi   = np.arctan2(c_dirs[:, 1], c_dirs[:, 0])
        rho   = np.sin(theta / 2) * np.sqrt(2)
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        if plot_type == 'scatter':
            sc = ax.scatter(x, y, c=self.f, cmap=cmap, s=5, alpha=0.6,
                            vmin=0, vmax=self.f.max())
        else:
            xi = np.linspace(-1, 1, 150)
            yi = np.linspace(-1, 1, 150)
            XX, YY = np.meshgrid(xi, yi)
            ZZ = griddata((x, y), self.f, (XX, YY), method='linear')
            ZZ[XX**2 + YY**2 > 1.01] = np.nan
            sc = ax.contourf(XX, YY, ZZ, levels=15, cmap=cmap)
            ax.contour(XX, YY, ZZ, levels=15, colors='k',
                       linewidths=0.25, alpha=0.4)

        plt.colorbar(sc, ax=ax, label='f(g) [m.r.d.]')
        _draw_circle(ax)
        _pf_axes(ax)
        ax.set_title('IPF (ND)', fontsize=10)
        return ax

    # ------------------------------------------------------------------
    # MTEX-style Euler-space ODF sections
    # ------------------------------------------------------------------

    def plotSections(self, phi2_vals=None, n_phi1=73, n_Phi=19,
                     cmap='jet', levels=15, plot_type='contourf',
                     show_contour_lines=True, fig=None, title=None):
        """
        Plot ODF as φ₂ sections through Euler angle space (MTEX style).

        Each panel shows f(φ₁, Φ) at a fixed φ₂ value with:
        - φ₁ (0–360°) on the x-axis
        - Φ  (0–90°)  on the y-axis (0 at top, matching MTEX convention)

        Parameters
        ----------
        phi2_vals : array-like of float (degrees), optional
            φ₂ section values.  Default: 9 equally-spaced values from 0°
            to the fundamental φ₂ limit for this crystal system.
        n_phi1, n_Phi : int
            Grid density for φ₁ and Φ axes.
        cmap : str
            Colormap (default ``'jet'``).
        levels : int
            Number of contour levels.
        plot_type : 'contourf' (default) or 'pcolormesh'
        show_contour_lines : bool
            Overlay thin contour lines (as in MTEX).
        fig : matplotlib Figure, optional
        title : str, optional
            Figure suptitle.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if phi2_vals is None:
            phi2_max = _phi2_fundamental(self.cs)
            phi2_vals = np.linspace(0, phi2_max, 9)

        phi2_vals = np.asarray(phi2_vals, dtype=float)
        n_sec = len(phi2_vals)

        phi1_deg = np.linspace(0, 360, n_phi1)
        Phi_deg  = np.linspace(0, 90,  n_Phi)
        PHI1, PHI = np.meshgrid(phi1_deg, Phi_deg, indexing='ij')  # (n_phi1, n_Phi)

        # Evaluate ODF on the full grid at once
        # Total query points: n_sec × n_phi1 × n_Phi
        n_pts = n_phi1 * n_Phi
        all_phi1 = np.tile(PHI1.ravel(), n_sec)
        all_Phi  = np.tile(PHI.ravel(),  n_sec)
        all_phi2 = np.repeat(phi2_vals, n_pts)

        ori = Orientation.byEuler(
            np.deg2rad(all_phi1),
            np.deg2rad(all_Phi),
            np.deg2rad(all_phi2),
            self.cs, convention='Bunge',
        )
        f_all = self.eval(ori)                             # (n_sec * n_pts,)
        f_sections = f_all.reshape(n_sec, n_phi1, n_Phi)  # (sec, phi1, Phi)

        # Layout constants
        SCOLS     = 5
        SCOLS_USE = min(n_sec, SCOLS)
        nrows     = (n_sec + SCOLS_USE - 1) // SCOLS_USE

        # Reserve a fixed-width strip on the right for the colorbar so it
        # never overlaps the section panels.
        sec_w, sec_h = 3.5, 2.4
        h_pad, v_pad = 0.55, 0.80
        top_mar      = 0.55
        cb_strip     = 0.55       # inches for colorbar strip on the right
        cb_pad_in    = 0.15       # gap between last column and colorbar

        grid_w = SCOLS_USE * (sec_w + h_pad) + h_pad
        fig_w  = grid_w + cb_pad_in + cb_strip
        fig_h  = nrows  * (sec_h + v_pad) + v_pad + top_mar

        if fig is None:
            fig, axes = plt.subplots(
                nrows, SCOLS_USE,
                figsize=(fig_w, fig_h),
                squeeze=False,
            )
        axes_flat = np.array(axes).ravel()

        vmin = 0
        vmax = max(f_all.max(), 1.0)

        last_cf = None
        for i, phi2 in enumerate(phi2_vals):
            ax = axes_flat[i]
            Z  = f_sections[i].T   # (n_Phi, n_phi1)

            if plot_type == 'contourf':
                last_cf = ax.contourf(
                    phi1_deg, Phi_deg, Z,
                    levels=np.linspace(vmin, vmax, levels + 1),
                    cmap=cmap, vmin=vmin, vmax=vmax,
                )
                if show_contour_lines:
                    ax.contour(
                        phi1_deg, Phi_deg, Z,
                        levels=np.linspace(vmin, vmax, levels + 1),
                        colors='k', linewidths=0.25, alpha=0.35,
                    )
            else:
                last_cf = ax.pcolormesh(
                    phi1_deg, Phi_deg, Z,
                    cmap=cmap, vmin=vmin, vmax=vmax, shading='auto',
                )

            ax.set_xlim(0, 360)
            ax.set_ylim(0, 90)
            ax.invert_yaxis()
            ax.set_xticks([0, 90, 180, 270, 360])
            ax.set_xticklabels(['0°', '90°', '180°', '270°', '360°'],
                               fontsize=6)
            ax.set_yticks([0, 30, 60, 90])
            ax.set_yticklabels(['0°', '30°', '60°', '90°'], fontsize=6)
            ax.set_xlabel('φ₁', fontsize=7)
            ax.set_ylabel('Φ',  fontsize=7)
            ax.set_title(f'φ₂ = {phi2:.0f}°', fontsize=9)
            ax.tick_params(length=2)

        # Blank unused cells without stealing space
        for i in range(n_sec, len(axes_flat)):
            axes_flat[i].axis('off')
            axes_flat[i].set_facecolor('none')

        # Lay out the grid panels first, leaving the right strip for the bar.
        # right = fraction of figure width occupied by the grid (no colorbar).
        right_frac = grid_w / fig_w - 0.01
        fig.subplots_adjust(
            left   = h_pad / fig_w,
            right  = right_frac,
            bottom = v_pad / fig_h,
            top    = 1.0 - top_mar / fig_h,
            wspace = h_pad / sec_w,
            hspace = v_pad / sec_h,
        )

        # Manually-placed colorbar axes: sits in the reserved right strip.
        if last_cf is not None:
            cb_left  = right_frac + cb_pad_in / fig_w
            cb_bot   = v_pad / fig_h
            cb_top   = 1.0 - top_mar / fig_h
            cax = fig.add_axes([cb_left, cb_bot,
                                 cb_strip * 0.35 / fig_w,
                                 cb_top - cb_bot])
            cb = fig.colorbar(last_cf, cax=cax)
            cb.set_label('f(g) [m.r.d.]', fontsize=8)
            cax.tick_params(labelsize=7)

        suptitle = title or (
            f'ODF φ₂ sections  –  {self.cs.name}  '
            f'(J = {self.texture_index():.2f})'
        )
        fig.suptitle(suptitle, fontsize=10)
        return fig

    # ------------------------------------------------------------------
    # Interactive 3D ODF  (plotly)
    # ------------------------------------------------------------------

    def plotODF3D(self,
                  n_phi1=37, n_Phi=19, n_phi2=None,
                  isomin=1.0, isomax=None,
                  surface_count=8, opacity=0.4,
                  colorscale='Jet',
                  show=True):
        """
        Interactive 3D ODF visualisation in Bunge Euler angle space.

        The ODF is rendered as semi-transparent isosurfaces stacked between
        *isomin* and *isomax* m.r.d.  The figure opens in the default browser
        (or VS Code plotly panel) where you can rotate, zoom, and hover over
        any isosurface to read f(g) values.

        Parameters
        ----------
        n_phi1, n_Phi, n_phi2 : int
            Grid density along each Euler angle axis.  Defaults give a
            37 × 19 × N grid (N set from crystal symmetry, typically 5–25).
        isomin : float
            Lowest isosurface level in m.r.d.  Default 1.0 (random texture).
            Orientations below this threshold are invisible — raise it to
            focus on strong components.
        isomax : float or None
            Highest isosurface level.  ``None`` uses the ODF maximum.
        surface_count : int
            Number of isosurfaces drawn between *isomin* and *isomax*.
        opacity : float
            Transparency of each surface (0 = fully transparent, 1 = opaque).
        colorscale : str
            Plotly colorscale name, e.g. ``'Jet'``, ``'Viridis'``, ``'Hot'``.
        show : bool
            If ``True`` (default) call ``fig.show()`` immediately to open the
            interactive window.  Set to ``False`` to obtain the figure object
            without displaying it (useful for saving to HTML).

        Returns
        -------
        plotly.graph_objects.Figure

        Examples
        --------
        >>> fig = odf.plotODF3D(isomin=2.0, surface_count=6)
        >>> fig.write_html('odf_3d.html')   # save for later / sharing
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError(
                "plotly is required for 3D ODF visualisation.\n"
                "Install with:  pip install plotly"
            )

        phi2_max = _phi2_fundamental(self.cs)
        if n_phi2 is None:
            n_phi2 = max(int(round(phi2_max / 5)) + 1, 5)

        phi1_deg = np.linspace(0, 360,      n_phi1)
        Phi_deg  = np.linspace(0, 90,       n_Phi)
        phi2_deg = np.linspace(0, phi2_max, n_phi2)

        PHI1, PHI, PHI2 = np.meshgrid(phi1_deg, Phi_deg, phi2_deg,
                                       indexing='ij')

        ori = Orientation.byEuler(
            np.deg2rad(PHI1.ravel()),
            np.deg2rad(PHI.ravel()),
            np.deg2rad(PHI2.ravel()),
            self.cs, convention='Bunge',
        )
        f_vals = self.eval(ori).reshape(PHI1.shape)

        vmax = float(isomax if isomax is not None else f_vals.max())
        vmin = float(isomin)

        # Aspect ratio: scale axes so the box looks proportional to the
        # physical Euler space, not stretched.
        asp_x = 360 / 90
        asp_z = phi2_max / 90

        fig = go.Figure(data=go.Volume(
            x=PHI1.ravel(),
            y=PHI.ravel(),
            z=PHI2.ravel(),
            value=f_vals.ravel(),
            isomin=vmin,
            isomax=vmax,
            opacity=opacity,
            surface_count=surface_count,
            colorscale=colorscale,
            colorbar=dict(
                title=dict(text='f(g) [m.r.d.]', side='right'),
                thickness=18,
            ),
            caps=dict(x_show=False, y_show=False, z_show=False),
            hovertemplate=(
                'φ₁ = %{x:.1f}°<br>'
                'Φ  = %{y:.1f}°<br>'
                'φ₂ = %{z:.1f}°<br>'
                'f(g) = %{value:.2f} m.r.d.'
                '<extra></extra>'
            ),
        ))

        fig.update_layout(
            title=dict(
                text=(f'ODF  –  {self.cs.name}  '
                      f'(J = {self.texture_index():.2f})  '
                      f'isomin = {vmin:.1f} m.r.d.'),
                x=0.5, font_size=14,
            ),
            scene=dict(
                xaxis=dict(title='φ₁ (°)', range=[0, 360]),
                yaxis=dict(title='Φ (°)',   range=[0, 90]),
                zaxis=dict(title='φ₂ (°)',  range=[0, phi2_max]),
                aspectmode='manual',
                aspectratio=dict(x=asp_x, y=1.0, z=asp_z),
                bgcolor='rgb(10,10,30)',
                xaxis_backgroundcolor='rgb(20,20,50)',
                yaxis_backgroundcolor='rgb(20,20,50)',
                zaxis_backgroundcolor='rgb(20,20,50)',
            ),
            paper_bgcolor='white',
            width=950, height=720,
            margin=dict(l=20, r=20, t=60, b=20),
            annotations=[dict(
                text=(f'Rotate: drag  ·  Zoom: scroll  ·  '
                      f'Pan: shift+drag  ·  '
                      f'Threshold: adjust isomin='),
                showarrow=False, x=0.5, y=-0.02,
                xref='paper', yref='paper',
                font=dict(size=10, color='grey'),
            )],
        )

        if show:
            fig.show()

        return fig

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self):
        return (f"ODF(grid_size={self.size}, "
                f"J={self.texture_index():.2f}, "
                f"cs={self.cs.name!r})")
