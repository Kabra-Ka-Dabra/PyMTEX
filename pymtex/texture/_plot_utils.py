"""
Shared pole-figure plotting helpers used by PoleFigure and ODF.

All multi-panel layouts use ImageGrid (mpl_toolkits.axes_grid1) which
guarantees equal-sized main axes regardless of how many panels are visible.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import ImageGrid


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
MAX_COLS      = 3     # pole figures per row
MAX_ROWS      = 4     # rows per figure before splitting into a new figure
PF_SIZE_IN    = 3.0   # fixed side length of every pole-figure cell (inches)
CB_FRAC       = 0.07  # colorbar width as fraction of PF_SIZE_IN
H_PAD         = 0.55  # horizontal spacing between cells (inches)
V_PAD         = 0.65  # vertical spacing between cells (inches)
TOP_MARGIN    = 0.45  # inches above the grid (for suptitle)
BOTTOM_MARGIN = 0.10  # inches below the grid


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def _proj(r, projection='equal_area'):
    """Project upper-hemisphere unit vectors r (N,3) → (x, y)."""
    theta = np.arccos(np.clip(r[:, 2], -1, 1))
    phi   = np.arctan2(r[:, 1], r[:, 0])
    rho   = (np.sin(theta / 2) * np.sqrt(2)
             if projection == 'equal_area'
             else np.tan(theta / 2))
    return rho * np.cos(phi), rho * np.sin(phi)


def _draw_circle(ax, lw=0.8):
    th = np.linspace(0, 2 * np.pi, 360)
    ax.plot(np.cos(th), np.sin(th), 'k-', lw=lw, zorder=5)


def _pf_frame(ax):
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')


# ---------------------------------------------------------------------------
# Single pole-figure renderer
# ---------------------------------------------------------------------------

def render_pf(ax, r, I, hkl_label, *,
              cmap='jet', plot_type='scatter',
              levels=15, projection='equal_area',
              upper_hemisphere=True):
    """
    Draw one pole figure (already normalised to m.r.d.) onto *ax*.

    Parameters
    ----------
    ax         : matplotlib Axes
    r          : (M, 3) unit vectors
    I          : (M,) intensities in m.r.d. (already normalised)
    hkl_label  : str, e.g. ``'(112)'``
    cmap       : colormap name
    plot_type  : ``'scatter'`` or ``'contour'``
    levels     : int  (contour levels)
    projection : ``'equal_area'`` (Schmidt) or ``'stereo'``
    upper_hemisphere : bool

    Returns
    -------
    mappable  (scatter collection or contourf object for colorbar attachment)
    """
    if upper_hemisphere:
        mask = r[:, 2] >= -0.05
        r, I = r[mask], I[mask]

    x, y  = _proj(r, projection)
    vmin, vmax = 0.0, max(float(I.max()), 1e-6)

    if plot_type == 'scatter':
        sc = ax.scatter(x, y, c=I, cmap=cmap, vmin=vmin, vmax=vmax, s=8,
                        zorder=3)
    else:
        xi = np.linspace(-1, 1, 200)
        XX, YY = np.meshgrid(xi, xi)
        ZZ = griddata((x, y), I, (XX, YY), method='linear')
        ZZ[XX ** 2 + YY ** 2 > 1.01] = np.nan
        lvls = np.linspace(vmin, vmax, levels + 1)
        sc = ax.contourf(XX, YY, ZZ, levels=lvls, cmap=cmap,
                         vmin=vmin, vmax=vmax, zorder=2)
        ax.contour(XX, YY, ZZ, levels=lvls,
                   colors='k', linewidths=0.25, alpha=0.35, zorder=4)

    _draw_circle(ax)
    _pf_frame(ax)
    ax.set_title(hkl_label, fontsize=9, pad=3)
    return sc


# ---------------------------------------------------------------------------
# Figure-size helper
# ---------------------------------------------------------------------------

def fig_size(nrows, ncols=MAX_COLS):
    """
    Compute figure dimensions so every PF cell is exactly PF_SIZE_IN × PF_SIZE_IN.

    Total width  = ncols  × (PF_SIZE_IN + H_PAD) + CB_FRAC×PF_SIZE_IN + H_PAD
    Total height = nrows  × (PF_SIZE_IN + V_PAD) + TOP_MARGIN + BOTTOM_MARGIN
    """
    cb_w  = CB_FRAC * PF_SIZE_IN
    w = ncols  * (PF_SIZE_IN + H_PAD) + cb_w + H_PAD
    h = nrows  * (PF_SIZE_IN + V_PAD) + TOP_MARGIN + BOTTOM_MARGIN
    return w, h


# ---------------------------------------------------------------------------
# Multi-panel grid layout  (equal-size guaranteed via ImageGrid)
# ---------------------------------------------------------------------------

def plot_pf_grid(h_list, r_list, I_list, *,
                 cmap='jet', plot_type='scatter',
                 levels=15, projection='equal_area',
                 upper_hemisphere=True,
                 suptitle=None,
                 max_cols=MAX_COLS, max_rows=MAX_ROWS):
    """
    Plot an arbitrary number of pole figures in a grid layout.

    Rules
    -----
    * Max *max_cols* (default 3) figures per row.
    * Max *max_rows* (default 4) rows per Figure.
    * Overflow spills into additional Figure objects.
    * **All pole-figure circles are the same physical size** in any given plot.

    Parameters
    ----------
    h_list, r_list, I_list : matched lists (Miller, (M,3) array, (M,) array)
        *I_list* values are normalised to m.r.d. inside this function.
    cmap, plot_type, levels, projection, upper_hemisphere : see render_pf()
    suptitle : str or None
    max_cols, max_rows : int

    Returns
    -------
    list of matplotlib.figure.Figure
        Single-element list when everything fits on one figure.
    """
    n = len(h_list)
    if n == 0:
        return []

    max_per_fig = max_cols * max_rows
    batches = [list(range(s, min(s + max_per_fig, n)))
               for s in range(0, n, max_per_fig)]
    n_figs = len(batches)
    figures = []

    for fig_idx, indices in enumerate(batches):
        n_in  = len(indices)
        nrows = (n_in + max_cols - 1) // max_cols

        # Figure size: each cell is PF_SIZE_IN × PF_SIZE_IN, spacing fixed.
        # Always allocate max_cols columns so unused cells hold space correctly.
        fw, fh = fig_size(nrows, max_cols)
        fig = plt.figure(figsize=(fw, fh))

        # ImageGrid guarantees equal-sized axes + consistent per-cell colorbars.
        cb_w  = f'{int(CB_FRAC * 100)}%'
        grid = ImageGrid(
            fig, 111,
            nrows_ncols=(nrows, max_cols),
            axes_pad=(H_PAD, V_PAD),
            cbar_mode='each',
            cbar_location='right',
            cbar_size=cb_w,
            cbar_pad='3%',
        )

        for cell_pos in range(nrows * max_cols):
            ax  = grid[cell_pos]
            cax = grid.cbar_axes[cell_pos]

            if cell_pos < n_in:
                idx  = indices[cell_pos]
                h_j  = h_list[idx]
                hkl  = '({}{}{})'.format(int(h_j.h), int(h_j.k), int(h_j.l))

                I    = I_list[idx]
                mean = I.mean()
                I_n  = I / mean if mean > 0 else I.copy()

                sc = render_pf(ax, r_list[idx], I_n, hkl,
                               cmap=cmap, plot_type=plot_type, levels=levels,
                               projection=projection,
                               upper_hemisphere=upper_hemisphere)
                cb = plt.colorbar(sc, cax=cax)
                cb.set_label('m.r.d.', fontsize=7)
                cax.tick_params(labelsize=6)
            else:
                # Unused cell: blank out both axes so they hold space silently.
                ax.axis('off')
                cax.axis('off')

        if suptitle:
            t = (suptitle if n_figs == 1
                 else f'{suptitle}  (part {fig_idx + 1}/{n_figs})')
            fig.suptitle(t, fontsize=11,
                         y=1 - TOP_MARGIN / (2 * fh))

        figures.append(fig)

    return figures
