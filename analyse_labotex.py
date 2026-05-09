"""
Full pole-figure → ODF → pole-figure pipeline for LaboTEX.epf
(Aluminium, cubic m-3m, 3 pole figures: {111}, {100}, {110})

Outputs
-------
labotex_measured_pf.png       – measured  (scatter, jet)
labotex_recalculated_pf.png   – calculated (contour, jet)
labotex_comparison.png         – side-by-side (measured scatter / calculated contour)
labotex_odf_sections.png       – MTEX-style ODF φ₂ sections
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from pymtex.texture import loadPoleFigureLaboTEX, calcODF
from pymtex.texture._plot_utils import (
    render_pf, fig_size,
    MAX_COLS, MAX_ROWS, PF_SIZE_IN, H_PAD, V_PAD, CB_FRAC, TOP_MARGIN,
)

FNAME = '/Users/sz2/Downloads/mtex-6.1.1/data/PoleFigure/LaboTEX.epf'
CMAP  = 'jet'

# ---------------------------------------------------------------------------
# 1. Load  (auto-detects LaboTEX format from .epf header)
# ---------------------------------------------------------------------------
pf = loadPoleFigureLaboTEX(FNAME)
cs = pf.cs
print(f"Loaded  : {pf}")
print(f"Symmetry: {cs}")

# ---------------------------------------------------------------------------
# 2. Measured pole figures  –  scatter (point-by-point), ≤3 per row
# ---------------------------------------------------------------------------
fig_m = pf.plot(
    cmap=CMAP, plot_type='scatter',
    suptitle='Measured pole figures – LaboTEX.epf (Al, m-3m)',
)
figs_m = [fig_m] if not isinstance(fig_m, list) else fig_m
for i, f in enumerate(figs_m):
    name = f'labotex_measured_pf{"" if len(figs_m)==1 else f"_part{i+1}"}.png'
    f.savefig(name, dpi=150, bbox_inches='tight')
    plt.close(f)
    print(f"Saved: {name}")

# ---------------------------------------------------------------------------
# 3. Compute ODF  (WIMV, 5° grid, 25 iterations)
# ---------------------------------------------------------------------------
print("\nComputing ODF …")
odf = calcODF(pf.normalize(), resolution_deg=5, n_iter=25, verbose=True)
print(f"\nTexture index  J = {odf.texture_index():.3f}")
print(f"Entropy        H = {odf.entropy():.3f}")

# ---------------------------------------------------------------------------
# 4. Re-calculated pole figures  –  contour, ≤3 per row
# ---------------------------------------------------------------------------
fig_c = odf.plotPFs(
    pf.h, cmap=CMAP, plot_type='contour',
    suptitle='Re-calculated pole figures – LaboTEX.epf (Al)',
)
figs_c = [fig_c] if not isinstance(fig_c, list) else fig_c
for i, f in enumerate(figs_c):
    name = f'labotex_recalculated_pf{"" if len(figs_c)==1 else f"_part{i+1}"}.png'
    f.savefig(name, dpi=150, bbox_inches='tight')
    plt.close(f)
    print(f"Saved: {name}")

# ---------------------------------------------------------------------------
# 5. Side-by-side comparison  (ImageGrid, equal-sized circles)
# ---------------------------------------------------------------------------
n_pf        = pf.numPF
max_per_fig = MAX_COLS * MAX_ROWS
batches     = [list(range(s, min(s + max_per_fig, n_pf)))
               for s in range(0, n_pf, max_per_fig)]
n_figs_cmp  = len(batches)
cb_w_str    = f'{int(CB_FRAC * 100)}%'

for bi, indices in enumerate(batches):
    n_in     = len(indices)
    n_pfrows = (n_in + MAX_COLS - 1) // MAX_COLS
    grid_rows = n_pfrows * 2   # measured + calculated per group

    fw, fh = fig_size(grid_rows, MAX_COLS)
    fig = plt.figure(figsize=(fw, fh))

    grid = ImageGrid(
        fig, 111,
        nrows_ncols=(grid_rows, MAX_COLS),
        axes_pad=(H_PAD, V_PAD * 0.5),
        cbar_mode='each',
        cbar_location='right',
        cbar_size=cb_w_str,
        cbar_pad='3%',
    )

    for pos, idx in enumerate(indices):
        col      = pos % MAX_COLS
        pf_row   = pos // MAX_COLS
        cell_m   = (pf_row * 2)     * MAX_COLS + col
        cell_c   = (pf_row * 2 + 1) * MAX_COLS + col

        ax_m,  cax_m  = grid[cell_m], grid.cbar_axes[cell_m]
        ax_c,  cax_c  = grid[cell_c], grid.cbar_axes[cell_c]

        h_j  = pf.h[idx]
        hkl  = '({}{}{})'.format(int(h_j.h), int(h_j.k), int(h_j.l))

        I_m   = pf.intensities[idx]
        mean  = I_m.mean()
        I_mn  = I_m / mean if mean > 0 else I_m
        sc_m  = render_pf(ax_m, pf.r[idx], I_mn, hkl, cmap=CMAP, plot_type='scatter')
        cb_m  = plt.colorbar(sc_m, cax=cax_m)
        cb_m.set_label('m.r.d.', fontsize=7)
        cax_m.tick_params(labelsize=6)

        r_c, I_c = odf.calcPF(h_j)
        sc_c  = render_pf(ax_c, r_c, I_c, hkl, cmap=CMAP, plot_type='contour')
        cb_c  = plt.colorbar(sc_c, cax=cax_c)
        cb_c.set_label('m.r.d.', fontsize=7)
        cax_c.tick_params(labelsize=6)

    # Blank unused cells
    for pos in range(n_in, n_pfrows * MAX_COLS):
        col = pos % MAX_COLS; pf_row = pos // MAX_COLS
        for ro in (0, 1):
            cell = (pf_row * 2 + ro) * MAX_COLS + col
            grid[cell].axis('off'); grid.cbar_axes[cell].axis('off')

    # Row-type labels
    for pf_row in range(n_pfrows):
        grid[pf_row * 2 * MAX_COLS].set_ylabel('Measured',   fontsize=8, labelpad=4)
        grid[(pf_row * 2 + 1) * MAX_COLS].set_ylabel('Calculated', fontsize=8, labelpad=4)

    t = (f'LaboTEX Al  –  Measured (scatter) ↑ / Calculated (contour) ↓\n'
         f'J = {odf.texture_index():.2f}'
         + (f'   (part {bi+1}/{n_figs_cmp})' if n_figs_cmp > 1 else ''))
    fig.suptitle(t, fontsize=10, y=1 - TOP_MARGIN / (2 * fh))

    name = f'labotex_comparison{"" if n_figs_cmp==1 else f"_part{bi+1}"}.png'
    fig.savefig(name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {name}")

# ---------------------------------------------------------------------------
# 6. MTEX-style ODF φ₂ sections
# ---------------------------------------------------------------------------
fig_odf = odf.plotSections(
    cmap=CMAP, levels=15,
    plot_type='contourf', show_contour_lines=True,
)
fig_odf.savefig('labotex_odf_sections.png', dpi=150, bbox_inches='tight')
plt.close(fig_odf)
print("Saved: labotex_odf_sections.png")

plt.close('all')
print("\nDone.")
