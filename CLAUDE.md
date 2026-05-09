# PyMTEX

Python port of the [MTEX](https://mtex-toolbox.github.io/) MATLAB crystallographic texture analysis toolbox.

## Project goal

Faithfully reproduce MTEX's classes and conventions in pure Python (NumPy/SciPy/matplotlib), module by module, with 189 passing tests and validated against real datasets.

## Commands

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# With interactive 3D ODF support
pip install -e ".[dev,interactive]"   # adds plotly

# Run all tests
python3 -m pytest tests/ -v

# End-to-end analysis examples
python3 analyse_labotex.py    # aluminium rolling texture
python3 analyse_go2.py        # quartz trigonal texture
```

## Architecture

```
pymtex/
  geometry/
    quaternion.py      # Quaternion  – MTEX convention (a,b,c,d scalar-first)
    rotation.py        # Rotation(Quaternion) – improper flag, XOR multiply
    symmetry.py        # CrystalSymmetry – all 32 point groups, group closure
    orientation.py     # Orientation  – rotation + crystal symmetry
    miller.py          # Miller  – hkl / uvw indices, 4-index support
  ebsd/
    ebsd.py            # EBSD  – spatial orientation map, KAM, grain segmentation
  texture/
    polefigure.py      # PoleFigure  – measured data, plot()
    odf.py             # ODF  – discrete SO(3) grid, all plotting, 3D interactive
    calcodf.py         # calcODF  – WIMV iterative inversion
    io.py              # All file format loaders (see below)
    _plot_utils.py     # ImageGrid layout – equal-size pole figures
tests/
  geometry/            # 189 tests total, all passing
  ebsd/
  texture/
USAGE.md               # Full user guide
CLAUDE.md              # This file – developer notes
```

## Key conventions (match MTEX exactly)

- Quaternion: `q = a + bi + cj + dk`, `a` is scalar part.
- `q` and `-q` are same rotation; both kept as distinct objects.
- Euler angles default to **Bunge ZXZ** (φ₁, Φ, φ₂) for orientations.
- Improper rotations carry `improper` bool flag; multiplication uses XOR.
- All classes support **batched** (array-valued) operations via NumPy.
- Crystal axes: a=x, b=y, c=z (MTEX default). Trigonal uses 321 setting.

## Supported file formats

| Format | Loader | Extension | Notes |
|---|---|---|---|
| LaboTEX | `loadPoleFigureLaboTEX` | `.epf` | Multi-pole, reads CS from file |
| POPLA | `loadPoleFigure(..., format='popla')` | `.epf`, `.pf` | Single pole per file |
| BEARTEX | `loadPoleFigure(..., format='beartex')` | `.bea` | Single pole |
| XPa | `loadPoleFigureXPa` | `.XPa/.XPb/.XPc` | Multi-pole, reads CS |
| Generic column | `loadPoleFigure` | `.txt/.csv/.dat` | (alpha, beta, I) columns |
| Bare matrix | `loadPoleFigure(..., format='matrix')` | any | Grid params as kwargs |

Auto-detection distinguishes LaboTEX from POPLA by checking for "number of Pole figures" in the header.

## Plotting defaults

- Colormap: **jet**
- Measured PF: **scatter** (point-by-point)
- Calculated PF: **contour** (filled + thin contour lines)
- Layout: max **3 per row**, max **4 rows per figure**; overflow → new figure
- Equal-size circles: guaranteed via **ImageGrid** (`mpl_toolkits.axes_grid1`)
- ODF colorbar: **manual axes placement** (`fig.add_axes`) to avoid overlap
- 3D ODF: **plotly Volume** isosurfaces, opens in browser

## Current status (2026-05-08) — 189 tests passing

All geometry, EBSD, texture/IO, and plotting modules complete.
See `USAGE.md` for full API documentation.

## Roadmap

1. Harmonic ODF inversion (Wigner D-functions) — faster + more accurate
2. ODFComponent / kernel ODF fitting
3. SO3Grid / SphericalGrid (fundamental zone grids)
4. Pole-figure symmetry-sector restriction in plots
5. @tensor / elasticity tensors
6. @grain2d / grain boundary network

## Reference

- MTEX website: https://mtex-toolbox.github.io/
- MTEX GitHub: `mtex-toolbox/mtex` — `geometry/@quaternion/`, `geometry/@symmetry/`
- Key MTEX source fetched: `symmetry.m` (calcQuat), `orientation.m`, `Miller.m`
