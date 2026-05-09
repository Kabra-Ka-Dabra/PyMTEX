# PyMTEX Usage Guide

Python port of the [MTEX](https://mtex-toolbox.github.io/) crystallographic texture analysis toolbox.

---

## Installation

```bash
# Editable install with dev/test dependencies
pip install -e ".[dev]"

# With interactive 3D ODF support (adds plotly)
pip install -e ".[dev,interactive]"

# Run all tests (189 tests)
python3 -m pytest tests/ -v
```

---

## Quick Start — Pole Figure → ODF Pipeline

```python
from pymtex.texture import loadPoleFigureLaboTEX, calcODF

# 1. Load pole figures (auto-detects format from file)
pf = loadPoleFigureLaboTEX('LaboTEX.epf')

# 2. Compute ODF via WIMV
odf = calcODF(pf.normalize(), resolution_deg=5, n_iter=25, verbose=True)
print(f"Texture index J = {odf.texture_index():.2f}")

# 3. Plot measured  (scatter, jet, ≤3 per row)
pf.plot()

# 4. Plot re-calculated pole figures (contour)
odf.plotPFs(pf.h)

# 5. ODF φ₂ sections  (MTEX style)
odf.plotSections()

# 6. Interactive 3D ODF — opens browser window
odf.plotODF3D(isomin=2.0)

# 7. Save / load the ODF
from pymtex.texture import saveODF, loadODF
saveODF(odf, 'my_odf.txt')
odf2 = loadODF('my_odf.txt', pf.cs)
```

---

## Module Reference

### 1. Geometry

#### `CrystalSymmetry`

All 32 crystallographic point groups. Generators follow MTEX conventions
(a = x, b = y, c = z; trigonal uses the 321 setting).

```python
from pymtex import CrystalSymmetry

cs = CrystalSymmetry('m-3m')       # cubic  (Al, Fe, Ni, Cu)
cs = CrystalSymmetry('6/mmm')      # hexagonal  (Mg, Ti, Zn)
cs = CrystalSymmetry('-3m')        # trigonal  (quartz)
cs = CrystalSymmetry('mmm')        # orthorhombic  (olivine)
cs = CrystalSymmetry('2/m')        # monoclinic

# Aliases also accepted
cs = CrystalSymmetry('cubic')      # → 'm-3m'
cs = CrystalSymmetry('Oh')         # Schoenflies → 'm-3m'

cs.nfold        # 48  (group order)
cs.system       # 'cubic'
cs.Laue.name    # 'm-3m'
cs.isLaue       # True / False
cs.rot          # Rotation array of all symmetry operators
```

**All supported names:** `'1'`, `'-1'`, `'2'`, `'m'`, `'2/m'`, `'222'`, `'mm2'`,
`'mmm'`, `'4'`, `'-4'`, `'4/m'`, `'422'`, `'4mm'`, `'-42m'`, `'4/mmm'`,
`'3'`, `'-3'`, `'32'`, `'3m'`, `'-3m'`, `'6'`, `'-6'`, `'6/m'`, `'622'`,
`'6mm'`, `'-6m2'`, `'6/mmm'`, `'23'`, `'m-3'`, `'432'`, `'-43m'`, `'m-3m'`.

---

#### `Quaternion`

Unit quaternion `q = a + bi + cj + dk` (MTEX convention: `a` is scalar).
`q` and `−q` represent the same rotation and are both kept as distinct objects.

```python
from pymtex import Quaternion
import numpy as np

q = Quaternion(1, 0, 0, 0)
q = Quaternion.identity()
q = Quaternion.rand(n=100)                   # uniform random on SO(3)
q = Quaternion.from_euler(phi1, Phi, phi2, convention='Bunge')  # radians
q = Quaternion.from_axis_angle([0, 0, 1], np.pi / 3)
q = Quaternion.from_matrix(M)               # (3,3) or (N,3,3)

q1 * q2          # Hamilton product
q.inv()
q.norm()
q.normalize()
q.angle()        # rotation angle (radians)
q.axis()         # unit rotation axis  (N, 3)

q.to_matrix()
q.to_euler('Bunge')    # (phi1, Phi, phi2) in radians
q.to_rodrigues()       # Rodrigues–Frank vector  (N, 3)
```

---

#### `Rotation`

Extends `Quaternion` with an `improper` flag (reflections, roto-inversions).
Multiplication uses XOR on the flags, matching MTEX `@rotation`.

```python
from pymtex import Rotation

r = Rotation.identity()
r = Rotation.inversion()
r = Rotation.from_axis_angle([1, 0, 0], np.pi / 4)
r = Rotation.from_rodrigues([0.2, 0.1, 0.0])

r.improper                     # bool array
r.rotate(v)                    # apply to 3D vector(s)
r.misorientation_angle(other)
```

---

#### `Orientation`

A `Rotation` annotated with crystal (and optionally specimen) symmetry.

```python
from pymtex import CrystalSymmetry, Orientation

cs = CrystalSymmetry('m-3m')

o = Orientation.byEuler(45, 35, 0, cs, degrees=True)   # Bunge
o = Orientation.byAxisAngle([0, 0, 1], np.pi / 4, cs)
o = Orientation.byMatrix(M, cs)
o = Orientation.rand(100, cs)

# Standard texture components (cubic)
cube   = Orientation.cube(cs)      # {001}<100>
goss   = Orientation.goss(cs)      # {110}<001>
brass  = Orientation.brass(cs)     # {110}<112>
copper = Orientation.copper(cs)    # {112}<111>

phi1, Phi, phi2 = o.toEuler(degrees=True)

o_fr  = o.project2FundamentalRegion()
equiv = o.symmetricEquivalent()
mis   = o1.calcMisorientation(o2)   # radians
```

---

#### `Miller`

Plane normals `(hkl)` and directions `[uvw]`. Accepts 3-index or
4-index Miller-Bravais notation (`hkil` / `UVTW`).

```python
from pymtex import CrystalSymmetry, Miller

cs = CrystalSymmetry('m-3m')

n = Miller(hkl=[1, 1, 0], cs=cs)
d = Miller(uvw=[1, 1, 1], cs=cs)

# Hexagonal 4-index (drops i automatically)
n_hex = Miller(hkl=[1, 0, -1, 0], cs=CrystalSymmetry('6/mmm'))

n.angle(Miller(hkl=[0, 0, 1], cs=cs), degrees=True)   # 90.0
equiv = n.symmetricEquivalent()
n.multiplicity                                          # 12 for {110}
```

---

### 2. EBSD

```python
from pymtex import CrystalSymmetry, Orientation, EBSD
import numpy as np

cs  = CrystalSymmetry('m-3m')
x   = np.linspace(0, 100, 50)
y   = np.linspace(0, 100, 50)
X, Y = np.meshgrid(x, y)
ori = Orientation.rand(X.size, cs)

ebsd = EBSD(X.ravel(), Y.ravel(), ori)

ebsd.numPixels
ebsd.isIndexed                          # bool mask
ebsd.filter(mask)
ebsd.indexed()
ebsd.meanOrientation()
kam = ebsd.calcKAM(max_angle=np.deg2rad(5))
grain_id, n_grains = ebsd.calcGrains(threshold=np.deg2rad(10))
```

---

### 3. Pole Figure I/O

#### Supported formats

| Format | Loader | Extension(s) | Auto-detected? |
|---|---|---|---|
| **LaboTEX** | `loadPoleFigureLaboTEX` | `.epf` | ✓ (header `"number of Pole figures"`) |
| **POPLA / EPF** | `loadPoleFigure(..., format='popla')` | `.epf`, `.pf` | ✓ |
| **BEARTEX** | `loadPoleFigure(..., format='beartex')` | `.bea` | ✓ |
| **XPa** (multi-pole BEARTEX) | `loadPoleFigureXPa` | `.XPa/.XPb/.XPc` | ✓ |
| **Generic column** | `loadPoleFigure` | `.txt/.csv/.dat` | ✓ (fallback) |
| **Bare matrix** | `loadPoleFigure(..., format='matrix')` | any | ✗ (explicit) |

> **LaboTEX vs POPLA disambiguation:** both use `.epf`. PyMTEX looks for
> `"number of Pole figures"` in the first 12 lines to tell them apart.

#### LaboTEX (multi-pole, reads crystal symmetry from file)

```python
from pymtex.texture import loadPoleFigureLaboTEX

pf = loadPoleFigureLaboTEX('LaboTEX.epf')
# Auto-reads n poles, hkl, alpha/beta grid, crystal symmetry
```

#### XPa / BEARTEX multi-pole

```python
from pymtex.texture import loadPoleFigureXPa

pf = loadPoleFigureXPa('go_2_2.XPa')   # 7 poles, trigonal -3m
```

#### POPLA single-pole EPF

```python
from pymtex.texture import loadPoleFigure
from pymtex import CrystalSymmetry, Miller

cs = CrystalSymmetry('m-3m')
pf = loadPoleFigure(
    ['100.epf', '110.epf', '111.epf'],
    [Miller(hkl=[1,0,0], cs=cs),
     Miller(hkl=[1,1,0], cs=cs),
     Miller(hkl=[1,1,1], cs=cs)],
    cs, format='popla',
)
```

#### Generic column  `(alpha_deg  beta_deg  intensity)`

```python
pf = loadPoleFigure('pf.txt', h, cs)                           # auto
pf = loadPoleFigure('pf.txt', h, cs, degrees=False)            # radians
pf = loadPoleFigure('pf.txt', h, cs,
                    alpha_col=1, beta_col=2, intensity_col=0)  # reorder
```

#### Bare matrix

```python
pf = loadPoleFigure('grid.dat', h, cs,
                    format='matrix',
                    alpha_start=0, alpha_stop=85, alpha_step=5,
                    beta_start=0,  beta_stop=355, beta_step=5)
```

---

### 4. ODF Calculation

```python
from pymtex.texture import calcODF

odf = calcODF(
    pf,                     # PoleFigure object (normalise first)
    resolution_deg=5,       # SO(3) grid step
    n_iter=25,              # WIMV iterations
    verbose=True,           # print R-factor each iteration
)

odf.texture_index()   # J = ∫ f² dg  (1.0 = random)
odf.entropy()         # H = −∫ f·log(f) dg
```

**Grid resolution guide:**

| `resolution_deg` | SO(3) cells | Approx. time (25 iter, 3 poles) |
|---|---|---|
| 10° | ~13 k | ~3 s |
| 5° | ~98 k | ~30 s |
| 2.5° | ~780 k | ~5 min |

---

### 5. ODF File I/O

Save a computed ODF and reload it later — no need to re-run WIMV.

```python
from pymtex.texture import saveODF, loadODF
from pymtex import CrystalSymmetry

# Save  (4-column ASCII: phi1 Phi phi2 f — same layout as MTEX export)
saveODF(odf, 'my_odf.txt')

# Reload
cs  = CrystalSymmetry('m-3m')
odf = loadODF('my_odf.txt', cs)          # auto-detects format
odf = loadODF('texture.wts', cs)         # POPLA weight file (.wts)

# All plotting methods work on loaded ODFs
odf.plotSections()
odf.plotODF3D()
```

**Saved file format:**
```
ODF  crystal symmetry: m-3m  J = 10.2456
% phi1(deg)  Phi(deg)  phi2(deg)  f(g)[m.r.d.]
  0.0000    0.0000    0.0000    1.052300
  5.0000    0.0000    5.0000    2.387600
  ...
```

**Supported ODF file formats:**

| Format | Extension | Notes |
|---|---|---|
| 4-column ASCII | `.txt`, `.csv`, `.dat` | Auto-detected; MTEX `export(odf)` compatible |
| POPLA weight file | `.wts` | Auto-detected from extension |

Not yet supported: BEARTEX `.cor`, MTEX `.mat` (MATLAB binary).
Convert with MTEX first: `export(odf, 'my_odf.txt')`.

---

### 6. Plotting

All plots use **jet** colormap by default. All pole figures in a given figure
are the same physical size (guaranteed via `ImageGrid`).

#### Measured pole figures

```python
# Default: scatter (point-by-point), ≤3 per row, ≤4 rows per figure
fig = pf.plot()
fig = pf.plot(plot_type='scatter')   # explicit
fig = pf.plot(plot_type='contour', levels=15)

# >12 pole figures → returns a list of figures
figs = pf.plot()
if isinstance(figs, list):
    for i, f in enumerate(figs):
        f.savefig(f'pf_part{i}.png', dpi=150)
```

#### Re-calculated pole figures (from ODF)

```python
ax  = odf.plotPF(Miller(hkl=[1,0,0], cs=cs))     # single pole
fig = odf.plotPFs(pf.h)                            # all poles, same layout rules
fig = odf.plotPFs(pf.h, plot_type='contour')       # default for calculated
```

#### ODF φ₂ sections (MTEX style)

Each panel shows f(φ₁, Φ) at a fixed φ₂.
φ₁ (0–360°) on the x-axis; Φ (0–90°) on the y-axis with 0° at top (MTEX convention).

```python
fig = odf.plotSections()
fig = odf.plotSections(
    phi2_vals=[0, 15, 30, 45, 60],   # custom sections
    cmap='jet', levels=15,
    plot_type='contourf',            # or 'pcolormesh'
    show_contour_lines=True,
)
fig.savefig('sections.png', dpi=150, bbox_inches='tight')
```

**Default φ₂ range by crystal system:**
cubic 0–90°, hexagonal 0–60°, trigonal 0–120°, tetragonal 0–90°,
orthorhombic 0–90°.

#### Inverse pole figure

```python
ax = odf.plotIPF()                          # ND direction (default)
ax = odf.plotIPF(r_specimen=[1, 0, 0])      # RD direction
ax = odf.plotIPF(plot_type='contour')
```

#### Interactive 3D ODF  (requires `pip install plotly`)

Opens in your default browser or the VS Code plotly panel.

```python
odf.plotODF3D()                              # opens immediately

odf.plotODF3D(
    isomin=2.0,          # hide f(g) < 2 m.r.d.
    isomax=None,         # auto  (= ODF maximum)
    surface_count=8,     # number of isosurface shells
    opacity=0.4,
    colorscale='Jet',    # or 'Hot', 'Viridis', 'RdYlBu_r'
)

# Save as self-contained HTML (no Python needed to view)
fig = odf.plotODF3D(show=False)
fig.write_html('odf_3d.html')
```

**Interactive controls:**
`Rotate` — click+drag · `Zoom` — scroll ·
`Pan` — shift+drag · `Hover` — shows φ₁, Φ, φ₂ and f(g) on each surface.

**Axes:** φ₁ (0–360°), Φ (0–90°), φ₂ (0–φ₂_max°) in Bunge convention.
Aspect ratio matches the Euler space proportions automatically.

---

### 7. End-to-end example scripts

| Script | Dataset | Crystal | J |
|---|---|---|---|
| `analyse_labotex.py` | `LaboTEX.epf` — rolled aluminium | m-3m | ~10 |
| `analyse_go2.py` | `go_2_2.XPa` — quartz | -3m | ~1.4 |

Each produces: measured PF (scatter), re-calculated PF (contour),
side-by-side comparison, ODF φ₂ sections, interactive 3D HTML.

---

## Angle Conventions

| Convention | Details |
|---|---|
| Quaternion | `q = a + bi + cj + dk`, scalar part `a` first |
| Euler angles | **Bunge ZXZ** (φ₁, Φ, φ₂) as default; ZYZ/Matthies also supported |
| Rotation sense | Active (vectors are rotated, not the frame) |
| Pole figures | α = tilt from ND (0–90°), β = rotation (0–360°) |
| Improper ops | Boolean flag; multiplication uses XOR (MTEX convention) |

---

## Project Structure

```
pymtex/
  geometry/
    quaternion.py      Quaternion  – base class
    rotation.py        Rotation(Quaternion) – improper flag
    symmetry.py        CrystalSymmetry  – all 32 point groups
    orientation.py     Orientation  – rotation + crystal symmetry
    miller.py          Miller  – hkl / uvw indices
  ebsd/
    ebsd.py            EBSD  – spatial map, KAM, grain segmentation
  texture/
    polefigure.py      PoleFigure  – measured data + plot()
    odf.py             ODF  – discrete SO(3) grid + all plotting
    calcodf.py         calcODF  – WIMV pole-figure inversion
    io.py              All file-format loaders + ODF save/load
    _plot_utils.py     ImageGrid layout (equal-size pole figures)
tests/                 189 unit tests (all passing)
analyse_labotex.py     Full pipeline example — aluminium
analyse_go2.py         Full pipeline example — quartz
USAGE.md               This file
CLAUDE.md              Developer / architecture notes
pyproject.toml         Dependencies: numpy, scipy, matplotlib; optional: plotly
```

---

## Roadmap

| Feature | Priority |
|---|---|
| Harmonic ODF inversion (Wigner D-functions) | High |
| `ODFComponent` / kernel ODF fitting | Medium |
| `SO3Grid` / fundamental-zone grids | Medium |
| Pole-figure symmetry-sector restriction | Low |
| `@tensor` / elasticity | Low |
| `@grain2d` / grain boundary network | Low |
| BEARTEX `.cor` ODF file reader | Low |
| File export (POPLA/BEARTEX write) | Low |
