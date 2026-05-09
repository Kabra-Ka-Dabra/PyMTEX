# PyMTEX Usage Guide

Python port of the [MTEX](https://mtex-toolbox.github.io/) crystallographic texture analysis toolbox.

**Repository:** https://github.com/Kabra-Ka-Dabra/PyMTEX

---

## Installation

```bash
# Clone and install in editable mode
git clone https://github.com/Kabra-Ka-Dabra/PyMTEX.git
cd PyMTEX
pip install -e ".[dev]"

# With interactive 3D ODF support
pip install -e ".[dev,interactive]"    # adds plotly

# Run all tests
python3 -m pytest tests/ -v            # 189 tests
```

---

## Quick Start — Pole Figure → ODF Pipeline

```python
from pymtex.texture import loadPoleFigureLaboTEX, calcODF, saveODF, loadODF

# 1. Load pole figures  (auto-detects format from file header)
pf = loadPoleFigureLaboTEX('LaboTEX.epf')

# 2. Compute ODF via WIMV
odf = calcODF(pf.normalize(), resolution_deg=5, n_iter=25, verbose=True)
print(f"Texture index J = {odf.texture_index():.2f}")

# 3. Measured pole figures  (scatter, jet, ≤3 per row)
pf.plot()

# 4. Re-calculated pole figures  (contour, jet)
odf.plotPFs(pf.h)

# 5. ODF φ₂ sections  (MTEX style)
odf.plotSections()

# 6. Interactive 3D ODF in browser
odf.plotODF3D(isomin=2.0)

# 7. Save / reload the ODF  (no need to re-run WIMV)
saveODF(odf, 'my_odf.txt')
odf2 = loadODF('my_odf.txt', pf.cs)
```

---

## Module Reference

### 1. `CrystalSymmetry`

All 32 crystallographic point groups. Generators follow MTEX conventions
(a = x, b = y, c = z; trigonal uses the 321 setting).

```python
from pymtex import CrystalSymmetry

cs = CrystalSymmetry('m-3m')       # cubic  (Al, Fe, Ni, Cu)
cs = CrystalSymmetry('6/mmm')      # hexagonal  (Mg, Ti, Zn)
cs = CrystalSymmetry('-3m')        # trigonal  (quartz)
cs = CrystalSymmetry('mmm')        # orthorhombic  (olivine)
cs = CrystalSymmetry('2/m')        # monoclinic

# Aliases
cs = CrystalSymmetry('cubic')      # → 'm-3m'
cs = CrystalSymmetry('Oh')         # Schoenflies → 'm-3m'
cs = CrystalSymmetry('D6h')        # Schoenflies → '6/mmm'

# Properties
cs.nfold             # 48  — group order
cs.numProper         # 24  — proper rotations only
cs.system            # 'cubic'
cs.Laue.name         # 'm-3m'
cs.isLaue            # True if centrosymmetric
cs.isProper          # True if all elements are proper rotations
cs.rot               # Rotation array — all symmetry operators
cs.properRotations   # Rotation array — proper operators only  (det = +1)
```

**All supported names:** `'1'`, `'-1'`, `'2'`, `'m'`, `'2/m'`, `'222'`, `'mm2'`,
`'mmm'`, `'4'`, `'-4'`, `'4/m'`, `'422'`, `'4mm'`, `'-42m'`, `'4/mmm'`,
`'3'`, `'-3'`, `'32'`, `'3m'`, `'-3m'`, `'6'`, `'-6'`, `'6/m'`, `'622'`,
`'6mm'`, `'-6m2'`, `'6/mmm'`, `'23'`, `'m-3'`, `'432'`, `'-43m'`, `'m-3m'`.

---

### 2. `Quaternion`

Unit quaternion `q = a + bi + cj + dk` (MTEX convention: `a` is scalar first).
`q` and `−q` represent the same rotation; both are kept as distinct objects.

```python
from pymtex import Quaternion
import numpy as np

# Construction
q = Quaternion(1, 0, 0, 0)
q = Quaternion.identity()
q = Quaternion.nan(shape=(5,))               # NaN-filled
q = Quaternion.rand(n=100)                   # uniform random on SO(3)
q = Quaternion.rand(n=50, max_angle=np.pi/4) # restrict rotation angle
q = Quaternion.from_euler(phi1, Phi, phi2, convention='Bunge')  # radians
q = Quaternion.from_axis_angle([0, 0, 1], np.pi / 3)
q = Quaternion.from_matrix(M)               # (3,3) or (N,3,3)
q = Quaternion.from_rodrigues(r)            # Rodrigues vector (3,) or (N,3)

# Arithmetic
q1 * q2          # Hamilton product
q.inv()          # inverse  (= conjugate for unit quaternions)
q.norm()
q.normalize()
q.conjugate()    # q* = a − bi − cj − dk
-q               # negate  (same rotation, opposite hemisphere)
q == q2          # element-wise comparison

# Components
q.a, q.b, q.c, q.d    # scalar arrays
q.components()         # (…, 4) array [a, b, c, d]
q.shape, q.size, q.ndim

# Geometric properties
q.angle()        # rotation angle in radians ∈ [0, π]
q.axis()         # unit rotation axis  (…, 3)

# Inner products
q.dot(q2)              # component-wise dot product
q.dot_outer(q2)        # (N, M) outer dot product  — all pairs

# Conversions
q.to_matrix()          # (3,3) or (N,3,3)
q.to_euler('Bunge')    # (phi1, Phi, phi2) in radians
q.to_euler('ZYZ')      # Matthies / ZYZ convention
q.to_rodrigues()       # Rodrigues–Frank vector  (…, 3)

# Indexing (batched quaternions behave like arrays)
q[0]        # first element
q[2:5]      # slice
len(q)      # number of elements (raises if scalar)
```

---

### 3. `Rotation`

Extends `Quaternion` with an `improper` flag for reflections and roto-inversions.
Multiplication XORs the flags (matching MTEX `@rotation`).

```python
from pymtex import Rotation

r = Rotation.identity()
r = Rotation.inversion()                    # improper identity
r = Rotation.from_axis_angle([1, 0, 0], np.pi / 4)
r = Rotation.from_euler(phi1, Phi, phi2, convention='Bunge')
r = Rotation.from_matrix(M)
r = Rotation.from_rodrigues(rod)

# Additional properties
r.improper                      # bool array
r.is_proper()                   # True where not improper
r.is_improper()                 # True where improper

# Operations
r.rotate(v)                     # apply to 3D vector(s)  → (3,) or (N,3)
r.inv()                         # inverse (conjugate, same improper flag)
r.misorientation_angle(other)   # rotation angle of self⁻¹ * other
r.angle_between(other)          # alias for misorientation_angle
```

---

### 4. `Orientation`

A `Rotation` annotated with crystal symmetry `cs` (and optional specimen symmetry `ss`).

```python
from pymtex import CrystalSymmetry, Orientation

cs = CrystalSymmetry('m-3m')

# Construction
o = Orientation.byEuler(45, 35, 0, cs, degrees=True)    # Bunge convention
o = Orientation.byEuler(phi1, Phi, phi2, cs)             # radians
o = Orientation.byAxisAngle([0, 0, 1], np.pi / 4, cs)
o = Orientation.byMatrix(M, cs)
o = Orientation.rand(100, cs)

# Standard texture components (cubic)
cube   = Orientation.cube(cs)       # {001}<100>  φ1=Φ=φ2=0°
goss   = Orientation.goss(cs)       # {110}<001>  φ1=0°, Φ=45°, φ2=0°
brass  = Orientation.brass(cs)      # {110}<112>  φ1=35.26°, Φ=45°, φ2=0°
copper = Orientation.copper(cs)     # {112}<111>  φ1=90°, Φ=35.26°, φ2=45°
s_comp = Orientation.rotationS(cs)  # {123}<634>  φ1=59°, Φ=37°, φ2=63°

# Euler angles
phi1, Phi, phi2 = o.toEuler(degrees=True)    # Bunge
phi1, Phi, phi2 = o.toEuler(convention='ZYZ')

# Symmetry operations
o_fr  = o.project2FundamentalRegion()        # minimum-angle representative
equiv = o.symmetricEquivalent()              # all CS-equivalent orientations
mis   = o1.calcMisorientation(o2)            # in radians (accounts for CS)
mis   = o1.angle_between(o2)                 # alias
```

---

### 5. `Miller`

Plane normals `(hkl)` and directions `[uvw]`. Accepts 3-index or
4-index Miller-Bravais notation (`hkil` / `UVTW`).

```python
from pymtex import CrystalSymmetry, Miller

cs = CrystalSymmetry('m-3m')

# 3-index construction
n = Miller(hkl=[1, 1, 0], cs=cs)     # plane normal
d = Miller(uvw=[1, 1, 1], cs=cs)     # direction

# 4-index (Miller-Bravais, hexagonal) — i dropped automatically
n_hex = Miller(hkl=[1, 0, -1, 0], cs=CrystalSymmetry('6/mmm'))
d_hex = Miller(uvw=[2, 1, -3, 0], cs=CrystalSymmetry('6/mmm'))

# Properties
n.h, n.k, n.l        # index components
n.type               # 'hkl' or 'uvw'
n.shape, n.size

# Geometry
n.toVector3d()       # unit Cartesian vector(s)  (…, 3)
n.normalize()        # unit-length copy
n.dot(m)             # dot product of unit vectors
n.angle(m, degrees=True)          # angle between two Miller objects
equiv = n.symmetricEquivalent()   # all equivalent planes/directions
n.multiplicity                    # number of equivalent directions
```

**Multiplicities for cubic {hkl}:**
`{100}` = 6, `{110}` = 12, `{111}` = 8, `{210}` = 24, `{211}` = 24.

---

### 6. `EBSD`

Spatially indexed orientation map.

```python
from pymtex import CrystalSymmetry, Orientation, EBSD
import numpy as np

cs  = CrystalSymmetry('m-3m')
x   = np.linspace(0, 100, 50)
y   = np.linspace(0, 100, 50)
X, Y = np.meshgrid(x, y)
ori = Orientation.rand(X.size, cs)

ebsd = EBSD(X.ravel(), Y.ravel(), ori)
# Optional: phase index and phase names
ebsd = EBSD(X.ravel(), Y.ravel(), ori,
            phase=[1]*X.size,
            phase_names=['notIndexed', 'Aluminium'])

# Properties
ebsd.numPixels          # total points
ebsd.stepSize           # median nearest-neighbour distance in x
ebsd.boundingBox        # (xmin, xmax, ymin, ymax)
ebsd.isIndexed          # bool mask  (phase > 0)

# Filtering
ebsd.filter(mask)       # subset by boolean mask
ebsd.indexed()          # only phase > 0 points

# Analysis
ebsd.meanOrientation()  # volume-weighted mean orientation
kam = ebsd.calcKAM(max_angle=np.deg2rad(5))   # kernel average misorientation
grain_id, n_grains = ebsd.calcGrains(threshold=np.deg2rad(10))
```

---

### 7. Pole Figure I/O

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

#### LaboTEX — multi-pole, reads crystal symmetry from file

```python
from pymtex.texture import loadPoleFigureLaboTEX

pf = loadPoleFigureLaboTEX('LaboTEX.epf')
# Crystal symmetry, hkl, and grid parameters all read automatically.
print(pf)   # PoleFigure(3 poles: (111), (100), (110), cs='m-3m')
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
pf = loadPoleFigure('pf.txt', h, cs)                            # auto-detect
pf = loadPoleFigure('pf.txt', h, cs, degrees=False)             # radians
pf = loadPoleFigure('pf.txt', h, cs,
                    alpha_col=1, beta_col=2, intensity_col=0)   # reorder cols
pf = loadPoleFigure('pf.txt', h, cs, skiprows=3)                # skip header
pf = loadPoleFigure('pf.txt', h, cs, delimiter='\t')            # tab-separated
```

#### Bare matrix (no angle columns in the file)

```python
pf = loadPoleFigure('grid.dat', h, cs,
                    format='matrix',
                    alpha_start=0, alpha_stop=85, alpha_step=5,
                    beta_start=0,  beta_stop=355, beta_step=5)
```

#### Constructing a `PoleFigure` directly

```python
from pymtex.texture import PoleFigure
import numpy as np

# From pre-built (r, I) arrays
pf = PoleFigure(h_list, r_list, I_list, cs)

# From (theta, phi) spherical angles
pf = PoleFigure.from_spherical(h, theta, phi, intensities, cs)  # radians

# From a regular hemisphere grid (useful for synthetic tests)
pf = PoleFigure.from_grid(h, theta_max=np.pi/2, n_theta=19,
                           n_phi=36, intensities=None, cs=cs)

# Pre-processing
pf_norm = pf.normalize()              # unit mean per pole figure
pf_corr = pf.correct_background(50)  # subtract constant background

# Spherical nearest-neighbour lookup
I_at_r = pf.interp(j=0, r_query=r_new)   # j = pole index, r_new = (N,3)
```

---

### 8. ODF Calculation

```python
from pymtex.texture import calcODF

odf = calcODF(
    pf,                     # PoleFigure (call .normalize() first)
    resolution_deg=5,       # SO(3) grid step
    n_iter=25,              # WIMV iterations
    verbose=True,           # prints R-factor each iteration
)

# Statistics
odf.texture_index()    # J = ∫ f² dg  (1.0 = random)
odf.entropy()          # H = −∫ f·log(f) dg
odf.size               # number of SO(3) grid cells
```

**Grid resolution guide:**

| `resolution_deg` | SO(3) cells | Approx. time (25 iter, 3 poles) |
|---|---|---|
| 10° | ~13 k | ~3 s |
| 5° | ~98 k | ~30 s |
| 2.5° | ~780 k | ~5 min |

**Evaluating the ODF at arbitrary orientations:**

```python
ori     = Orientation.rand(200, cs)
f_vals  = odf.eval(ori)               # (200,) — f(g) values by NN lookup

r, pf_calc = odf.calcPF(h, n_theta=19, n_phi=72)   # forward-compute pole figure
# r : (M,3) specimen unit vectors
# pf_calc : (M,) intensities in m.r.d.
```

---

### 9. ODF File I/O

Save a computed ODF and reload it later — no need to re-run WIMV.

```python
from pymtex.texture import saveODF, loadODF
from pymtex import CrystalSymmetry

# Save  (4-column ASCII: phi1 Phi phi2 f — MTEX export compatible)
saveODF(odf, 'my_odf.txt')

# Reload — auto-detects format
cs  = CrystalSymmetry('m-3m')
odf = loadODF('my_odf.txt', cs)
odf = loadODF('texture.wts', cs)           # POPLA weight file

# Column-reorder (if file has a different layout)
odf = loadODF('odf.csv', cs, phi1_col=2, Phi_col=1,
               phi2_col=0, f_col=3)

# Radians in file
odf = loadODF('odf_rad.txt', cs, degrees=False)
```

**Saved file format:**
```
ODF  crystal symmetry: m-3m  J = 10.2456
% phi1(deg)  Phi(deg)  phi2(deg)  f(g)[m.r.d.]
  0.0000     0.0000     0.0000     1.052300
  5.0000     0.0000     5.0000     2.387600
  ...
```

**Supported ODF file formats:**

| Format | Extension | Auto-detected? |
|---|---|---|
| 4-column ASCII `(phi1 Phi phi2 f)` | `.txt`, `.csv`, `.dat` | ✓ |
| POPLA weight file | `.wts` | ✓ (from extension) |

> Not yet supported: BEARTEX `.cor`, MTEX `.mat` (MATLAB binary).
> Convert from MTEX first: `export(odf, 'my_odf.txt')`.

---

### 10. Plotting

All plots use the **jet** colormap by default. All pole figures in a figure
are guaranteed to be the same physical size (enforced via `ImageGrid`).

#### Plot types

| `plot_type=` | Description |
|---|---|
| `'scatter'` | Each measurement point drawn as a coloured dot — default for **measured** data |
| `'contour'` | Filled contours + thin contour-line overlay — default for **calculated** data |

#### Measured pole figures

```python
# Default: scatter, ≤3 per row, ≤4 rows per figure; overflow → new figure
fig = pf.plot()
fig = pf.plot(cmap='jet', plot_type='scatter')
fig = pf.plot(plot_type='contour', levels=15)
fig = pf.plot(projection='stereo')       # stereographic  (default: equal_area)

# >12 poles → returns a list
figs = pf.plot()
if isinstance(figs, list):
    for i, f in enumerate(figs):
        f.savefig(f'pf_part{i+1}.png', dpi=150, bbox_inches='tight')
```

#### Re-calculated pole figures (from ODF)

```python
ax  = odf.plotPF(Miller(hkl=[1,0,0], cs=cs))   # single pole, contour default
fig = odf.plotPFs(pf.h)                          # all poles, same layout rules
fig = odf.plotPFs(pf.h, cmap='hot', levels=20)
```

#### ODF φ₂ sections (MTEX style)

Each panel: f(φ₁, Φ) at a fixed φ₂.
φ₁ (0–360°) on x-axis; Φ (0–90°) on y-axis with 0° at top (MTEX convention).
Single shared colorbar — no overlap with panels.

```python
fig = odf.plotSections()                     # auto φ₂ range from symmetry
fig = odf.plotSections(
    phi2_vals=[0, 15, 30, 45, 60],           # custom section values (degrees)
    cmap='jet',
    levels=15,
    plot_type='contourf',                    # or 'pcolormesh'
    show_contour_lines=True,
)
fig.savefig('sections.png', dpi=150, bbox_inches='tight')
```

**Default φ₂ range by crystal system:**

| System | φ₂ range |
|---|---|
| Cubic | 0–90° |
| Hexagonal | 0–60° |
| Trigonal | 0–120° |
| Tetragonal | 0–90° |
| Orthorhombic | 0–90° |
| Monoclinic | 0–180° |
| Triclinic | 0–360° |

#### Inverse pole figure

```python
ax = odf.plotIPF()                           # specimen ND (default)
ax = odf.plotIPF(r_specimen=[1, 0, 0])       # rolling direction
ax = odf.plotIPF(plot_type='contour')
```

#### Interactive 3D ODF  (requires `pip install plotly`)

```python
odf.plotODF3D()                              # opens browser / VS Code panel

odf.plotODF3D(
    isomin=2.0,          # lowest isosurface (m.r.d.) — raise to declutter
    isomax=None,         # auto (= ODF maximum)
    surface_count=8,     # number of isosurface shells
    opacity=0.4,         # transparency per shell
    colorscale='Jet',    # 'Hot', 'Viridis', 'RdYlBu_r', 'Plasma', …
    show=True,           # False = return figure without displaying
)

# Save self-contained HTML (no Python required to view)
fig = odf.plotODF3D(show=False)
fig.write_html('odf_3d.html')
```

**Interactive controls:**  
`Rotate` — click+drag · `Zoom` — scroll · `Pan` — shift+drag  
`Hover` — shows φ₁, Φ, φ₂ and f(g) value on each isosurface

**Axes:** φ₁ (0–360°), Φ (0–90°), φ₂ (0–φ₂_max°) in Bunge convention.
Aspect ratio matches Euler space proportions automatically.

---

### 11. End-to-end example scripts

| Script | Dataset | Crystal | J |
|---|---|---|---|
| `analyse_labotex.py` | `LaboTEX.epf` — rolled aluminium | m-3m | ~10 |
| `analyse_go2.py` | `go_2_2.XPa` — quartz | -3m | ~1.4 |

Each script produces:
- `*_measured_pf.png` — measured pole figures (scatter)
- `*_recalculated_pf.png` — ODF re-calculated pole figures (contour)
- `*_comparison.png` — side-by-side measured vs calculated
- `*_odf_sections.png` — MTEX-style φ₂ sections
- `*_odf_3d.html` — interactive 3D ODF (open in any browser)

---

## Angle Conventions

| Convention | Details |
|---|---|
| Quaternion storage | `q = a + bi + cj + dk`, scalar part `a` first |
| Euler angles | **Bunge ZXZ** (φ₁, Φ, φ₂) as default; ZYZ/Matthies also supported |
| Rotation sense | Active (vectors are rotated, not the coordinate frame) |
| Pole figures | α = tilt from specimen ND (0–90°), β = rotation (0–360°) |
| Improper ops | Boolean flag on Rotation; multiplication uses XOR (MTEX convention) |
| Crystal axes | a = x, b = y, c = z (MTEX default); trigonal uses 321 setting |

---

## Project Structure

```
pymtex/
  geometry/
    quaternion.py      Quaternion  — base class, MTEX convention
    rotation.py        Rotation(Quaternion)  — improper flag, XOR multiply
    symmetry.py        CrystalSymmetry  — all 32 point groups
    orientation.py     Orientation  — rotation + crystal symmetry
    miller.py          Miller  — hkl / uvw indices, 4-index support
  ebsd/
    ebsd.py            EBSD  — spatial map, KAM, flood-fill grain segmentation
  texture/
    polefigure.py      PoleFigure  — measured data container + plotting
    odf.py             ODF  — discrete SO(3) grid + all ODF methods
    calcodf.py         calcODF  — WIMV iterative pole-figure inversion
    io.py              All file-format loaders + ODF save/load
    _plot_utils.py     ImageGrid layout (equal-size pole figures)
tests/                 189 unit tests  (geometry, EBSD, texture, I/O)
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
| Harmonic ODF inversion (Wigner D-functions) | High — faster + more accurate than WIMV |
| `ODFComponent` / kernel ODF fitting | Medium — fit sharp texture components |
| `SO3Grid` / fundamental-zone grids | Medium |
| Pole-figure symmetry-sector restriction in plots | Low |
| `@tensor` / elasticity tensors | Low |
| `@grain2d` / grain boundary network | Low |
| BEARTEX `.cor` ODF file reader | Low |
| Pole figure file export (write POPLA/BEARTEX) | Low |
