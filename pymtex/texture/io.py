"""
Pole figure file I/O – port of MTEX's loadPoleFigure_* interfaces.

Supported formats
-----------------
generic / column
    Any ASCII file with columns  alpha(deg)  beta(deg)  intensity.
    Comment lines (starting with ``#`` or non-numeric first token) are skipped.
    Column order can be re-mapped via *alpha_col*, *beta_col*, *intensity_col*.

popla / epf
    POPLA pole figure format (Kallend et al.).  File contains a one-line
    comment, a fixed-format parameter line, then free-form intensity values.
    Standard in neutron diffraction labs (NRSF2/SNS, HIPPO/LANL, etc.).

beartex
    BEARTEX software format (Wenk et al.).  Seven-line header, then 76 lines
    of 5-char-wide intensity values (18 α × 72 β = 1296 values).

matrix
    Bare intensity matrix (rows = α, cols = β) with grid parameters
    supplied as keyword arguments.

Angle conventions (all formats)
--------------------------------
α (alpha / theta): tilt from specimen normal direction ND  [0°, 90°]
β (beta  / rho):   rotation angle                          [0°, 360°)
Cartesian unit vector: r = (sin α cos β,  sin α sin β,  cos α)
"""

from __future__ import annotations

import os
import re
import numpy as np
from typing import Union, List, Optional

from pymtex.geometry.miller import Miller
from pymtex.geometry.symmetry import CrystalSymmetry
from pymtex.texture.polefigure import PoleFigure, _spherical_to_xyz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _alpha_beta_to_r(alpha_deg, beta_deg):
    """Convert tilt / rotation angles (degrees) to unit vectors."""
    a = np.deg2rad(np.asarray(alpha_deg, dtype=float))
    b = np.deg2rad(np.asarray(beta_deg,  dtype=float))
    return _spherical_to_xyz(a, b)   # (N, 3)


def _detect_format(fname: str, lines: list) -> str:
    """Infer file format from extension and content."""
    ext = os.path.splitext(fname)[1].lower()
    first_nonempty = next((l.strip() for l in lines if l.strip()), '')

    if ext.lower() in ('.xpa', '.xpb', '.xpc'):
        return 'xpa'
    if ext in ('.epf', '.pf'):
        # Distinguish LaboTEX from POPLA: LaboTEX has a
        # "n  number of Pole figures" line in the first ~10 lines.
        head = ' '.join(l.lower() for l in lines[:12])
        if 'number of pole' in head:
            return 'labotex'
        return 'popla'
    if first_nonempty.lower().startswith('beartex'):
        return 'beartex'
    if ext in ('.bea',):
        return 'beartex'
    # XPa: title lines end with '#'
    if any(l.strip().endswith('#') and len(l.strip()) > 1 for l in lines[:5]):
        return 'xpa'

    return 'generic'


def _read_lines(fname: str):
    with open(fname, 'r', errors='replace') as fh:
        return fh.readlines()


# ---------------------------------------------------------------------------
# Format parsers
# ---------------------------------------------------------------------------

def _parse_generic(lines, alpha_col=0, beta_col=1, intensity_col=2,
                   skiprows=0, degrees=True, delimiter=None):
    """
    Parse any ASCII column file:  alpha  beta  intensity  [more cols…]

    Parameters
    ----------
    lines : list of str
    alpha_col, beta_col, intensity_col : int  (0-based column indices)
    skiprows : int   explicit header lines to skip (on top of auto-skip)
    degrees  : bool  if True angles are in degrees, else radians
    delimiter: str or None

    Returns
    -------
    r : (N, 3)  unit vectors
    I : (N,)    intensities
    """
    rows = []
    n_skip = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            n_skip += 1
            continue
        # Check if the line starts with a numeric value
        first_tok = stripped.split(delimiter)[0] if delimiter else stripped.split()[0]
        try:
            float(first_tok)
        except ValueError:
            n_skip += 1
            continue
        break  # first numeric line found

    # Read with numpy, skipping non-numeric headers
    data = []
    in_data = False
    skip_count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        first_tok = stripped.split(delimiter)[0] if delimiter else stripped.split()[0]
        try:
            float(first_tok)
            in_data = True
        except ValueError:
            continue
        if in_data:
            skip_count += 1
            if skip_count <= skiprows:
                continue
            parts = stripped.split(delimiter) if delimiter else stripped.split()
            try:
                row = [float(p) for p in parts]
                data.append(row)
            except ValueError:
                continue

    if not data:
        raise ValueError("No numeric data found in file.")

    arr = np.array(data)
    if arr.shape[1] <= max(alpha_col, beta_col, intensity_col):
        raise ValueError(
            f"File has only {arr.shape[1]} columns; "
            f"requested columns {alpha_col}, {beta_col}, {intensity_col}."
        )

    alpha = arr[:, alpha_col]
    beta  = arr[:, beta_col]
    I     = arr[:, intensity_col]

    if not degrees:
        alpha = np.degrees(alpha)
        beta  = np.degrees(beta)

    r = _alpha_beta_to_r(alpha, beta)
    return r, np.maximum(I, 0)


def _parse_popla(lines):
    """
    Parse POPLA EPF format.

    Line 1: title / comment (ignored)
    Line 2: fixed-format parameter string:
        chars  1– 5: hkl  (e.g. "  1 1")        – NOT parsed here
        chars  6-10: dtheta  step in alpha (deg)
        chars 11-15: mtheta  max alpha (deg)
        chars 16-20: drho    step in beta (deg)
        chars 21-25: mrho    max beta (deg)
        chars 26-27: shifttheta  (0 or 1)
        chars 28-29: shiftrho    (0 or 1)
        … (scaling, background ignored)
    Lines 3+: intensity values (free-form, any number per line)

    Returns
    -------
    r : (N, 3) unit vectors
    I : (N,)   intensities
    grid_params : dict  {'dtheta', 'mtheta', 'drho', 'mrho'}
    """
    # Skip blank / comment lines to find the parameter line
    data_lines = [l for l in lines if l.strip()]

    if len(data_lines) < 2:
        raise ValueError("POPLA file has fewer than 2 non-empty lines.")

    # title = data_lines[0]  (ignored)
    param_line = data_lines[1]

    # Parse parameter line with fixed 5-char fields, then 2-char fields
    def _field5(s, k):
        start = 5 * k
        end   = 5 * (k + 1)
        return s[start:end].strip() if len(s) >= end else ''

    def _field2(s, k):
        start = 25 + 2 * k
        end   = start + 2
        return s[start:end].strip() if len(s) >= end else ''

    # --- Try strict 5-char fixed-width fields ---
    parsed = False
    try:
        dtheta = float(_field5(param_line, 1))
        mtheta = float(_field5(param_line, 2))
        drho   = float(_field5(param_line, 3))
        mrho   = float(_field5(param_line, 4))
        assert 0 < dtheta < 90 and 0 < mtheta <= 180
        assert 0 < drho  < 90 and 0 < mrho   <= 360
        shifttheta = int(_field2(param_line, 0) or '0')
        shiftrho   = int(_field2(param_line, 1) or '0')
        parsed = True
    except (ValueError, AssertionError, IndexError):
        pass

    # --- Fallback: whitespace-token heuristic ---
    if not parsed:
        nums = []
        for tok in param_line.split():
            try:
                nums.append(float(tok))
            except ValueError:
                pass
        # Expect sequence: [h k l?]  dtheta  mtheta  drho  mrho  [shift…]
        # Find a run of 4 values consistent with grid parameters.
        found = False
        for i in range(len(nums) - 3):
            dt, mt, dr, mr = nums[i], nums[i+1], nums[i+2], nums[i+3]
            if (0 < dt < 90 and 30 <= mt <= 180 and
                    0 < dr < 90 and 100 <= mr <= 360):
                dtheta, mtheta, drho, mrho = dt, mt, dr, mr
                shifttheta = int(nums[i+4]) if i+4 < len(nums) and nums[i+4] in (0,1) else 0
                shiftrho   = int(nums[i+5]) if i+5 < len(nums) and nums[i+5] in (0,1) else 0
                found = True
                break
        if not found:
            raise ValueError(
                "Could not parse POPLA parameter line.\n"
                f"  Line content: {param_line!r}\n"
                "  Expected fixed 5-char fields or space-separated: "
                "dθ  θmax  dβ  βmax  …"
            )

    # Alpha grid
    a_start = 0.0 if shifttheta else dtheta / 2.0
    alpha_vals = np.arange(a_start, mtheta + dtheta / 4, dtheta)

    # Beta grid
    b_start = 0.0 if shiftrho else drho / 2.0
    beta_vals = np.arange(b_start, mrho, drho)

    n_alpha = len(alpha_vals)
    n_beta  = len(beta_vals)

    # Read all intensity numbers from lines 3 onwards
    intensity_flat = []
    for line in data_lines[2:]:
        for tok in line.split():
            try:
                intensity_flat.append(float(tok))
            except ValueError:
                pass

    expected = n_alpha * n_beta
    if len(intensity_flat) < expected:
        raise ValueError(
            f"POPLA file: expected {expected} intensity values "
            f"({n_alpha} α × {n_beta} β) but found {len(intensity_flat)}."
        )

    I_matrix = np.array(intensity_flat[:expected]).reshape(n_alpha, n_beta)

    # Build direction grid
    ALPHA, BETA = np.meshgrid(alpha_vals, beta_vals, indexing='ij')
    r = _alpha_beta_to_r(ALPHA.ravel(), BETA.ravel())
    I = np.maximum(I_matrix.ravel(), 0)

    return r, I, {'dtheta': dtheta, 'mtheta': mtheta,
                  'drho': drho, 'mrho': mrho}


def _parse_beartex(lines):
    """
    Parse BEARTEX pole figure format.

    7-line header:
        Line 1: 'beartex…' identifier
        Line 2: sample title
        Line 3: comment
        Line 4: comment
        Line 5: comment
        Line 6: crystal parameters  a b c α β γ (Å/°) + symmetry ids
        Line 7: hkl  info_block  (5-char fixed fields: θmin θmax dθ βmax dβ scale)

    Then 76 lines of intensity data (each line has ≤ 20 values in 5-char fields;
    total 18 α × 72 β = 1296 values from α = 0°…85° and β = 0°…355°).

    Returns
    -------
    r : (N, 3) unit vectors
    I : (N,)   intensities
    """
    data_lines = [l.rstrip('\n') for l in lines]

    # Skip leading blank lines
    start = 0
    for i, l in enumerate(data_lines):
        if l.strip():
            start = i
            break

    if len(data_lines) - start < 7:
        raise ValueError("BEARTEX file has fewer than 7 header lines.")

    info_line = data_lines[start + 6]  # Line 7 (0-indexed)

    # Parse info block: 5-char fields starting at column 10
    def _bf(s, k):
        pos = 10 + 5 * k
        return s[pos: pos + 5].strip() if len(s) >= pos + 5 else '0'

    try:
        theta_min = float(_bf(info_line, 0))
        theta_max = float(_bf(info_line, 1))
        d_theta   = float(_bf(info_line, 2))
        beta_max  = float(_bf(info_line, 3))
        d_beta    = float(_bf(info_line, 4))
    except (ValueError, IndexError):
        # Fall back to MTEX defaults: 18 alpha steps (5°, 0–85°), 72 beta steps (5°, 0–355°)
        theta_min, theta_max, d_theta = 0.0, 85.0, 5.0
        beta_max, d_beta = 355.0, 5.0

    alpha_vals = np.arange(theta_min, theta_max + d_theta / 4, d_theta)
    beta_vals  = np.arange(0.0,      beta_max  + d_beta  / 4, d_beta)
    n_alpha = len(alpha_vals)
    n_beta  = len(beta_vals)

    # Read intensity values from line 8 onwards (5-char wide each)
    intensity_flat = []
    for line in data_lines[start + 7:]:
        # 5-char fixed-width fields
        for pos in range(0, len(line) - 3, 5):
            tok = line[pos: pos + 5].strip()
            if tok:
                try:
                    intensity_flat.append(float(tok))
                except ValueError:
                    pass

    expected = n_alpha * n_beta
    if len(intensity_flat) < expected:
        # Fallback: try free-form parsing
        intensity_flat = []
        for line in data_lines[start + 7:]:
            for tok in line.split():
                try:
                    intensity_flat.append(float(tok))
                except ValueError:
                    pass

    if len(intensity_flat) < expected:
        raise ValueError(
            f"BEARTEX file: expected {expected} intensity values "
            f"({n_alpha} α × {n_beta} β) but found {len(intensity_flat)}."
        )

    I_matrix = np.array(intensity_flat[:expected]).reshape(n_alpha, n_beta)
    ALPHA, BETA = np.meshgrid(alpha_vals, beta_vals, indexing='ij')
    r = _alpha_beta_to_r(ALPHA.ravel(), BETA.ravel())
    I = np.maximum(I_matrix.ravel(), 0)
    return r, I


def _parse_matrix(lines, alpha_start=0.0, alpha_stop=90.0, alpha_step=5.0,
                  beta_start=0.0, beta_stop=355.0, beta_step=5.0,
                  skiprows=0):
    """
    Parse a plain intensity matrix (no angle information in the file).

    Row index → alpha, column index → beta.
    """
    alpha_vals = np.arange(alpha_start, alpha_stop + alpha_step / 4, alpha_step)
    beta_vals  = np.arange(beta_start,  beta_stop  + beta_step  / 4, beta_step)

    numbers = []
    skip = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        first = stripped.split()[0]
        try:
            float(first)
        except ValueError:
            if skip < skiprows:
                skip += 1
                continue
        for tok in stripped.split():
            try:
                numbers.append(float(tok))
            except ValueError:
                pass

    n_alpha, n_beta = len(alpha_vals), len(beta_vals)
    expected = n_alpha * n_beta
    if len(numbers) < expected:
        raise ValueError(
            f"Matrix file: expected {expected} values "
            f"({n_alpha}×{n_beta}) but found {len(numbers)}."
        )
    I_mat = np.array(numbers[:expected]).reshape(n_alpha, n_beta)
    ALPHA, BETA = np.meshgrid(alpha_vals, beta_vals, indexing='ij')
    r = _alpha_beta_to_r(ALPHA.ravel(), BETA.ravel())
    return r, np.maximum(I_mat.ravel(), 0)


# ---------------------------------------------------------------------------
# LaboTEX .epf multi-pole-figure format
# ---------------------------------------------------------------------------

def _parse_labotex(lines):
    """
    Parse LaboTEX .epf pole figure file.

    Header (0-indexed lines)
    ~~~~~~~~~~~~~~~~~~~~~~~~
    0  : title / sample description
    1  : comment
    2  : column headers for structure  (Structure Code  a  b  c  ALPHA BETA GAMMA)
    3  : structure params              code  a  b  c  alpha  beta  gamma
    4  : "n  number of Pole figures…"
    5  : column headers for PF params  (2TH  A-first  A-last  A-step  …  H  K  L  BG)
    6…6+n-1 : one parameter line per pole figure
    6+n … : intensity data (all PFs concatenated, free-form numeric)

    PF parameter line columns
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    0  2theta      1  alpha_first   2  alpha_last    3  alpha_step
    4  beta_first  5  beta_last     6  beta_step      7  scan_mode
    8  H           9  K            10  L             11  Background

    Intensities are written row-major: alpha varies slow, beta varies fast.

    Returns
    -------
    list of dict – same structure as ``_parse_xpa``
    """
    # --- locate "number of Pole figures" line (usually line index 4) ---
    npf_idx  = None
    for i, line in enumerate(lines[:15]):
        if 'number of pole' in line.lower():
            npf_idx = i
            break
    if npf_idx is None:
        raise ValueError("LaboTEX: could not find 'number of Pole figures' line.")

    n_pf = int(lines[npf_idx].split()[0])

    # --- crystal symmetry from structure-params line (index 3) ---
    cs_id = 7   # default cubic
    try:
        cs_id = int(float(lines[3].split()[0]))
    except (ValueError, IndexError):
        pass
    cs_name = _BEARTEX_CS_ID.get(cs_id, 'm-3m')

    # --- PF parameter lines start two lines after npf_idx (skip header row) ---
    param_start = npf_idx + 2
    pf_params   = []
    for j in range(n_pf):
        tok = lines[param_start + j].split()
        pf_params.append({
            'a_first': float(tok[1]), 'a_last': float(tok[2]),
            'a_step':  float(tok[3]),
            'b_first': float(tok[4]), 'b_last': float(tok[5]),
            'b_step':  float(tok[6]),
            'hkl':     (int(float(tok[8])), int(float(tok[9])),
                        int(float(tok[10]))),
        })

    # --- intensity data: everything after the PF parameter block ---
    data_start = param_start + n_pf
    all_vals   = []
    for line in lines[data_start:]:
        for tok in line.split():
            try:
                all_vals.append(float(tok))
            except ValueError:
                pass

    # --- split into per-PF arrays ---
    results = []
    offset  = 0
    for p in pf_params:
        alpha_vals = np.arange(p['a_first'], p['a_last'] + p['a_step'] / 4,
                               p['a_step'])
        beta_vals  = np.arange(p['b_first'], p['b_last'] + p['b_step']  / 4,
                               p['b_step'])
        n_alpha, n_beta = len(alpha_vals), len(beta_vals)
        expected        = n_alpha * n_beta

        if offset + expected > len(all_vals):
            raise ValueError(
                f"LaboTEX: expected {expected} values for PF {p['hkl']}, "
                f"only {len(all_vals) - offset} remain."
            )

        I_mat = np.array(all_vals[offset:offset + expected]).reshape(
            n_alpha, n_beta)
        offset += expected

        ALPHA, BETA = np.meshgrid(alpha_vals, beta_vals, indexing='ij')
        r = _alpha_beta_to_r(ALPHA.ravel(), BETA.ravel())
        I = np.maximum(I_mat.ravel(), 0.0)

        results.append({
            'hkl':     p['hkl'],
            'r':       r,
            'I':       I,
            'cs_id':   cs_id,
            'cs_name': cs_name,
        })

    return results


def loadPoleFigureLaboTEX(fname, cs=None, ss=None):
    """
    Load all pole figures from a LaboTEX ``.epf`` file.

    Crystal symmetry is read from the file (structure-code field) unless
    *cs* is explicitly supplied.

    Parameters
    ----------
    fname : str
    cs : CrystalSymmetry, optional
    ss : CrystalSymmetry, optional

    Returns
    -------
    PoleFigure
    """
    lines  = _read_lines(fname)
    blocks = _parse_labotex(lines)

    if not blocks:
        raise ValueError(f"No valid pole figures found in {fname!r}.")

    if cs is None:
        cs = CrystalSymmetry(blocks[0]['cs_name'])

    h_list = [Miller(hkl=list(b['hkl']), cs=cs) for b in blocks]
    r_list = [b['r'] for b in blocks]
    I_list = [b['I'] for b in blocks]

    return PoleFigure(h_list, r_list, I_list, cs, ss)


# ---------------------------------------------------------------------------
# XPa / BEARTEX multi-pole-figure format
# ---------------------------------------------------------------------------

# Map BEARTEX symmetry ID → HM point group symbol (Laue group used for texture)
_BEARTEX_CS_ID = {
    1: '1',    # C1  → triclinic
    2: '2/m',  # C2  → monoclinic Laue
    3: 'mmm',  # D2  → orthorhombic Laue
    4: '4/m',  # C4  → tetragonal
    5: '4/mmm',# D4  → tetragonal Laue
    6: 'm-3',  # T   → cubic
    7: 'm-3m', # O   → cubic Laue
    8: '-3',   # C3  → trigonal
    9: '-3m',  # D3  → trigonal Laue  (quartz uses this)
    10: '6/m', # C6  → hexagonal
    11: '6/mmm',# D6 → hexagonal Laue
}


def _parse_xpa_block(block):
    """
    Parse one pole-figure block from an XPa file.

    Block structure (lines are 0-indexed, blanks preserved):
        0   : title line (ends with '#')
        1   : sample description
        2-4 : blank lines
        5   : crystal parameters  a b c α β γ  cs_id  ss_id
        6   : grid line  h k l  θ_min θ_max dθ  β_min β_max dβ  flag1 flag2
        7+  : intensity data (4-char wide fields, 18 per line, 76 lines)

    Returns
    -------
    hkl        : (int, int, int)
    alpha_vals : np.ndarray  (degrees)
    beta_vals  : np.ndarray  (degrees)
    I_matrix   : np.ndarray, shape (n_alpha, n_beta)
    cs_id      : int
    """
    if len(block) < 8:
        raise ValueError("Block too short to contain a pole figure.")

    crys_line  = block[5]
    param_line = block[6]

    # --- Crystal symmetry ID ---
    crys_tokens = crys_line.split()
    cs_id = int(float(crys_tokens[6])) if len(crys_tokens) >= 7 else 9

    # --- hkl: first 3 integers on the grid line ---
    hkl_vals = []
    for tok in param_line.split():
        try:
            hkl_vals.append(int(float(tok)))
        except ValueError:
            pass
        if len(hkl_vals) == 3:
            break
    h, k, l = (hkl_vals + [0, 0, 0])[:3]

    # --- Grid parameters: 5-char fixed-width fields from position 10 ---
    # Layout (0-indexed):  chars 10-14 = θ_min, 15-19 = θ_max, 20-24 = dθ,
    #                      25-29 = β_min, 30-34 = β_max, 35-39 = dβ
    # (The "0.0355.0" concatenation at positions 25-34 is handled by the
    #  fixed-width slicing: chars 25-29 = "  0.0" and 30-34 = "355.0".)
    try:
        p = param_line
        theta_min = float(p[10:15].strip())
        theta_max = float(p[15:20].strip())
        d_theta   = float(p[20:25].strip())
        beta_min  = float(p[25:30].strip())
        beta_max  = float(p[30:35].strip())
        d_beta    = float(p[35:40].strip())
    except (ValueError, IndexError):
        # Fallback: skip the 3 hkl tokens and read next 6 numeric values
        nums = [float(t) for t in param_line.split() if _is_numeric(t)]
        params = nums[3:9]
        theta_min, theta_max, d_theta = params[0], params[1], params[2]
        beta_min,  beta_max,  d_beta  = params[3], params[4], params[5]

    alpha_vals = np.arange(theta_min, theta_max + d_theta / 4, d_theta)
    beta_vals  = np.arange(beta_min,  beta_max  + d_beta  / 4, d_beta)
    n_alpha, n_beta = len(alpha_vals), len(beta_vals)

    # --- Intensity data: 4-char wide, skip first char of each data line ---
    all_vals = []
    for line in block[7:]:
        raw = line.rstrip('\n\r')
        if not raw.strip():
            continue
        pos = 1  # skip leading space
        while pos + 4 <= len(raw):
            chunk = raw[pos:pos + 4].strip()
            if chunk:
                try:
                    all_vals.append(float(chunk))
                except ValueError:
                    pass
            pos += 4
        if len(all_vals) >= n_alpha * n_beta:
            break

    expected = n_alpha * n_beta
    if len(all_vals) < expected:
        raise ValueError(
            f"XPa block hkl=({h},{k},{l}): expected {expected} values "
            f"({n_alpha}α×{n_beta}β) but found {len(all_vals)}."
        )

    I_mat = np.array(all_vals[:expected]).reshape(n_alpha, n_beta)
    return (h, k, l), alpha_vals, beta_vals, I_mat, cs_id


def _is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _parse_xpa(lines, cs_override=None):
    """
    Parse an XPa file (multi-pole-figure BEARTEX variant).

    The file may contain any number of pole figure blocks.  Each block
    starts with a title line that ends with ``#``.

    Parameters
    ----------
    lines : list of str
    cs_override : CrystalSymmetry or None
        If given, use this symmetry instead of reading from the file.

    Returns
    -------
    list of dict, each with keys:
        'hkl'        : (h, k, l) tuple
        'r'          : np.ndarray (M, 3)  specimen directions
        'I'          : np.ndarray (M,)    intensities
        'cs_id'      : int
        'cs_name'    : str  HM point group name
    """
    # Split into blocks: a new block starts with a line containing '#'
    block_starts = []
    for i, line in enumerate(lines):
        if line.strip().endswith('#') and len(line.strip()) > 1:
            block_starts.append(i)

    if not block_starts:
        raise ValueError("No XPa pole figure blocks found (no '#'-terminated title lines).")

    results = []
    for bi, start in enumerate(block_starts):
        end = block_starts[bi + 1] if bi + 1 < len(block_starts) else len(lines)
        block = lines[start:end]   # raw lines, blanks preserved

        try:
            (h, k, l), alpha_vals, beta_vals, I_mat, cs_id = \
                _parse_xpa_block(block)
        except (ValueError, IndexError):
            continue

        cs_name = _BEARTEX_CS_ID.get(cs_id, '32')
        ALPHA, BETA = np.meshgrid(alpha_vals, beta_vals, indexing='ij')
        r = _alpha_beta_to_r(ALPHA.ravel(), BETA.ravel())
        I = np.maximum(I_mat.ravel(), 0)

        results.append({
            'hkl': (h, k, l),
            'r':    r,
            'I':    I,
            'cs_id':   cs_id,
            'cs_name': cs_name,
        })

    return results


def loadPoleFigureXPa(fname, cs=None, ss=None):
    """
    Load all pole figures from an XPa (multi-pole BEARTEX variant) file.

    Crystal symmetry is read from the file unless *cs* is explicitly provided.

    Parameters
    ----------
    fname : str
    cs : CrystalSymmetry, optional
    ss : CrystalSymmetry, optional

    Returns
    -------
    PoleFigure
    """
    lines = _read_lines(fname)
    blocks = _parse_xpa(lines)

    if not blocks:
        raise ValueError(f"No valid pole figures found in {fname!r}.")

    cs_name = blocks[0]['cs_name']
    if cs is None:
        cs = CrystalSymmetry(cs_name)

    h_list = [Miller(hkl=list(b['hkl']), cs=cs) for b in blocks]
    r_list = [b['r'] for b in blocks]
    I_list = [b['I'] for b in blocks]

    return PoleFigure(h_list, r_list, I_list, cs, ss)


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def loadPoleFigure(
    fname: Union[str, List[str]],
    h: Union[Miller, List[Miller]],
    cs: CrystalSymmetry,
    ss: Optional[CrystalSymmetry] = None,
    format: str = 'auto',
    # Generic column options
    alpha_col: int = 0,
    beta_col:  int = 1,
    intensity_col: int = 2,
    skiprows: int = 0,
    degrees: bool = True,
    delimiter: Optional[str] = None,
    # Matrix / POPLA / BEARTEX grid override
    alpha_start: float = 0.0,
    alpha_stop:  float = 90.0,
    alpha_step:  float = 5.0,
    beta_start:  float = 0.0,
    beta_stop:   float = 355.0,
    beta_step:   float = 5.0,
) -> PoleFigure:
    """
    Load pole figure(s) from one or more files.

    Parameters
    ----------
    fname : str or list of str
        Path(s) to pole figure file(s).  When a list is given each file is
        loaded as one pole figure; *h* must then also be a list of the same
        length.
    h : Miller or list of Miller
        Crystal direction(s) corresponding to each file.
    cs : CrystalSymmetry
    ss : CrystalSymmetry, optional
        Specimen symmetry (default: triclinic).
    format : str
        One of ``'auto'``, ``'generic'`` (column ASCII), ``'popla'``/``'epf'``,
        ``'beartex'``, or ``'matrix'``.  ``'auto'`` detects from extension and
        file content.

    Column-format options (format='generic')
    ----------------------------------------
    alpha_col, beta_col, intensity_col : int
        0-based column indices (default 0, 1, 2).
    skiprows : int
        Additional header rows to skip after auto-skip.
    degrees : bool
        ``True`` if angles are in degrees (default).
    delimiter : str or None
        Column delimiter; ``None`` means any whitespace.

    Grid-override options (format='matrix' or when overriding POPLA/BEARTEX)
    -------------------------------------------------------------------------
    alpha_start, alpha_stop, alpha_step : float  (degrees)
    beta_start,  beta_stop,  beta_step  : float  (degrees)

    Returns
    -------
    PoleFigure

    Examples
    --------
    **Generic column file** ``(alpha deg, beta deg, intensity)``::

        from pymtex import CrystalSymmetry, Miller
        from pymtex.texture import loadPoleFigure

        cs  = CrystalSymmetry('m-3m')
        pf  = loadPoleFigure('pf_111.txt',
                              Miller(hkl=[1,1,1], cs=cs), cs)

    **POPLA EPF file**::

        pf = loadPoleFigure('cu_111.epf',
                             Miller(hkl=[1,1,1], cs=cs), cs,
                             format='popla')

    **BEARTEX file**::

        pf = loadPoleFigure('cu.bea',
                             Miller(hkl=[1,0,0], cs=cs), cs,
                             format='beartex')

    **Multiple files, one per pole**::

        pf = loadPoleFigure(
            ['100.epf', '110.epf', '111.epf'],
            [Miller(hkl=[1,0,0], cs=cs),
             Miller(hkl=[1,1,0], cs=cs),
             Miller(hkl=[1,1,1], cs=cs)],
            cs, format='popla')

    **Re-ordering columns** (file has: intensity, alpha, beta)::

        pf = loadPoleFigure('pf.txt', h, cs,
                             alpha_col=1, beta_col=2, intensity_col=0)

    **Intensity-matrix file** (no angle columns, just a grid of numbers)::

        pf = loadPoleFigure('pf_grid.dat', h, cs,
                             format='matrix',
                             alpha_start=0, alpha_stop=85, alpha_step=5,
                             beta_start=0,  beta_stop=355, beta_step=5)
    """
    # Normalise inputs to lists
    if isinstance(fname, str):
        fnames = [fname]
    else:
        fnames = list(fname)

    if isinstance(h, Miller):
        h_list = [h] * len(fnames)
    else:
        h_list = list(h)

    if len(fnames) != len(h_list):
        raise ValueError(
            f"Number of files ({len(fnames)}) must match number of "
            f"Miller indices ({len(h_list)})."
        )

    r_list = []
    I_list = []

    for fname_i, h_i in zip(fnames, h_list):
        lines = _read_lines(fname_i)

        # Determine format
        fmt = format.lower()
        if fmt == 'auto':
            fmt = _detect_format(fname_i, lines)
        if fmt in ('epf',):
            fmt = 'popla'

        if fmt == 'generic' or fmt == 'column':
            r_i, I_i = _parse_generic(
                lines,
                alpha_col=alpha_col,
                beta_col=beta_col,
                intensity_col=intensity_col,
                skiprows=skiprows,
                degrees=degrees,
                delimiter=delimiter,
            )

        elif fmt == 'popla':
            r_i, I_i, _ = _parse_popla(lines)

        elif fmt == 'beartex':
            r_i, I_i = _parse_beartex(lines)

        elif fmt in ('labotex',):
            raise ValueError(
                "LaboTEX .epf files contain multiple poles in one file. "
                "Use loadPoleFigureLaboTEX(fname) instead of loadPoleFigure()."
            )

        elif fmt == 'matrix':
            r_i, I_i = _parse_matrix(
                lines,
                alpha_start=alpha_start, alpha_stop=alpha_stop,
                alpha_step=alpha_step,
                beta_start=beta_start,   beta_stop=beta_stop,
                beta_step=beta_step,
                skiprows=skiprows,
            )

        else:
            raise ValueError(
                f"Unknown format {format!r}.  "
                "Use 'auto', 'generic', 'popla', 'beartex', or 'matrix'."
            )

        r_list.append(r_i)
        I_list.append(I_i)

    return PoleFigure(h_list, r_list, I_list, cs, ss)


# Convenience alias
load_pole_figure = loadPoleFigure


# ===========================================================================
# ODF file I/O
# ===========================================================================
"""
Supported ODF file formats
--------------------------
generic / column
    4-column ASCII:  phi1(deg)  Phi(deg)  phi2(deg)  f(g)[m.r.d.]
    Comment lines starting with ``%`` or ``#`` are skipped.
    This is the format written by MTEX's ``export(odf, fname)`` and by
    PyMTEX's own ``saveODF``.

popla / wts
    POPLA ODF weight file (.wts).  Header encodes the grid, then values
    row-major:  phi1 varies slow, phi2 varies fast, Phi in the middle.

    Header format::
        <step_deg>    phi1/Phi/phi2 step (single value, degrees)
        <n_phi1>  <n_Phi>  <n_phi2>   grid dimensions
        <values …>

auto
    Detected from file extension (.wts → popla) and content.
"""


def _detect_odf_format(fname: str, lines: list) -> str:
    ext = os.path.splitext(fname)[1].lower()
    if ext in ('.wts',):
        return 'popla'
    # Generic: has 4 numeric columns (possibly after % / # headers)
    return 'generic'


# ---------------------------------------------------------------------------
# ODF parsers
# ---------------------------------------------------------------------------

def _parse_odf_generic(lines, phi1_col=0, Phi_col=1, phi2_col=2, f_col=3,
                        skiprows=0, delimiter=None):
    """
    Parse a 4-column ASCII ODF file.  Skips lines whose first token starts
    with '%' or '#', or is non-numeric (e.g. column-name headers).
    """
    data = []
    skipped = 0
    for line in lines:
        s = line.strip()
        if not s:
            continue
        first = (s.split(delimiter) if delimiter else s.split())[0]
        if first.startswith('%') or first.startswith('#'):
            continue
        try:
            float(first)
        except ValueError:
            if skipped < skiprows:
                skipped += 1
            continue
        parts = s.split(delimiter) if delimiter else s.split()
        try:
            data.append([float(parts[i])
                         for i in (phi1_col, Phi_col, phi2_col, f_col)])
        except (IndexError, ValueError):
            pass

    if not data:
        raise ValueError("No numeric data found in ODF file.")

    arr  = np.array(data)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def _parse_odf_popla(lines):
    """
    Parse a POPLA .wts ODF weight file.

    Expected header (first two non-blank, non-comment lines):
        Line 1:  step_deg              (single float, degrees)
        Line 2:  n_phi1  n_Phi  n_phi2 (three integers)
    Then all ODF values in reading order: phi2 varies fastest, phi1 slowest.
    """
    nums = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith('%') or s.startswith('#'):
            continue
        for tok in s.split():
            try:
                nums.append(float(tok))
            except ValueError:
                pass

    if len(nums) < 4:
        raise ValueError("POPLA .wts file too short.")

    # First value = step (degrees), next three = grid dimensions
    step      = nums[0]
    n_phi1    = int(nums[1])
    n_Phi     = int(nums[2])
    n_phi2    = int(nums[3])
    expected  = n_phi1 * n_Phi * n_phi2

    if len(nums) < 4 + expected:
        raise ValueError(
            f"POPLA .wts: expected {expected} ODF values "
            f"({n_phi1}×{n_Phi}×{n_phi2}) but found {len(nums) - 4}."
        )

    f_flat = np.array(nums[4: 4 + expected])

    phi1_vals = np.arange(0, n_phi1) * step
    Phi_vals  = np.arange(0, n_Phi)  * step
    phi2_vals = np.arange(0, n_phi2) * step

    PHI1, PHI, PHI2 = np.meshgrid(phi1_vals, Phi_vals, phi2_vals,
                                    indexing='ij')
    f_arr = f_flat.reshape(n_phi1, n_Phi, n_phi2)

    return (PHI1.ravel(), PHI.ravel(), PHI2.ravel(), f_arr.ravel())


# ---------------------------------------------------------------------------
# Public ODF loaders / savers
# ---------------------------------------------------------------------------

def loadODF(fname, cs, ss=None, format='auto',
            phi1_col=0, Phi_col=1, phi2_col=2, f_col=3,
            skiprows=0, degrees=True, delimiter=None):
    """
    Load a pre-computed ODF from a file.

    The file may contain ODF values on any regular grid of Bunge Euler angles.
    PyMTEX creates an :class:`~pymtex.texture.ODF` object directly from the
    stored (φ₁, Φ, φ₂, f) data; no SO(3) grid is assumed — the grid is
    whatever the file contains.

    Parameters
    ----------
    fname : str
        Path to the ODF file.
    cs : CrystalSymmetry
    ss : CrystalSymmetry, optional
    format : str
        ``'auto'`` (default), ``'generic'`` / ``'column'``, or ``'popla'``.
    phi1_col, Phi_col, phi2_col, f_col : int
        0-based column indices (generic format only).
    skiprows : int
        Extra header rows to skip (generic format).
    degrees : bool
        ``True`` (default) if angles in the file are in degrees.
    delimiter : str or None
        Column separator for generic format.

    Returns
    -------
    ODF

    Examples
    --------
    Load an ODF saved by PyMTEX or exported from MTEX::

        odf = loadODF('my_odf.txt', CrystalSymmetry('m-3m'))

    Load a POPLA weight file::

        odf = loadODF('texture.wts', CrystalSymmetry('m-3m'))

    Load when columns are in a different order::

        odf = loadODF('odf.txt', cs, phi1_col=2, Phi_col=1,
                      phi2_col=0, f_col=3)
    """
    from pymtex.geometry.orientation import Orientation
    from pymtex.texture.odf import ODF

    lines = _read_lines(fname)

    fmt = format.lower()
    if fmt == 'auto':
        fmt = _detect_odf_format(fname, lines)

    if fmt in ('generic', 'column', 'mtex'):
        phi1, Phi, phi2, f = _parse_odf_generic(
            lines, phi1_col, Phi_col, phi2_col, f_col, skiprows, delimiter)
    elif fmt in ('popla', 'wts'):
        phi1, Phi, phi2, f = _parse_odf_popla(lines)
        degrees = True   # POPLA always in degrees
    else:
        raise ValueError(
            f"Unknown ODF format {format!r}.  "
            "Use 'auto', 'generic', or 'popla'."
        )

    phi1 = np.asarray(phi1, dtype=float)
    Phi  = np.asarray(Phi,  dtype=float)
    phi2 = np.asarray(phi2, dtype=float)
    f    = np.maximum(np.asarray(f, dtype=float), 0.0)

    if degrees:
        phi1_rad = np.deg2rad(phi1)
        Phi_rad  = np.deg2rad(Phi)
        phi2_rad = np.deg2rad(phi2)
    else:
        phi1_rad, Phi_rad, phi2_rad = phi1, Phi, phi2

    ori = Orientation.byEuler(phi1_rad, Phi_rad, phi2_rad,
                               cs, convention='Bunge')

    # Volume weights ∝ sin(Φ) — normalised so Σ w_i = 1
    w = np.sin(Phi_rad)
    w = np.where(w < 1e-10, 1e-10, w)   # guard sin(0) = 0 at the pole
    w /= w.sum()

    # Normalise f so that Σ f_i * w_i = 1 (ODF convention)
    norm = np.dot(f, w)
    if norm > 0:
        f = f / norm

    return ODF(ori, f, w, cs, ss)


def saveODF(odf, fname, degrees=True, fmt_str='%10.4f %10.4f %10.4f %14.6f'):
    """
    Save an ODF to a 4-column ASCII file.

    The output format is::

        % phi1(deg)  Phi(deg)  phi2(deg)  f(g)[m.r.d.]
        0.0000     0.0000     0.0000     1.052300
        5.0000     0.0000     0.0000     0.987600
        ...

    This is the same column layout as MTEX's ``export(odf, fname)`` so the
    file can be re-imported by either program.

    Parameters
    ----------
    odf : ODF
    fname : str
    degrees : bool
        Write angles in degrees (default) or radians.
    fmt_str : str
        printf format string for each row.

    Examples
    --------
    Round-trip a computed ODF::

        saveODF(odf, 'my_odf.txt')
        odf2 = loadODF('my_odf.txt', CrystalSymmetry('m-3m'))
    """
    phi1, Phi, phi2 = odf.orientations.to_euler('Bunge')
    if degrees:
        phi1 = np.degrees(phi1) % 360.0
        Phi  = np.degrees(Phi)  % 180.0   # Phi ∈ [0°, 180°]
        phi2 = np.degrees(phi2) % 360.0

    header = (
        f'ODF  crystal symmetry: {odf.cs.name}  '
        f'J = {odf.texture_index():.4f}\n'
        + ('% phi1(deg)  Phi(deg)  phi2(deg)  f(g)[m.r.d.]'
           if degrees else
           '% phi1(rad)  Phi(rad)  phi2(rad)  f(g)[m.r.d.]')
    )

    data = np.column_stack([phi1, Phi, phi2, odf.f])
    np.savetxt(fname, data, header=header, comments='', fmt=fmt_str)
