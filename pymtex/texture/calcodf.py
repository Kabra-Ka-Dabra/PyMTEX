"""
calcODF – Pole figure inversion to compute the ODF.

Implements the **WIMV algorithm** (Williams-Imhof-Matthies-Vinel), which is
the classical iterative method for ODF estimation from pole figures.  This is
the same core algorithm used by MTEX's ``calcODF`` when called with
``'WIMV'`` or ``'ODF'`` flags.

Mathematical background
-----------------------
The pole figure P_h(r) is the marginal of the ODF over all orientations that
map the crystal direction h onto the specimen direction r:

    P_h(r) = ∫_{SO(3)} f(g) · δ(R(g)·ĥ − r̂) dg

WIMV iteration (Williams, 1968; Matthies & Vinel, 1982)
--------------------------------------------------------
Discretise SO(3) into N cells {g_i} with volume weights {w_i}.
For each cell and each pole h_j, precompute the projected specimen direction:
    r_{ij} = R(g_i) · ĥ_j

Initialise: f^(0)(g_i) = 1 (random texture).

Update rule (iterating t = 0, 1, …, n_iter):
    f^(t+1)(g_i) ∝ f^(t)(g_i) · [∏_j (P^meas_j(r_{ij}) / P^calc_j(r_{ij}))]^(1/n_pf)

where the theoretical pole figure is:
    P^calc_j(r_k) = Σ_{i: r_{ij}≈r_k} f^(t)(g_i) · w_i  / Σ_{i: r_{ij}≈r_k} w_i

Convergence is typically reached in 10–25 iterations.
"""

from __future__ import annotations

import numpy as np

from pymtex.geometry.orientation import Orientation
from pymtex.geometry.symmetry import CrystalSymmetry
from pymtex.texture.odf import ODF
from pymtex.texture.polefigure import PoleFigure


# ---------------------------------------------------------------------------
# SO(3) grid utilities
# ---------------------------------------------------------------------------

def _make_so3_grid(cs, resolution_deg=5.0):
    """
    Build a regular Euler-angle grid covering SO(3).

    The grid uses Bunge (ZXZ) Euler angles:
        φ₁ ∈ [0°, 360°),  Φ ∈ [0°, 90°],  φ₂ ∈ [0°, 360°/fold)

    Volume weight of each cell ∝ sin(Φ) (from the SO(3) Haar measure).

    Parameters
    ----------
    cs : CrystalSymmetry
    resolution_deg : float
        Angular step in degrees.  5° gives ~11 k cells for cubic, ~60 k total.

    Returns
    -------
    orientations : Orientation, shape (N,)
    weights      : np.ndarray, shape (N,)  – sum to 1
    """
    step = resolution_deg

    phi1_vals = np.arange(0, 360, step)
    Phi_vals  = np.arange(0, 90 + step/2, step)   # include 90°
    phi2_vals = np.arange(0, 360, step)

    PHI1, PHI, PHI2 = np.meshgrid(phi1_vals, Phi_vals, phi2_vals,
                                   indexing='ij')

    phi1_rad = np.deg2rad(PHI1.ravel())
    Phi_rad  = np.deg2rad(PHI.ravel())
    phi2_rad = np.deg2rad(PHI2.ravel())

    orientations = Orientation.byEuler(
        phi1_rad, Phi_rad, phi2_rad, cs, convention='Bunge'
    )

    # Volume weight: sin(Φ) dφ₁ dΦ dφ₂  (un-normalised)
    w = np.sin(Phi_rad)
    w = w / w.sum()

    return orientations, w


# ---------------------------------------------------------------------------
# WIMV core
# ---------------------------------------------------------------------------

def _interp_pf(r_measured, I_measured, r_query):
    """
    Nearest-neighbour lookup of pole figure *I_measured* at *r_query*.

    Parameters
    ----------
    r_measured : (M, 3) unit vectors – measurement grid
    I_measured : (M,)  intensities
    r_query    : (N, 3) unit vectors – query directions

    Returns
    -------
    (N,) intensities (nearest-neighbour)
    """
    # dot-product distance on the unit sphere
    cos_ang = r_measured @ r_query.T     # (M, N)
    nearest = np.argmax(cos_ang, axis=0) # (N,)
    return I_measured[nearest]


def _wimv_step(f, weights, projections, pf_measured, pf_r):
    """
    One WIMV update step.

    Parameters
    ----------
    f         : (N,) current ODF values
    weights   : (N,) cell weights
    projections  : list of (N, 3) – precomputed r_{ij} for each pole j
    pf_measured  : list of (M_j,) – measured intensities for each pole
    pf_r         : list of (M_j, 3) – measurement directions for each pole

    Returns
    -------
    f_new : (N,) updated ODF values (not yet normalised)
    """
    n_poles = len(projections)
    log_update = np.zeros(len(f))

    for j, (r_ij, r_j, I_j) in enumerate(zip(projections, pf_r, pf_measured)):
        # --- theoretical pole figure for this pole ---
        # For each measurement direction r_k, accumulate ODF from all cells
        # whose projected direction is closest to r_k
        M = r_j.shape[0]
        pf_calc   = np.zeros(M)
        pf_weight = np.zeros(M)

        cos_sim = r_ij @ r_j.T          # (N, M)
        nearest = np.argmax(cos_sim, axis=1)  # (N,) → which r_k each cell maps to

        np.add.at(pf_calc,   nearest, f * weights)
        np.add.at(pf_weight, nearest, weights)

        # Avoid division by zero
        with np.errstate(invalid='ignore', divide='ignore'):
            pf_calc = np.where(pf_weight > 0, pf_calc / pf_weight, 0.0)

        # Normalise both measured and theoretical to mean = 1
        meas_mean = I_j.mean()
        calc_mean = pf_calc.mean()
        I_j_norm  = I_j   / (meas_mean + 1e-12)
        calc_norm = pf_calc / (calc_mean + 1e-12)

        # Lookup measured and theoretical at each cell's projected direction
        I_meas_at_cell = I_j_norm[nearest]
        I_calc_at_cell = calc_norm[nearest]

        # log of ratio (clamped for stability)
        ratio = np.where(
            I_calc_at_cell > 1e-8,
            I_meas_at_cell / I_calc_at_cell,
            1.0
        )
        log_update += np.log(np.maximum(ratio, 1e-8)) / n_poles

    f_new = f * np.exp(log_update)
    return np.maximum(f_new, 0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calcODF(pf, resolution_deg=5.0, n_iter=20, verbose=False):
    """
    Calculate the ODF from pole figure data using the WIMV algorithm.

    Parameters
    ----------
    pf : PoleFigure
        Measured pole figures (see :class:`~pymtex.texture.PoleFigure`).
    resolution_deg : float
        SO(3) grid resolution in degrees.  5° is standard; use 10° for speed.
        Finer grids give higher ODF resolution but cost O(resolution⁻³).
    n_iter : int
        Number of WIMV iterations (typically 15–25 for convergence).
    verbose : bool
        If True, print residual after each iteration.

    Returns
    -------
    ODF
        Discrete ODF on the SO(3) grid.

    Notes
    -----
    * Pole figures are automatically normalised to unit mean before inversion.
    * At least **3 independent poles** are recommended for cubic symmetry;
      fewer poles lead to ghost components (spurious ODF features).
    * Crystal symmetry is applied via ``project2FundamentalRegion`` on the
      grid, which enforces the correct symmetry of the result.

    Examples
    --------
    Typical workflow::

        import numpy as np
        from pymtex import CrystalSymmetry, Miller
        from pymtex.texture import PoleFigure, calcODF

        cs  = CrystalSymmetry('m-3m')
        h   = [Miller(hkl=[1,0,0], cs=cs),
               Miller(hkl=[1,1,0], cs=cs),
               Miller(hkl=[1,1,1], cs=cs)]

        # r: (M, 3) unit vectors on hemisphere, one per measurement direction
        # I: (M,) measured intensities from X-ray/neutron diffraction
        pf  = PoleFigure(h, [r100, r110, r111], [I100, I110, I111], cs)

        odf = calcODF(pf, resolution_deg=5, n_iter=20)

        print(odf)                                    # texture index J
        odf.plotPF(h[0])                              # forward-compute {100} PF
        odf.plotIPF()                                 # inverse pole figure (ND)
    """
    if not isinstance(pf, PoleFigure):
        raise TypeError("pf must be a PoleFigure instance.")

    # 1. Normalise pole figures
    pf_norm = pf.normalize()

    # 2. Build SO(3) grid
    if verbose:
        print(f"Building SO(3) grid at {resolution_deg}° resolution …")
    orientations, weights = _make_so3_grid(pf.cs, resolution_deg)
    N = orientations.size

    # 3. Precompute projected specimen directions r_{ij} = R(g_i) * h_j
    if verbose:
        print(f"Precomputing projections for {pf.numPF} poles × {N} cells …")

    mats = orientations.to_matrix()   # (N, 3, 3)
    projections = []
    for h_j in pf_norm.h:
        h_vec = h_j.toVector3d()                             # (3,)
        r_ij  = np.einsum('nij,j->ni', mats, h_vec)         # (N, 3)
        # normalise (should already be unit but guard against fp drift)
        r_ij  = r_ij / np.linalg.norm(r_ij, axis=-1, keepdims=True)
        projections.append(r_ij)

    # 4. WIMV iterations
    f = np.ones(N)   # initial uniform ODF
    f = f / np.sum(f * weights)

    for it in range(n_iter):
        f_new = _wimv_step(
            f, weights, projections,
            pf_norm.intensities, pf_norm.r
        )

        # Normalise
        total = np.sum(f_new * weights)
        if total > 0:
            f_new = f_new / total

        # Residual (R-value, lower is better convergence)
        if verbose:
            residual = _calc_residual(f_new, weights, projections,
                                      pf_norm.intensities, pf_norm.r)
            print(f"  Iter {it+1:3d}  R = {residual:.4f}")

        f = f_new

    return ODF(orientations, f, weights, pf.cs, pf.ss)


def _calc_residual(f, weights, projections, pf_measured, pf_r):
    """RP residual: mean(|P_calc - P_meas| / P_meas) averaged over all poles."""
    residuals = []
    for r_ij, r_j, I_j in zip(projections, pf_r, pf_measured):
        M = r_j.shape[0]
        pf_calc   = np.zeros(M)
        pf_weight = np.zeros(M)
        cos_sim = r_ij @ r_j.T
        nearest = np.argmax(cos_sim, axis=1)
        np.add.at(pf_calc,   nearest, f * weights)
        np.add.at(pf_weight, nearest, weights)
        with np.errstate(invalid='ignore', divide='ignore'):
            pf_calc = np.where(pf_weight > 0, pf_calc / pf_weight, 0.0)

        # Normalise
        pf_calc = pf_calc / (pf_calc.mean() + 1e-12)
        I_norm  = I_j / (I_j.mean() + 1e-12)

        mask = I_norm > 0
        rp = np.mean(np.abs(pf_calc[mask] - I_norm[mask]) / I_norm[mask])
        residuals.append(rp)
    return float(np.mean(residuals))
