"""
Tests for pole figure file I/O (pymtex/texture/io.py).
Uses synthetic in-memory files via tmp_path fixture.
"""

import os
import textwrap
import numpy as np
import pytest

from pymtex import CrystalSymmetry, Miller
from pymtex.texture import loadPoleFigure, PoleFigure


cs = CrystalSymmetry('m-3m')
h100 = Miller(hkl=[1, 0, 0], cs=cs)
h110 = Miller(hkl=[1, 1, 0], cs=cs)
h111 = Miller(hkl=[1, 1, 1], cs=cs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(textwrap.dedent(content))
    return str(p)


def _assert_valid_pf(pf, n_poles=1, min_points=3):
    assert isinstance(pf, PoleFigure)
    assert pf.numPF == n_poles
    for I in pf.intensities:
        assert len(I) >= min_points
        assert np.all(I >= 0)
    for r in pf.r:
        norms = np.linalg.norm(r, axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Generic column format
# ---------------------------------------------------------------------------

class TestGenericFormat:

    def test_basic_three_columns(self, tmp_path):
        content = """\
            # alpha(deg) beta(deg) intensity
            0.0   0.0   1.00
            0.0  90.0   1.10
            0.0 180.0   0.95
            0.0 270.0   1.05
           30.0   0.0   1.20
           30.0  90.0   1.15
           60.0   0.0   0.90
           90.0   0.0   0.80
        """
        f = _write(tmp_path, 'pf.txt', content)
        pf = loadPoleFigure(f, h100, cs)
        _assert_valid_pf(pf, n_poles=1, min_points=8)
        assert np.isclose(pf.intensities[0][0], 1.00)

    def test_comment_skip(self, tmp_path):
        content = """\
            # This is a comment
            # Another comment
            5.0   0.0   2.5
            5.0  30.0   2.3
            5.0  60.0   2.1
            10.0   0.0   1.9
        """
        f = _write(tmp_path, 'pf_comments.txt', content)
        pf = loadPoleFigure(f, h100, cs)
        _assert_valid_pf(pf)

    def test_reordered_columns(self, tmp_path):
        # File order: beta, intensity, alpha
        content = """\
              0.0   1.00   0.0
             90.0   1.10   0.0
            180.0   0.95   0.0
              0.0   1.20  30.0
             90.0   1.15  30.0
        """
        f = _write(tmp_path, 'pf_reorder.txt', content)
        pf = loadPoleFigure(f, h100, cs,
                            alpha_col=2, beta_col=0, intensity_col=1)
        _assert_valid_pf(pf)
        assert np.isclose(pf.intensities[0][0], 1.00)

    def test_tab_delimiter(self, tmp_path):
        content = "alpha\tbeta\tintensity\n" + \
                  "".join(f"{a}\t{b}\t1.0\n"
                          for a in [0, 30, 60]
                          for b in [0, 90, 180, 270])
        f = _write(tmp_path, 'pf_tabs.txt', content)
        pf = loadPoleFigure(f, h100, cs, delimiter='\t')
        _assert_valid_pf(pf, min_points=12)

    def test_radians_flag(self, tmp_path):
        # Same directions as 0°, 0°  and 90°, 0° – but in radians
        content = """\
            0.0        0.0       1.00
            1.5707963  0.0       1.10
            0.5235988  1.5707963 1.05
        """
        f = _write(tmp_path, 'pf_rad.txt', content)
        pf = loadPoleFigure(f, h100, cs, degrees=False)
        _assert_valid_pf(pf, min_points=3)
        # alpha=π/2 → r_z ≈ 0
        assert abs(pf.r[0][1, 2]) < 0.01

    def test_multiple_files(self, tmp_path):
        row = "5.0  0.0  1.0\n10.0  0.0  1.0\n15.0  0.0  1.0\n"
        f1 = _write(tmp_path, 'p1.txt', row)
        f2 = _write(tmp_path, 'p2.txt', row)
        pf = loadPoleFigure([f1, f2], [h100, h110], cs)
        _assert_valid_pf(pf, n_poles=2)

    def test_bad_column_index_raises(self, tmp_path):
        content = "5.0  0.0  1.0\n10.0  0.0  1.0\n"
        f = _write(tmp_path, 'pf_bad.txt', content)
        with pytest.raises(ValueError, match='column'):
            loadPoleFigure(f, h100, cs, intensity_col=5)

    def test_auto_detects_generic(self, tmp_path):
        content = "5.0 0.0 1.0\n10.0 0.0 1.1\n20.0 0.0 1.2\n"
        f = _write(tmp_path, 'data.dat', content)
        pf = loadPoleFigure(f, h100, cs, format='auto')
        _assert_valid_pf(pf)


# ---------------------------------------------------------------------------
# POPLA / EPF format
# ---------------------------------------------------------------------------

class TestPOPLAFormat:

    def _make_epf(self, tmp_path, name='cu_111.epf',
                  dtheta=5, mtheta=85, drho=5, mrho=355,
                  shifttheta=0, shiftrho=0, n_alpha=None, n_beta=None):
        """Generate a minimal valid POPLA EPF file with uniform intensity=1."""
        if n_alpha is None:
            n_alpha = int((mtheta - 0) / dtheta) + (1 if shifttheta else 0)
            alpha_vals = np.arange(dtheta/2 * (not shifttheta),
                                   mtheta + dtheta/4, dtheta)
            n_alpha = len(alpha_vals)
        if n_beta is None:
            beta_vals = np.arange(drho/2 * (not shiftrho),
                                  mrho, drho)
            n_beta = len(beta_vals)

        intensities = np.ones(n_alpha * n_beta)

        # Build parameter line: space-separated (our lenient parser handles this)
        # Format: hkl dtheta mtheta drho mrho shifttheta shiftrho ...
        param = (f"  111 {dtheta:5.1f} {mtheta:5.1f} {drho:5.1f} {mrho:5.1f}"
                 f" {shifttheta:1d} {shiftrho:1d}  1  1  1 1.000  0.000")

        # Data: 18 values per line
        data_str = ''
        for i, v in enumerate(intensities):
            data_str += f'{v:7.3f}'
            if (i + 1) % 18 == 0:
                data_str += '\n'
        if len(intensities) % 18 != 0:
            data_str += '\n'

        content = f"Cu (111) pole figure  sample\n{param}\n{data_str}"
        return _write(tmp_path, name, content)

    def test_basic_popla(self, tmp_path):
        f = self._make_epf(tmp_path)
        pf = loadPoleFigure(f, h111, cs, format='popla')
        _assert_valid_pf(pf)
        assert np.allclose(pf.intensities[0], 1.0)

    def test_auto_detect_epf_extension(self, tmp_path):
        f = self._make_epf(tmp_path, name='test.epf')
        pf = loadPoleFigure(f, h111, cs, format='auto')
        _assert_valid_pf(pf)

    def test_shifted_grid(self, tmp_path):
        f = self._make_epf(tmp_path, shifttheta=1, shiftrho=1)
        pf = loadPoleFigure(f, h111, cs, format='popla')
        _assert_valid_pf(pf)

    def test_uniform_intensity(self, tmp_path):
        f = self._make_epf(tmp_path)
        pf = loadPoleFigure(f, h111, cs, format='popla')
        assert np.allclose(pf.intensities[0], 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# BEARTEX format
# ---------------------------------------------------------------------------

class TestBEARTEXFormat:

    def _make_beartex(self, tmp_path, name='cu.bea'):
        """Generate a minimal valid BEARTEX file with uniform intensity=1."""
        # Standard BEARTEX grid: 18 alpha (0–85°, step 5°) × 72 beta (0–355°, step 5°)
        n_alpha, n_beta = 18, 72
        intensities = np.ones(n_alpha * n_beta)

        # Header (7 lines)
        header = (
            "beartex98\n"
            "Test sample\n"
            "Cu FCC\n"
            "Pole figure\n"
            "Experiment\n"
            "  3.615  3.615  3.615 90.00 90.00 90.00  7  1\n"
            "100       85  85    5  355    5 1.000\n"
        )

        # Data: 5-char wide per value
        data_str = ''
        for i, v in enumerate(intensities):
            data_str += f'{v:5.1f}'
            if (i + 1) % 16 == 0:
                data_str += '\n'
        if len(intensities) % 16 != 0:
            data_str += '\n'

        return _write(tmp_path, name, header + data_str)

    def test_basic_beartex(self, tmp_path):
        f = self._make_beartex(tmp_path)
        pf = loadPoleFigure(f, h100, cs, format='beartex')
        _assert_valid_pf(pf)

    def test_auto_detect_bea(self, tmp_path):
        f = self._make_beartex(tmp_path, name='test.bea')
        pf = loadPoleFigure(f, h100, cs, format='auto')
        _assert_valid_pf(pf)

    def test_uniform_intensity(self, tmp_path):
        f = self._make_beartex(tmp_path)
        pf = loadPoleFigure(f, h100, cs, format='beartex')
        assert np.allclose(pf.intensities[0], 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Matrix format
# ---------------------------------------------------------------------------

class TestMatrixFormat:

    def test_basic_matrix(self, tmp_path):
        # 4 alpha rows × 8 beta cols
        mat = np.ones((4, 8)) * 1.5
        content = '\n'.join(' '.join(f'{v:.3f}' for v in row) for row in mat)
        f = _write(tmp_path, 'mat.dat', content)
        pf = loadPoleFigure(f, h100, cs, format='matrix',
                            alpha_start=0, alpha_stop=15, alpha_step=5,
                            beta_start=0,  beta_stop=315, beta_step=45)
        _assert_valid_pf(pf, min_points=32)
        assert np.allclose(pf.intensities[0], 1.5)

    def test_matrix_with_comments(self, tmp_path):
        mat = np.ones((3, 6))
        content = '# This is a matrix file\n'
        content += '\n'.join(' '.join(f'{v:.2f}' for v in row) for row in mat)
        f = _write(tmp_path, 'mat_c.dat', content)
        pf = loadPoleFigure(f, h100, cs, format='matrix',
                            alpha_start=0, alpha_stop=10, alpha_step=5,
                            beta_start=0,  beta_stop=300, beta_step=60)
        _assert_valid_pf(pf, min_points=18)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:

    def test_mismatched_files_and_poles(self, tmp_path):
        row = "5.0  0.0  1.0\n10.0  0.0  1.0\n"
        f = _write(tmp_path, 'pf.txt', row)
        with pytest.raises(ValueError, match='match'):
            loadPoleFigure([f], [h100, h110], cs)

    def test_unknown_format_raises(self, tmp_path):
        f = _write(tmp_path, 'pf.txt', "1 2 3\n")
        with pytest.raises(ValueError, match='Unknown format'):
            loadPoleFigure(f, h100, cs, format='xyzformat')

    def test_empty_file_raises(self, tmp_path):
        f = _write(tmp_path, 'empty.txt', '# only comments\n# nothing numeric\n')
        with pytest.raises(ValueError):
            loadPoleFigure(f, h100, cs, format='generic')
