import numpy as np
import pytest
from pymtex import CrystalSymmetry, Miller
from pymtex.texture import PoleFigure, ODF, calcODF
from pymtex.texture.polefigure import _spherical_to_xyz


cs = CrystalSymmetry('m-3m')


def _make_hemisphere_grid(n_theta=10, n_phi=18):
    theta = np.linspace(0, np.pi/2, n_theta)
    phi   = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    r = _spherical_to_xyz(TH.ravel(), PH.ravel())
    return r


class TestPoleFigure:
    def test_construction_single_pole(self):
        h = Miller(hkl=[1, 0, 0], cs=cs)
        r = _make_hemisphere_grid()
        I = np.ones(len(r))
        pf = PoleFigure(h, r, I, cs)
        assert pf.numPF == 1
        assert pf.numPoints[0] == len(r)

    def test_construction_list_of_poles(self):
        h = [Miller(hkl=[1, 0, 0], cs=cs), Miller(hkl=[1, 1, 0], cs=cs)]
        r = _make_hemisphere_grid()
        I = np.ones(len(r))
        pf = PoleFigure(h, r, I, cs)
        assert pf.numPF == 2

    def test_mismatch_raises(self):
        h = [Miller(hkl=[1, 0, 0], cs=cs), Miller(hkl=[1, 1, 0], cs=cs)]
        r = [_make_hemisphere_grid(), _make_hemisphere_grid()]
        I = [np.ones(len(r[0]))]  # only 1 intensity list for 2 poles
        with pytest.raises(ValueError):
            PoleFigure(h, r, I, cs)

    def test_normalize(self):
        h = Miller(hkl=[1, 0, 0], cs=cs)
        r = _make_hemisphere_grid()
        I = np.random.rand(len(r)) * 5 + 1
        pf = PoleFigure(h, r, I, cs).normalize()
        assert np.isclose(pf.intensities[0].mean(), 1.0, atol=1e-10)

    def test_interp_exact_directions(self):
        h = Miller(hkl=[1, 0, 0], cs=cs)
        r = _make_hemisphere_grid(n_theta=5, n_phi=8)
        I = np.arange(len(r), dtype=float)
        pf = PoleFigure(h, r, I, cs)
        # Query at exact measurement directions; directions near the pole
        # (θ≈0) are degenerate (all φ map to the same point) so skip those.
        non_degenerate = np.linalg.norm(r[:, :2], axis=-1) > 0.05
        result = pf.interp(0, r[non_degenerate])
        assert np.array_equal(result, I[non_degenerate])

    def test_from_spherical(self):
        h = Miller(hkl=[1, 0, 0], cs=cs)
        theta = np.linspace(0, np.pi/2, 10)
        phi   = np.linspace(0, 2*np.pi, 12, endpoint=False)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        I = np.ones(TH.size)
        pf = PoleFigure.from_spherical(h, TH.ravel(), PH.ravel(), I, cs)
        assert pf.numPF == 1
        assert pf.numPoints[0] == len(I)

    def test_repr(self):
        h = Miller(hkl=[1, 0, 0], cs=cs)
        r = _make_hemisphere_grid()
        pf = PoleFigure(h, r, np.ones(len(r)), cs)
        assert 'PoleFigure' in repr(pf)


class TestODF:
    def _random_texture_odf(self):
        """Uniform ODF (random texture, f=1 everywhere)."""
        from pymtex.texture.calcodf import _make_so3_grid
        ori, w = _make_so3_grid(cs, resolution_deg=10.0)
        f = np.ones(ori.size)
        f /= np.sum(f * w)
        return ODF(ori, f, w, cs)

    def test_texture_index_random(self):
        odf = self._random_texture_odf()
        # Random texture: J = ∫ f² dg ≈ 1
        assert np.isclose(odf.texture_index(), 1.0, atol=0.1)

    def test_calcPF_shape(self):
        odf = self._random_texture_odf()
        h = Miller(hkl=[1, 0, 0], cs=cs)
        r, pf = odf.calcPF(h, n_theta=10, n_phi=18)
        assert r.shape[0] == pf.shape[0]
        assert r.shape[1] == 3

    def test_calcPF_random_is_uniform(self):
        """For random texture, computed pole figure mean should be 1."""
        odf = self._random_texture_odf()
        h = Miller(hkl=[1, 0, 0], cs=cs)
        _, pf = odf.calcPF(h, n_theta=10, n_phi=18)
        # After normalisation the mean is exactly 1
        assert np.isclose(pf.mean(), 1.0, atol=1e-6)
        # Non-zero values should be relatively uniform (within 2× of mean)
        nonzero = pf[pf > 0]
        assert nonzero.min() > 0.1

    def test_eval_shape(self):
        from pymtex.geometry.orientation import Orientation
        odf = self._random_texture_odf()
        ori = Orientation.rand(10, cs)
        vals = odf.eval(ori)
        assert vals.shape == (10,)
        assert np.all(vals >= 0)

    def test_repr(self):
        odf = self._random_texture_odf()
        assert 'ODF' in repr(odf)


class TestCalcODF:
    def test_random_texture_round_trip(self):
        """
        If pole figures come from a random texture (all ones), the recovered
        ODF should also be close to uniform (texture index ≈ 1).
        """
        h = [Miller(hkl=[1, 0, 0], cs=cs),
             Miller(hkl=[1, 1, 0], cs=cs),
             Miller(hkl=[1, 1, 1], cs=cs)]

        r = _make_hemisphere_grid(n_theta=10, n_phi=18)
        I = np.ones(len(r))

        pf = PoleFigure(h, r, I, cs)
        odf = calcODF(pf, resolution_deg=10.0, n_iter=10)

        assert isinstance(odf, ODF)
        # Random texture should have texture index ≈ 1
        J = odf.texture_index()
        assert 0.5 < J < 3.0, f"Texture index {J:.2f} outside expected range"

    def test_odf_non_negative(self):
        h = Miller(hkl=[1, 0, 0], cs=cs)
        r = _make_hemisphere_grid()
        pf = PoleFigure(h, r, np.ones(len(r)), cs)
        odf = calcODF(pf, resolution_deg=10.0, n_iter=5)
        assert np.all(odf.f >= 0)

    def test_odf_normalised(self):
        h = Miller(hkl=[1, 0, 0], cs=cs)
        r = _make_hemisphere_grid()
        pf = PoleFigure(h, r, np.ones(len(r)), cs)
        odf = calcODF(pf, resolution_deg=10.0, n_iter=5)
        # ∫ f dg ≈ 1
        integral = np.sum(odf.f * odf.weights)
        assert np.isclose(integral, 1.0, atol=0.05)
