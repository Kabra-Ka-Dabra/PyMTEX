import numpy as np
import pytest
from pymtex.geometry.symmetry import CrystalSymmetry


class TestGroupOrder:
    """Each point group must produce the correct number of elements."""

    @pytest.mark.parametrize("name,expected", [
        ('1', 1), ('-1', 2),
        ('2', 2), ('m', 2), ('2/m', 4),
        ('222', 4), ('mm2', 4), ('mmm', 8),
        ('4', 4), ('-4', 4), ('4/m', 8),
        ('422', 8), ('4mm', 8), ('-42m', 8), ('4/mmm', 16),
        ('3', 3), ('-3', 6), ('32', 6), ('3m', 6), ('-3m', 12),
        ('6', 6), ('-6', 6), ('6/m', 12),
        ('622', 12), ('6mm', 12), ('-6m2', 12), ('6/mmm', 24),
        ('23', 12), ('m-3', 24), ('432', 24), ('-43m', 24), ('m-3m', 48),
    ])
    def test_nfold(self, name, expected):
        cs = CrystalSymmetry(name)
        assert cs.nfold == expected, (
            f"{name}: expected {expected} ops, got {cs.nfold}"
        )


class TestProperCounts:
    """Proper rotation count = nfold / 2 for Laue groups, nfold for proper groups."""

    @pytest.mark.parametrize("name,n_proper", [
        ('m-3m', 24), ('6/mmm', 12), ('4/mmm', 8), ('mmm', 4),
        ('-3m', 6), ('432', 24), ('422', 8), ('622', 12),
        ('m-3', 12), ('-3', 3),
        # purely proper groups: all ops are proper
        ('1', 1), ('2', 2), ('222', 4), ('3', 3), ('32', 6),
        ('4', 4), ('422', 8), ('6', 6), ('622', 12), ('23', 12), ('432', 24),
    ])
    def test_proper_count(self, name, n_proper):
        cs = CrystalSymmetry(name)
        assert cs.numProper == n_proper


class TestAliases:
    def test_cubic_alias(self):
        assert CrystalSymmetry('cubic').name == 'm-3m'

    def test_schoenflies(self):
        assert CrystalSymmetry('Oh').name == 'm-3m'
        assert CrystalSymmetry('D6h').name == '6/mmm'
        assert CrystalSymmetry('Td').name == '-43m'

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            CrystalSymmetry('xyz')


class TestProperties:
    def test_system(self):
        assert CrystalSymmetry('m-3m').system == 'cubic'
        assert CrystalSymmetry('6/mmm').system == 'hexagonal'
        assert CrystalSymmetry('-3m').system == 'trigonal'
        assert CrystalSymmetry('4/mmm').system == 'tetragonal'
        assert CrystalSymmetry('mmm').system == 'orthorhombic'
        assert CrystalSymmetry('2/m').system == 'monoclinic'
        assert CrystalSymmetry('-1').system == 'triclinic'

    def test_laue(self):
        assert CrystalSymmetry('432').Laue.name == 'm-3m'
        assert CrystalSymmetry('32').Laue.name == '-3m'
        assert CrystalSymmetry('4').Laue.name == '4/mmm'

    def test_is_laue(self):
        assert CrystalSymmetry('m-3m').isLaue
        assert CrystalSymmetry('6/mmm').isLaue
        assert not CrystalSymmetry('432').isLaue

    def test_is_proper(self):
        assert CrystalSymmetry('432').isProper
        assert CrystalSymmetry('23').isProper
        assert not CrystalSymmetry('-43m').isProper
        assert not CrystalSymmetry('m-3m').isProper


class TestRotationOperators:
    def test_rot_are_unit_quaternions(self):
        cs = CrystalSymmetry('m-3m')
        rot = cs.rot
        norms = np.sqrt(rot.a**2 + rot.b**2 + rot.c**2 + rot.d**2)
        assert np.allclose(norms, 1.0, atol=1e-8)

    def test_proper_rotations_det_one(self):
        cs = CrystalSymmetry('m-3m')
        mats = cs.properRotations.to_matrix()
        dets = np.linalg.det(mats)
        assert np.allclose(dets, 1.0, atol=1e-8)

    def test_group_is_closed(self):
        """For every pair (a,b) in G, a*b should also be in G."""
        cs = CrystalSymmetry('432')
        rot = cs.rot
        # Compare via dot_outer: |dot(q_i, q_j)| ≈ 1 means same rotation
        D = rot.dot_outer(rot)   # (24, 24) – self-products
        # For each product r*s, find it in rot
        for i in range(cs.nfold):
            for j in range(cs.nfold):
                q_prod = rot[i] * rot[j]
                dots = np.abs(q_prod.dot(rot))
                assert np.any(dots > 1 - 1e-6), (
                    f"Product of ops {i},{j} not found in group"
                )
