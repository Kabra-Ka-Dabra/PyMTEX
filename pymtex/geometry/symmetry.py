"""
CrystalSymmetry – port of MTEX's @symmetry / @crystalSymmetry.

Groups are generated from generators via closure, mirroring MTEX's calcQuat
which uses symAxis(axis, n) to build cyclic generator sets then closes under
multiplication.  Axes follow MTEX defaults: a=x, b=y, c=z.
"""

import numpy as np
from pymtex.geometry.rotation import Rotation
from pymtex.geometry.quaternion import Quaternion


# ---------------------------------------------------------------------------
# Low-level rotation helpers
# ---------------------------------------------------------------------------

def _Rn(axis, angle):
    """Rotation matrix by *angle* (rad) about unit *axis* (Rodrigues formula)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])


def _Rx(a): return _Rn([1, 0, 0], a)
def _Ry(a): return _Rn([0, 1, 0], a)
def _Rz(a): return _Rn([0, 0, 1], a)

_PI = np.pi


def _sym_axis(axis, n):
    """All *n* proper rotations of the cyclic group C_n about *axis*."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return [_Rn(axis, 2.0 * _PI * k / n) for k in range(n)]


# ---------------------------------------------------------------------------
# Group closure
# ---------------------------------------------------------------------------

def _mats_equal(A, B, tol=1e-9):
    return np.allclose(A, B, atol=tol)


def _group_closure(proper_gens, improper_gen=None, max_order=96):
    """Build the full point group by closure.

    Parameters
    ----------
    proper_gens : list of (3,3) arrays
        Generators of proper rotations.
    improper_gen : (3,3) array or None
        Rotation-matrix part of one improper generator.  The operation acts
        on a vector as -(M @ v).  If given, the coset g*H is appended.
    max_order : int
        Safety limit on group size.

    Returns
    -------
    list of (ndarray (3,3), bool)
        Each tuple = (rotation_matrix, is_improper).
    """
    # Seed with identity
    elements = [(np.eye(3), False)]

    all_gens = [(g, False) for g in proper_gens]
    if improper_gen is not None:
        all_gens.append((improper_gen, True))

    for g, flag in all_gens:
        if not any(_mats_equal(g, e) and flag == f for e, f in elements):
            elements.append((g, flag))

    changed = True
    while changed and len(elements) < max_order:
        changed = False
        new = []
        for A, fa in elements:
            for B, fb in elements:
                C, fc = A @ B, fa ^ fb
                if not any(_mats_equal(C, e) and fc == f
                           for e, f in elements + new):
                    new.append((C, fc))
                    changed = True
        elements.extend(new)

    return elements


def _to_rotation(elements):
    """Convert list of (matrix, improper) to a Rotation array."""
    mats = np.array([m for m, _ in elements])
    imps = np.array([f for _, f in elements], dtype=bool)
    q = Quaternion.from_matrix(mats)
    return Rotation(q.a, q.b, q.c, q.d, improper=imps)


# ---------------------------------------------------------------------------
# CrystalSymmetry
# ---------------------------------------------------------------------------

class CrystalSymmetry:
    """
    Crystallographic point group symmetry.

    Parameters
    ----------
    name : str
        Point group in Hermann-Mauguin notation, e.g. ``'m-3m'``, ``'6/mmm'``.
        Common aliases (``'cubic'``, ``'hexagonal'``, …) and Schoenflies
        symbols (``'Oh'``, ``'D4h'``, …) are also accepted.

    Examples
    --------
    >>> cs = CrystalSymmetry('m-3m')
    >>> cs.nfold
    48
    >>> cs.system
    'cubic'
    >>> cs.rot          # Rotation array of all 48 operators
    """

    # --- name aliases -------------------------------------------------------
    _ALIASES = {
        # crystal systems
        'cubic': 'm-3m', 'hexagonal': '6/mmm', 'tetragonal': '4/mmm',
        'orthorhombic': 'mmm', 'monoclinic': '2/m',
        'trigonal': '-3m', 'triclinic': '-1',
        # Schoenflies → HM
        'Oh': 'm-3m', 'O': '432', 'Td': '-43m', 'Th': 'm-3', 'T': '23',
        'D6h': '6/mmm', 'D6': '622', 'C6v': '6mm', 'D3h': '-6m2',
        'C6h': '6/m',  'C6': '6',   'C3h': '-6',
        'D4h': '4/mmm', 'D4': '422', 'C4v': '4mm', 'D2d': '-42m',
        'C4h': '4/m',  'C4': '4',   'S4': '-4',
        'D3d': '-3m',  'D3': '32',  'C3v': '3m', 'S6': '-3', 'C3': '3',
        'D2h': 'mmm',  'D2': '222', 'C2v': 'mm2',
        'C2h': '2/m',  'C2': '2',   'Cs': 'm',
        'Ci': '-1', 'C1': '1',
    }

    # Expected group orders (used in repr and validation)
    _ORDER = {
        '1': 1, '-1': 2,
        '2': 2, 'm': 2, '2/m': 4,
        '222': 4, 'mm2': 4, 'mmm': 8,
        '4': 4, '-4': 4, '4/m': 8, '422': 8, '4mm': 8, '-42m': 8, '4/mmm': 16,
        '3': 3, '-3': 6, '32': 6, '3m': 6, '-3m': 12,
        '6': 6, '-6': 6, '6/m': 12, '622': 12, '6mm': 12, '-6m2': 12, '6/mmm': 24,
        '23': 12, 'm-3': 24, '432': 24, '-43m': 24, 'm-3m': 48,
    }

    def __init__(self, name='1'):
        canon = self._ALIASES.get(name, name)
        if canon not in self._ORDER:
            raise ValueError(
                f"Unknown point group {name!r}.  "
                f"Use Hermann-Mauguin notation, e.g. 'm-3m', '6/mmm', '222'."
            )
        self.name = canon
        self._elements = self._build_group(canon)

    # --- public properties --------------------------------------------------

    @property
    def rot(self):
        """All symmetry operators as a :class:`Rotation` array."""
        return _to_rotation(self._elements)

    @property
    def nfold(self):
        """Number of symmetry elements (group order)."""
        return len(self._elements)

    @property
    def properRotations(self):
        """Proper (det = +1) symmetry operators as a :class:`Rotation` array."""
        proper = [(m, f) for m, f in self._elements if not f]
        return _to_rotation(proper)

    @property
    def numProper(self):
        """Number of proper symmetry elements."""
        return sum(1 for _, f in self._elements if not f)

    @property
    def system(self):
        """Crystal system name as a string."""
        _sys = {
            'cubic':        {'23', 'm-3', '432', '-43m', 'm-3m'},
            'hexagonal':    {'6', '-6', '6/m', '622', '6mm', '-6m2', '6/mmm'},
            'trigonal':     {'3', '-3', '32', '3m', '-3m'},
            'tetragonal':   {'4', '-4', '4/m', '422', '4mm', '-42m', '4/mmm'},
            'orthorhombic': {'222', 'mm2', 'mmm'},
            'monoclinic':   {'2', 'm', '2/m'},
        }
        for sys, pgs in _sys.items():
            if self.name in pgs:
                return sys
        return 'triclinic'

    @property
    def Laue(self):
        """Return the Laue (centrosymmetric) group containing this group."""
        _laue = {
            '1': '-1',   '-1': '-1',
            '2': '2/m',  'm': '2/m',  '2/m': '2/m',
            '222': 'mmm', 'mm2': 'mmm', 'mmm': 'mmm',
            '4': '4/mmm', '-4': '4/mmm', '4/m': '4/mmm',
            '422': '4/mmm', '4mm': '4/mmm', '-42m': '4/mmm', '4/mmm': '4/mmm',
            '3': '-3',   '-3': '-3',
            '32': '-3m', '3m': '-3m', '-3m': '-3m',
            '6': '6/mmm', '-6': '6/mmm', '6/m': '6/mmm',
            '622': '6/mmm', '6mm': '6/mmm', '-6m2': '6/mmm', '6/mmm': '6/mmm',
            '23': 'm-3',  'm-3': 'm-3',
            '432': 'm-3m', '-43m': 'm-3m', 'm-3m': 'm-3m',
        }
        return CrystalSymmetry(_laue[self.name])

    @property
    def isLaue(self):
        """True if this group is a Laue (centrosymmetric) group."""
        return self.Laue.name == self.name

    @property
    def isProper(self):
        """True if all elements are proper rotations."""
        return all(not f for _, f in self._elements)

    def __repr__(self):
        return (f"CrystalSymmetry('{self.name}', "
                f"{self.nfold} ops, {self.system})")

    def __eq__(self, other):
        if not isinstance(other, CrystalSymmetry):
            return NotImplemented
        return self.name == other.name

    # --- symmetry element generation ----------------------------------------

    @staticmethod
    def _build_group(name):
        """
        Generate all symmetry elements for *name* following MTEX conventions.

        MTEX's calcQuat uses symAxis(axis, n) to build cyclic generator sets,
        then closes under multiplication, then applies inversion/improper.
        We follow the same logic using a closure algorithm.

        Standard axes: a = x, b = y, c = z (MTEX default).
        """
        pi = _PI

        # Standard axes (MTEX default: a=x, b=y, c=z)
        _a   = np.array([1.0, 0.0, 0.0])
        _b   = np.array([0.0, 1.0, 0.0])
        _c   = np.array([0.0, 0.0, 1.0])
        _ll0 = np.array([1.0, 1.0, 0.0])   # [110]
        _lll = np.array([1.0, 1.0, 1.0])   # [111]
        _m   = _a - _b                       # [1,-1,0] (MTEX: m = a1 - a2)

        # Commonly used rotation matrices
        C2a  = _Rn(_a, pi);   C2b  = _Rn(_b, pi);   C2c  = _Rn(_c, pi)
        C3c  = _Rn(_c, 2*pi/3)
        C4c  = _Rn(_c, pi/2)
        C6c  = _Rn(_c, pi/3)
        C3lll = _Rn(_lll, 2*pi/3)
        C2ll0 = _Rn(_ll0, pi)
        C2m   = _Rn(_m, pi)         # 2-fold about [1,-1,0]
        C4c3  = _Rn(_c, 3*pi/2)     # = C4c^{-1}
        I     = np.eye(3)

        # Dispatch: (proper_generators, improper_gen_or_None)
        # improper_gen M means the operation acts as -(M @ v).
        # np.eye(3) as improper_gen = inversion centre.
        tbl = {
            # triclinic
            '1':     ([], None),
            '-1':    ([], I),
            # monoclinic  (unique axis = c = z, matching MTEX default)
            '2':     ([C2c], None),
            'm':     ([], C2b),        # σ_xz (mirror perp to b)
            '2/m':   ([C2c], I),
            # orthorhombic
            '222':   ([C2a, C2c], None),
            'mm2':   ([C2c], C2b),
            'mmm':   ([C2a, C2c], I),
            # tetragonal
            '4':     ([C4c], None),
            '-4':    ([], C4c),         # -4 roto-inversion
            '4/m':   ([C4c], I),
            '422':   ([C4c, C2a], None),
            '4mm':   ([C4c], C2a),
            '-42m':  ([C2c, C2a], C4c3),
            '4/mmm': ([C4c, C2a], I),
            # trigonal  (321 setting: C2 along a1 = x)
            '3':     ([C3c], None),
            '-3':    ([C3c], I),
            '32':    ([C3c, C2a], None),
            '3m':    ([C3c], C2a),
            '-3m':   ([C3c, C2a], I),
            # hexagonal
            '6':     ([C6c], None),
            '-6':    ([C3c], C2c),      # C3h: C3 + σh (= improper C2c)
            '6/m':   ([C6c], I),
            '622':   ([C6c, C2a], None),
            '6mm':   ([C6c], C2a),
            '-6m2':  ([C3c, C2a], C2c),
            '6/mmm': ([C6c, C2a], I),
            # cubic
            '23':    ([C3lll, C2a, C2c], None),
            'm-3':   ([C3lll, C2a, C2c], I),
            '432':   ([C4c, C3lll, C2ll0], None),
            '-43m':  ([C3lll, C2a, C2c], C2m),   # T proper + σd coset
            'm-3m':  ([C4c, C3lll], I),
        }
        proper_gens, imp_gen = tbl[name]
        return _group_closure(proper_gens, imp_gen)
