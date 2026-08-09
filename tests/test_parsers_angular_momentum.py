"""Unit tests for elkpy.parsers.angular_momentum, on synthetic data -- no
Elk run needed (mirrors tests/test_parsers_spin.py's role: pin the
Python-transcribed lopzflm.f90 convention against known algebraic
identities before trusting the Fortran path that reuses the real lopzflm).

Two things are pinned here, both on the FULL, untruncated (2l+1)-dimensional
analytic matrices (not a band-window-truncated diagonalisation output --
see docs/design.md #19 for why the identities below do NOT hold as matrix
products on elkpy_angmomproj's actual nst x nst return values):

1. Hermiticity of Lx, Ly, Lz for l=0,1,2,3.
2. The su(2) commutation relation [Lx, Ly] = i*Lz (and cyclic) and the
   Casimir Lx^2+Ly^2+Lz^2 = l(l+1)*1 -- standard angular-momentum algebra,
   here specifically discriminating against the one bug class Hermiticity
   structurally cannot catch: Ly is purely imaginary off-diagonal (unlike
   the real Lx/Lz), so transposing L_a's role (bra/ket) sends Ly -> -Ly
   while leaving Lx/Lz untouched and staying perfectly Hermitian -- the
   commutator's sign pins Ly against Lz.
"""

import numpy as np
import pytest

from elkpy.parsers.angular_momentum import angular_momentum_matrices


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_hermitian(l):
    lx, ly, lz = angular_momentum_matrices(l)
    for name, m in (("Lx", lx), ("Ly", ly), ("Lz", lz)):
        assert m == pytest.approx(m.conj().T, abs=1e-12), f"{name} not Hermitian at l={l}"


@pytest.mark.parametrize("l", [1, 2, 3])
def test_commutation_relations(l):
    lx, ly, lz = angular_momentum_matrices(l)
    assert lx @ ly - ly @ lx == pytest.approx(1j * lz, abs=1e-10)
    assert ly @ lz - lz @ ly == pytest.approx(1j * lx, abs=1e-10)
    assert lz @ lx - lx @ lz == pytest.approx(1j * ly, abs=1e-10)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_casimir(l):
    lx, ly, lz = angular_momentum_matrices(l)
    n = 2 * l + 1
    casimir = lx @ lx + ly @ ly + lz @ lz
    assert casimir == pytest.approx(l * (l + 1) * np.eye(n), abs=1e-10)


def test_l0_is_trivially_zero():
    lx, ly, lz = angular_momentum_matrices(0)
    assert lx.shape == (1, 1)
    assert lx == pytest.approx(np.zeros((1, 1)))
    assert ly == pytest.approx(np.zeros((1, 1)))
    assert lz == pytest.approx(np.zeros((1, 1)))


def test_lz_eigenvalues_are_m():
    for l in (1, 2, 3):
        _, _, lz = angular_momentum_matrices(l)
        assert np.diagonal(lz).real == pytest.approx(np.arange(-l, l + 1))
        assert np.diagonal(lz).imag == pytest.approx(np.zeros(2 * l + 1))


def test_ly_sign_is_not_hermiticity_invisible():
    """A regression pin for the one bug class Hermiticity cannot catch: if
    Ly's sign were flipped (e.g. a mislabelled Fortran output column), it
    would remain perfectly Hermitian but fail the commutator identity."""
    lx, ly, lz = angular_momentum_matrices(1)
    wrong_ly = -ly
    assert wrong_ly == pytest.approx(wrong_ly.conj().T, abs=1e-12)  # still Hermitian
    commutator = lx @ wrong_ly - wrong_ly @ lx
    assert commutator != pytest.approx(1j * lz, abs=1e-6)  # but the algebra breaks
