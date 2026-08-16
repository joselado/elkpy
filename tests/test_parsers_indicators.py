"""Unit tests for elkpy.parsers.indicators -- rotation eigenvalues and the
Benalcazar-Li-Hughes corner-charge indices -- on synthetic data, no Elk run.

The formulas are transcribed from arXiv:1809.02142 (PRB 99, 245151 (2019)),
Eqs. 3, 4 and 14; these tests pin the transcription against hand-computed
cases, so a real-binary disagreement later is attributable to the export
path rather than to the arithmetic here.
"""

import numpy as np
import pytest

from elkpy.parsers import indicators


def _operator_with_eigenvalues(p_list, order, seed=0):
    """A unitary matrix whose eigenvalues are the requested n-th roots of
    unity, in a RANDOM basis -- so the tests exercise the diagonalisation
    rather than a conveniently diagonal matrix. High-symmetry spectra are
    degenerate and the diagonaliser returns an arbitrary basis within each
    multiplet, which is exactly why the module never reads the diagonal."""
    rng = np.random.default_rng(seed)
    n = len(p_list)
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    q = q * (np.diagonal(r) / np.abs(np.diagonal(r)))
    vals = np.exp(2j * np.pi * (np.array(p_list) - 1) / order)
    return q @ np.diag(vals) @ q.conj().T


# --- eigenvalue extraction ---


@pytest.mark.parametrize("order", [2, 3, 4, 6])
def test_eigenvalues_recovered_from_a_random_basis(order):
    p = [1 + (i % order) for i in range(2 * order)]
    got = indicators.rotation_eigenvalues(_operator_with_eigenvalues(p, order), order)
    assert sorted(got) == sorted(p)


def test_diagonal_is_not_the_eigenvalues():
    """Why the module diagonalises rather than reading the diagonal: in a
    rotated basis the diagonal entries are convex combinations of the
    eigenvalues, so they lie strictly INSIDE the unit circle and are not
    roots of unity at all. (Binning them to the nearest root can coincide
    with the right answer by luck, which is why this asserts on the moduli
    rather than on the recovered labels.)"""
    p = [1, 2, 3]
    o = _operator_with_eigenvalues(p, 3)
    assert np.max(np.abs(np.diagonal(o))) < 0.99
    assert sorted(indicators.rotation_eigenvalues(o, 3)) == sorted(p)


def test_non_unitary_operator_raises():
    o = _operator_with_eigenvalues([1, 2, 3], 3)
    o[0, 1] += 0.4
    with pytest.raises(ValueError, match="not unitary"):
        indicators.rotation_eigenvalues(o, 3)


def test_eigenvalue_off_the_root_lattice_raises():
    """An eigenvalue that is not an n-th root of unity means the k-point is
    not actually fixed by the operation, or the window is not invariant."""
    o = np.diag([1.0 + 0j, np.exp(0.7j), np.exp(4j * np.pi / 3)])
    with pytest.raises(ValueError, match="roots of unity"):
        indicators.rotation_eigenvalues(o, 3)


def test_counts_sum_to_the_band_count():
    counts = indicators.eigenvalue_counts(_operator_with_eigenvalues([1, 1, 2, 3, 3, 3], 3), 3)
    assert list(counts) == [2, 1, 3]
    assert counts.sum() == 6


# --- the indices ---


def test_gamma_relative_to_itself_is_zero():
    """[Gamma_p] = 0 by construction -- a sanity check the real-binary path
    can assert for free."""
    c = np.array([3, 1, 2])
    assert list(indicators.relative_indices(c, c)) == [0, 0, 0]


def test_corner_charge_c3_matches_the_paper_formula():
    """BLH Eq. 14: Q^(3) = (e/3)[K_2^(3)] mod e. Only the SECOND index
    enters -- a transcription slip onto [K_1] would be invisible in any
    structural check, so it is pinned here."""
    assert indicators.corner_charge({"K": np.array([0, 1, -1])}, 3) == pytest.approx(1 / 3)
    assert indicators.corner_charge({"K": np.array([0, 2, -2])}, 3) == pytest.approx(2 / 3)
    assert indicators.corner_charge({"K": np.array([0, 3, -3])}, 3) == pytest.approx(0.0)
    # a nonzero [K_1] alone must NOT produce a charge
    assert indicators.corner_charge({"K": np.array([1, 0, -1])}, 3) == pytest.approx(0.0)


def test_corner_charge_c2_c4_c6_formulas():
    # Q^(2) = (e/4)(-[X_1] - [Y_1] + [M_1])
    q2 = indicators.corner_charge(
        {"X": np.array([1, 0]), "Y": np.array([0, 0]), "M": np.array([2, 0])}, 2)
    assert q2 == pytest.approx(((-1 - 0 + 2) / 4) % 1.0)
    # Q^(4) = (e/4)([X_1] + 2[M_1] + 3[M_2])
    q4 = indicators.corner_charge({"X": np.array([1, 0]), "M": np.array([1, 1, 0, 0])}, 4)
    assert q4 == pytest.approx(((1 + 2 * 1 + 3 * 1) / 4) % 1.0)
    # Q^(6) = (e/4)[M_1] + (e/6)[K_1]
    q6 = indicators.corner_charge({"M": np.array([1, 0]), "K": np.array([1, 0, 0])}, 6)
    assert q6 == pytest.approx((1 / 4 + 1 / 6) % 1.0)


def test_corner_charge_is_modulo_e():
    """Q is defined mod e, so it always lands in [0, 1) -- a large index set
    must wrap rather than accumulate."""
    q = indicators.corner_charge({"K": np.array([0, 7, -7])}, 3)
    assert 0.0 <= q < 1.0
    assert q == pytest.approx((7 / 3) % 1.0)


def test_unsupported_order_raises():
    with pytest.raises(ValueError, match="n = 2, 3, 4, 6"):
        indicators.corner_charge({"K": np.array([0, 1, 0, 0, 0])}, 5)


# --- picking the operation out of the crystal's symmetry list ---


def _op(rot, symmorphic=True, isym=1):
    return {"isym": isym, "rotation": np.array(rot), "symmorphic": symmorphic,
            "translation": np.zeros(3)}


C3_HEX = [[0, -1, 0], [1, -1, 0], [0, 0, 1]]     # 3-fold on a hexagonal lattice
C2_Z = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]


def test_find_rotation_picks_the_right_order():
    ops = [_op(np.eye(3, dtype=int), isym=1), _op(C2_Z, isym=2), _op(C3_HEX, isym=3)]
    assert indicators.find_rotation(ops, 3)["isym"] == 3
    assert indicators.find_rotation(ops, 2)["isym"] == 2


def test_find_rotation_skips_nonsymmorphic():
    """A glide/screw is refused by the Fortran side, so it must not be
    selected here either."""
    ops = [_op(C3_HEX, symmorphic=False, isym=5)]
    with pytest.raises(ValueError, match="no symmorphic"):
        indicators.find_rotation(ops, 3)


def test_fixes_kpoint_uses_the_transpose():
    """The rotation acts on reciprocal fractional coordinates by the
    TRANSPOSE of the real-space lattice matrix. K = (1/3, 1/3, 0) is fixed
    by the hexagonal C_3 modulo a reciprocal lattice vector; a generic point
    is not."""
    op = _op(C3_HEX)
    assert indicators.fixes_kpoint(op, (0.0, 0.0, 0.0))
    assert indicators.fixes_kpoint(op, (1 / 3, 1 / 3, 0.0))
    assert not indicators.fixes_kpoint(op, (0.21, 0.07, 0.0))


def test_c2z_fixes_every_trim():
    op = _op(C2_Z)
    for k in [(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0.5, 0.5, 0)]:
        assert indicators.fixes_kpoint(op, k)
