"""Unit tests for elkpy.parsers.symmetry -- parity eigenvalues and the
Fu-Kane symmetry-indicator Z2 invariants -- on synthetic data, no Elk run.

Same role as test_parsers_optical.py / test_wilson_gauge_invariance.py: pin
the Python arithmetic independently of any Fortran, so a real-binary
disagreement later can be attributed to the export path rather than to the
formulas here.

The decisive check is that the Fu-Kane counting reproduces a known model
answer, not merely that it is self-consistent: an inversion-symmetric
Bernevig-Hughes-Zhang-style band inversion at exactly one TRIM must give
nu = 1, and at zero or two TRIM must give nu = 0.
"""

import numpy as np
import pytest

from elkpy.parsers import symmetry


def _parity_matrix(xi, seed=0):
    """A synthetic parity operator with the given eigenvalues, in a random
    basis -- so tests exercise the eigendecomposition rather than a
    conveniently diagonal matrix (TRIM spectra are degenerate, and the
    diagonalisation returns an arbitrary basis within each multiplet)."""
    rng = np.random.default_rng(seed)
    n = len(xi)
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    q = q * (np.diagonal(r) / np.abs(np.diagonal(r)))
    return q @ np.diag(np.array(xi, dtype=complex)) @ q.conj().T


# --- parity eigenvalue extraction ---


def test_eigenvalues_recovered_from_a_rotated_basis():
    xi = [1, 1, -1, -1, 1, -1]
    vals = symmetry.parity_eigenvalues(_parity_matrix(xi))
    assert sorted(vals) == sorted(xi)


def test_diagonal_is_not_the_eigenvalues():
    """Why parity_eigenvalues() diagonalises instead of reading the
    diagonal: in a rotated basis the diagonal entries are not +-1 at all."""
    p = _parity_matrix([1, 1, -1, -1])
    assert not np.allclose(np.abs(np.diagonal(p).real), 1.0, atol=1e-2)
    assert sorted(symmetry.parity_eigenvalues(p)) == [-1, -1, 1, 1]


def test_non_involutive_operator_raises():
    """An eigenvalue away from +-1 means the window is not an
    inversion-invariant group of bands -- fail loud, don't round."""
    p = _parity_matrix([1, -1, 0.3, -1])
    with pytest.raises(ValueError, match=r"must be \+-1"):
        symmetry.parity_eigenvalues(p)


def test_non_hermitian_operator_raises():
    p = _parity_matrix([1, -1, 1, -1])
    p[0, 1] += 0.5
    with pytest.raises(ValueError, match="not Hermitian"):
        symmetry.parity_eigenvalues(p)


# --- Kramers structure and delta ---


def test_delta_counts_one_member_per_kramers_pair():
    """delta = (-1)^(N_minus/2). Four negative eigenvalues (two Kramers
    pairs) give +1; two (one pair) give -1."""
    assert symmetry.trim_delta(_parity_matrix([1, 1, -1, -1, -1, -1])) == 1
    assert symmetry.trim_delta(_parity_matrix([1, 1, 1, 1, -1, -1])) == -1


def test_delta_over_all_states_would_be_useless():
    """The reason delta pairs states: with Kramers degeneracy every parity
    eigenvalue appears an even number of times, so the product over ALL
    occupied states is identically +1 and carries no information. The
    correct pairing must therefore distinguish cases that the naive product
    does not."""
    a = [1, 1, -1, -1, -1, -1]   # naive product +1, delta +1
    b = [1, 1, 1, 1, -1, -1]     # naive product +1, delta -1
    assert int(np.prod(a)) == int(np.prod(b)) == 1
    assert symmetry.trim_delta(_parity_matrix(a)) != symmetry.trim_delta(_parity_matrix(b))


def test_split_kramers_pair_raises():
    with pytest.raises(ValueError, match="odd number of negative parity"):
        symmetry.trim_delta(_parity_matrix([1, 1, 1, -1]))
    with pytest.raises(ValueError, match="odd number of states"):
        symmetry.trim_delta(_parity_matrix([1, -1, -1]))


# --- the invariants ---


def test_single_band_inversion_gives_a_topological_z2():
    """A band inversion at exactly one TRIM flips that TRIM's delta and no
    other -- the Bernevig-Hughes-Zhang mechanism -- giving nu = 1. Zero or
    two inversions give nu = 0. This is the model answer the counting must
    reproduce, not just an internal consistency check."""
    trivial = {k: 1 for k in symmetry.TRIM_2D}
    assert symmetry.fu_kane_z2_2d(trivial) == 0

    one = dict(trivial)
    one[symmetry.TRIM_2D[0]] = -1
    assert symmetry.fu_kane_z2_2d(one) == 1

    two = dict(one)
    two[symmetry.TRIM_2D[1]] = -1
    assert symmetry.fu_kane_z2_2d(two) == 0


def test_global_sign_flip_cannot_change_any_invariant():
    """Every invariant is a product over an EVEN number of TRIM, so
    flipping the sign convention of the parity operator globally leaves all
    of them unchanged -- which is why this module does not chase an
    absolute per-band sign pin (see its docstring)."""
    rng = np.random.default_rng(3)
    deltas = {k: int(rng.choice([-1, 1])) for k in symmetry.TRIM_3D}
    flipped = {k: -v for k, v in deltas.items()}
    assert symmetry.fu_kane_z2_3d(deltas) == symmetry.fu_kane_z2_3d(flipped)


def test_strong_versus_weak_3d():
    """nu0 is the product over all 8 TRIM; a single -1 makes a strong TI.
    Two -1 sharing a plane make nu0 = 0 with a weak index set instead."""
    trivial = {k: 1 for k in symmetry.TRIM_3D}
    assert symmetry.fu_kane_z2_3d(trivial) == {"nu0": 0, "nu": (0, 0, 0)}

    strong = dict(trivial)
    strong[(0.0, 0.0, 0.0)] = -1
    assert symmetry.fu_kane_z2_3d(strong)["nu0"] == 1

    # A weak TI needs the two -1 TRIM to STRADDLE the k3 = 0 and k3 = 1/2
    # planes: nu0 (all 8) is then even, while the k3 = 1/2 plane alone holds
    # an odd number -- i.e. stacked 2D QSH layers along direction 3. Putting
    # both -1 in the same plane instead leaves every index 0.
    weak = dict(trivial)
    weak[(0.0, 0.0, 0.0)] = -1
    weak[(0.0, 0.0, 0.5)] = -1
    out = symmetry.fu_kane_z2_3d(weak)
    assert out["nu0"] == 0
    assert out["nu"] == (0, 0, 1)

    # "Same plane" is per-axis, not absolute: these two -1 share the
    # k3 = 1/2 plane (so nu3 = 0) but differ in their k2 component, which
    # makes the k2 = 1/2 plane's product odd instead -- a weak stacking
    # along direction 2 rather than 3.
    same_k3 = dict(trivial)
    same_k3[(0.0, 0.0, 0.5)] = -1
    same_k3[(0.0, 0.5, 0.5)] = -1
    assert symmetry.fu_kane_z2_3d(same_k3) == {"nu0": 0, "nu": (0, 1, 0)}


def test_wrong_number_of_trim_raises():
    with pytest.raises(ValueError, match="expected 8"):
        symmetry.fu_kane_z2_3d({k: 1 for k in symmetry.TRIM_2D})
    with pytest.raises(ValueError, match="expected 4"):
        symmetry.fu_kane_z2_2d([1, 1, 1])


def test_non_trim_key_raises():
    bad = {k: 1 for k in symmetry.TRIM_3D}
    bad.pop((0.0, 0.0, 0.0))
    bad[(1 / 3, 0.0, 0.0)] = 1
    with pytest.raises(ValueError, match="not a TRIM"):
        symmetry.fu_kane_z2_3d(bad)


# --- the TRIM predicate ---


@pytest.mark.parametrize("k", [(0, 0, 0), (0.5, 0, 0), (0.5, 0.5, 0.5), (1.0, -0.5, 0)])
def test_is_trim_accepts(k):
    assert symmetry.is_trim(k)


@pytest.mark.parametrize("k", [(1 / 3, 1 / 3, 0), (0.25, 0, 0), (0.5, 0.1, 0)])
def test_is_trim_rejects(k):
    assert not symmetry.is_trim(k)


# --- the window must be a gapped band group, not merely Kramers-consistent ---


def test_check_window_gap_accepts_a_gapped_window():
    e = np.array([0.0, 0.0, 1.0, 1.0, 5.0, 5.0])
    symmetry.check_window_gap(e, 1, 4)  # gap of 4.0 above state 4


def test_check_window_gap_rejects_a_cut_band_group():
    """The hole this closes: a window boundary inside a band group passes
    every check in trim_delta() (Hermitian, +-1 eigenvalues, even Kramers
    counts) while describing no topological group at all. Measured on the
    [111]-dimerized diamond structure, where ist0 = 19 did exactly that and
    returned a confident but meaningless nu0 = 1."""
    e = np.array([0.0, 0.0, 1.0, 1.0, 1.0 + 1e-6, 1.0 + 1e-6, 5.0, 5.0])
    with pytest.raises(ValueError, match="not separated"):
        symmetry.check_window_gap(e, 5, 8)      # starts inside the group at 1.0
    with pytest.raises(ValueError, match="not gapped"):
        symmetry.check_window_gap(e, 1, 4)      # ends inside the same group


def test_check_window_gap_ignores_the_spectrum_edges():
    """No gap is required below state 1 or above the last state -- there is
    nothing there to be separated from."""
    e = np.array([0.0, 0.0, 1.0, 1.0])
    symmetry.check_window_gap(e, 1, 4)
