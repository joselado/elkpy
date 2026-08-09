"""Unit tests for elkpy.parsers.spin, on synthetic data -- no Elk run needed
(mirrors tests/test_berry_gauge_invariance.py and
tests/test_quantum_geometry_gauge_invariance.py's style and role: pin the
Python-side arithmetic against known algebraic identities before trusting it
against a real diagonalization).

Two independent things are pinned here:

1. The spin-1/2 algebra itself -- Hermiticity, the su(2) commutation
   relation [Sx, Sy] = i*Sz, and the Casimir Sx^2+Sy^2+Sz^2 = (3/4) I --
   holds for ANY unitary change of basis (i.e. any evecsv from a real
   diagonalization), since S_a in the eigenbasis is exactly
   evecsv^H (I_nstfv (x) sigma_a/2) evecsv, a similarity transform of the
   fixed physical operator. This is checked on a random unitary evecsv, not
   just the trivial identity basis.
2. The absolute normalization and sign convention against known pure spin
   states: a spin-up/spin-down pair (built directly from the up/down block
   split, no diagonalization involved) gives <Sz> = +-1/2 and <Sx>=<Sy>=0;
   the sigma_x/sigma_y eigenstates (|up>+-|down>)/sqrt(2) and
   (|up>+-i|down>)/sqrt(2) give <Sx>=+-1/2 and <Sy>=+-1/2 respectively.
"""

import numpy as np
import pytest

from elkpy.parsers.spin import compute_spin_operator


def _random_unitary(rng, n):
    m = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(m)
    phases = np.diagonal(r) / np.abs(np.diagonal(r))
    return q * phases


def test_mismatched_nstfv_raises():
    evecsv = np.eye(6, dtype=complex)
    with pytest.raises(ValueError, match="nspinor"):
        compute_spin_operator(evecsv, nstfv=4, ist0=1, ist1=6)


def test_hermitian_commutation_and_casimir_on_random_basis():
    rng = np.random.default_rng(0)
    nstfv = 4
    nstsv = 2 * nstfv
    evecsv = _random_unitary(rng, nstsv)
    ops = compute_spin_operator(evecsv, nstfv, ist0=1, ist1=nstsv)
    sx, sy, sz = ops["sx"], ops["sy"], ops["sz"]

    for name, s in ops.items():
        assert s == pytest.approx(s.conj().T, abs=1e-10), f"{name} not Hermitian"

    # su(2): [Sx, Sy] = i Sz (and cyclic), invariant under the similarity
    # transform evecsv^H (.) evecsv that produced these matrices.
    assert sx @ sy - sy @ sx == pytest.approx(1j * sz, abs=1e-10)
    assert sy @ sz - sz @ sy == pytest.approx(1j * sx, abs=1e-10)
    assert sz @ sx - sx @ sz == pytest.approx(1j * sy, abs=1e-10)

    # Casimir S^2 = (3/4) I for spin-1/2, basis-independent.
    casimir = sx @ sx + sy @ sy + sz @ sz
    assert casimir == pytest.approx(0.75 * np.eye(nstsv), abs=1e-10)


def test_pure_spin_up_and_down_states():
    nstfv = 3
    nstsv = 2 * nstfv
    evecsv = np.zeros((nstsv, 2), dtype=complex)
    evecsv[0, 0] = 1.0  # spin-up on spatial state 0
    evecsv[nstfv, 1] = 1.0  # spin-down on spatial state 0
    ops = compute_spin_operator(evecsv, nstfv, ist0=1, ist1=2)

    assert np.diag(ops["sz"]).real == pytest.approx([0.5, -0.5])
    # Sx/Sy flip spin, so their off-diagonal <up|.|down> is nonzero (0.5) --
    # only the *diagonal* expectation value of a definite-Sz state vanishes.
    assert np.diag(ops["sx"]).real == pytest.approx([0.0, 0.0], abs=1e-12)
    assert np.diag(ops["sy"]).real == pytest.approx([0.0, 0.0], abs=1e-12)


def test_sigma_x_and_sigma_y_eigenstates():
    nstfv = 2
    nstsv = 2 * nstfv
    sqrt2 = 2**0.5
    evecsv = np.zeros((nstsv, 4), dtype=complex)
    evecsv[0, 0], evecsv[nstfv, 0] = 1 / sqrt2, 1 / sqrt2  # sigma_x = +1
    evecsv[0, 1], evecsv[nstfv, 1] = 1 / sqrt2, -1 / sqrt2  # sigma_x = -1
    evecsv[0, 2], evecsv[nstfv, 2] = 1 / sqrt2, 1j / sqrt2  # sigma_y = +1
    evecsv[0, 3], evecsv[nstfv, 3] = 1 / sqrt2, -1j / sqrt2  # sigma_y = -1
    ops = compute_spin_operator(evecsv, nstfv, ist0=1, ist1=4)

    assert np.diag(ops["sx"]).real == pytest.approx([0.5, -0.5, 0.0, 0.0], abs=1e-12)
    assert np.diag(ops["sy"]).real == pytest.approx([0.0, 0.0, 0.5, -0.5], abs=1e-12)
