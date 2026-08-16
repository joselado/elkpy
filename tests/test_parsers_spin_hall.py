"""Correctness pins for elkpy.parsers.spin_hall -- the spin Berry curvature
built from the spin current operator J^s_a = (1/2){S_s, v_a} -- on synthetic
models, no Elk run needed.

Same spirit as test_parsers_optical.py, whose massive Dirac model this
reuses: for H_tau(k) = v(tau k_x sigma_x + k_y sigma_y) + Delta sigma_z the
velocity operator is exactly p = dH/dk, so a synthetic (energies, pmat, S)
triple can be handed straight to the module with no numerical
differentiation anywhere.

The decisive check is not internal consistency but agreement with an
ALREADY-TRUSTED independent code path -- parsers.optical's ordinary Kubo
Berry curvature, itself sign-pinned against direct numerical
differentiation. A system with conserved S_z decouples into two spin
sectors, in which the spin Berry curvature must reduce to

    Omega^s = sum_sigma s_sigma Omega^sigma = (Omega^up - Omega^down) / 2

(the factor 1/2 because elkpy's S has eigenvalues +-1/2, not +-1 -- see
parsers/spin_hall.py's docstring), with each Omega^sigma computed by
parsers.optical on that sector alone.

Two sector combinations are checked, because either alone is fooled by a
plausible bug the other catches:

  * the time-reversal pair (tau_up = +1, tau_down = -1): charge curvature
    cancels to zero while the spin curvature is maximal -- so a
    J-vs-v swap, which would make this return the charge curvature, shows
    up as 0 instead of a large number;
  * two copies of the SAME valley (tau_up = tau_down = +1): the charge
    curvature adds while the spin curvature cancels to zero -- the
    opposite pattern, which catches a missing or misplaced S factor that
    the first case's "spin = one sector's Omega" coincidence would hide.
"""

import numpy as np
import pytest

from elkpy.parsers import optical, spin_hall

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def _dirac_states(kx, ky, delta, tau=1, v=1.0):
    """One massive-Dirac valley: (energies, pmat) in parsers.optical's
    convention, with pmat = dH/dk exact (test_parsers_optical.py's model)."""
    h = v * (tau * kx * SX + ky * SY) + delta * SZ
    energies, vecs = np.linalg.eigh(h)
    p_cart = [v * tau * SX, v * SY, np.zeros((2, 2), dtype=complex)]
    pmat = np.array([vecs.conj().T @ p @ vecs for p in p_cart])
    return energies, pmat


def _two_sector_states(kx, ky, delta, tau_up=1, tau_dn=-1, v=1.0):
    """Two decoupled spin sectors glued into one 4-state system with
    conserved S_z: block-diagonal velocity, S_z = +-1/2 on the blocks.

    The combined states are sorted by energy and pmat/S_z permuted with
    the SAME permutation, so that the band window [1, 2] really is "both
    valence bands" rather than "one sector's pair".
    """
    e_up, p_up = _dirac_states(kx, ky, delta, tau_up, v)
    e_dn, p_dn = _dirac_states(kx, ky, delta, tau_dn, v)
    energies = np.concatenate([e_up, e_dn])
    pmat = np.zeros((3, 4, 4), dtype=complex)
    for a in range(3):
        pmat[a][np.ix_([0, 1], [0, 1])] = p_up[a]
        pmat[a][np.ix_([2, 3], [2, 3])] = p_dn[a]
    sz = np.diag([0.5, 0.5, -0.5, -0.5]).astype(complex)
    order = np.argsort(energies, kind="stable")
    energies = energies[order]
    pmat = pmat[:, order][:, :, order]
    sz = sz[np.ix_(order, order)]
    return energies, pmat, sz, (e_up, p_up), (e_dn, p_dn)


# --- the reduction to two spin sectors: the pin with teeth ---


@pytest.mark.parametrize("tau_up,tau_dn", [(1, -1), (1, 1), (-1, -1), (-1, 1)])
@pytest.mark.parametrize("kx,ky", [(0.0, 0.0), (0.3, -0.2), (0.6, 0.45)])
def test_conserved_sz_reduces_to_the_sector_curvature_difference(tau_up, tau_dn, kx, ky):
    """With S_z conserved the spin current is J^z_a = s_sigma v_a within
    each sector, so the spin Berry curvature of the occupied window must
    equal sum_sigma s_sigma Omega^sigma, each Omega^sigma computed by
    parsers.optical (an independent, already sign-pinned code path) on
    that sector alone.

    This is an exact identity, not an approximation -- checked to
    floating-point tolerance, and over all four valley combinations so
    that neither the "spin = one sector" nor the "spin = 0" special case
    can carry the test on its own.
    """
    delta = 0.5
    energies, pmat, sz, (e_up, p_up), (e_dn, p_dn) = _two_sector_states(
        kx, ky, delta, tau_up, tau_dn
    )
    got = spin_hall.spin_berry_curvature(energies, pmat, sz, ist0=1, ist1=2)
    omega_up = optical.kubo_berry_curvature(e_up, p_up, ist0=1, ist1=1)
    omega_dn = optical.kubo_berry_curvature(e_dn, p_dn, ist0=1, ist1=1)
    expected = 0.5 * omega_up - 0.5 * omega_dn
    assert got == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_time_reversal_pair_has_zero_charge_and_maximal_spin_curvature():
    """The physically interesting case: a time-reversal-symmetric pair of
    valleys (the Kane-Mele/quantum-spin-Hall setup). The two sectors carry
    exactly opposite ordinary curvature, so the CHARGE curvature of the
    occupied window vanishes identically while the SPIN curvature does
    not -- the sign-of-the-effect statement that a spin Hall response can
    exist with no anomalous Hall response at all.

    A J-vs-v swap inside spin_berry_curvature() would return the charge
    curvature here, i.e. 0 instead of a large number.
    """
    delta = 0.5
    energies, pmat, sz, (e_up, p_up), _ = _two_sector_states(0.2, 0.1, delta, 1, -1)
    charge = optical.kubo_berry_curvature(energies, pmat, ist0=1, ist1=2)
    spin = spin_hall.spin_berry_curvature(energies, pmat, sz, ist0=1, ist1=2)
    omega_up = optical.kubo_berry_curvature(e_up, p_up, ist0=1, ist1=1)
    assert charge == pytest.approx(0.0, abs=1e-12)
    assert spin == pytest.approx(omega_up, rel=1e-10)
    assert abs(spin) > 1.0


def test_two_copies_of_one_valley_has_zero_spin_and_doubled_charge_curvature():
    """The complementary case, which the one above cannot catch: two
    copies of the SAME valley labelled up and down. Now the ordinary
    curvatures add (charge curvature doubles) and the spin curvature
    cancels exactly. A dropped or mis-signed S factor breaks this while
    leaving the time-reversal case's numbers untouched."""
    delta = 0.5
    energies, pmat, sz, (e_up, p_up), _ = _two_sector_states(0.2, 0.1, delta, 1, 1)
    charge = optical.kubo_berry_curvature(energies, pmat, ist0=1, ist1=2)
    spin = spin_hall.spin_berry_curvature(energies, pmat, sz, ist0=1, ist1=2)
    omega_up = optical.kubo_berry_curvature(e_up, p_up, ist0=1, ist1=1)
    assert charge == pytest.approx(2 * omega_up, rel=1e-10)
    assert spin == pytest.approx(0.0, abs=1e-12)


def test_identity_spin_operator_recovers_the_charge_curvature():
    """S -> 1 makes J^s_a = v_a identically, so the spin Berry curvature
    must collapse onto parsers.optical's ordinary one. The cheapest
    possible pin that the two share a sign convention (A = i<u|grad u>,
    Omega = curl A) and that the anticommutator carries no stray factor:
    (1/2){1, v} = v, not 2v or v/2."""
    energies, pmat = _dirac_states(0.3, -0.2, delta=0.5)
    identity = np.eye(2, dtype=complex)
    got = spin_hall.spin_berry_curvature(energies, pmat, identity, ist0=1, ist1=1)
    expected = optical.kubo_berry_curvature(energies, pmat, ist0=1, ist1=1)
    assert got == pytest.approx(expected, rel=1e-12)


# --- the spin current operator itself ---


def test_spin_current_operator_is_hermitian_when_s_and_v_do_not_commute():
    """The symmetrization is load-bearing, not cosmetic: S v alone is not
    Hermitian once [S, v] != 0 (the spin-orbit-coupled case this whole
    module exists for), and a non-Hermitian J would break the exact
    pairwise cancellation of intra-window terms that makes the
    window-summed curvature well defined at all."""
    rng = np.random.default_rng(11)
    n = 5
    def _herm():
        m = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        return m + m.conj().T
    pmat = np.array([_herm() for _ in range(3)])
    s = _herm()
    j = spin_hall.spin_current_operator(pmat, s, direction=2)
    assert np.allclose(j, j.conj().T)
    # and the naive unsymmetrized product genuinely is not Hermitian here,
    # so this test is not vacuous
    naive = s @ pmat[1]
    assert not np.allclose(naive, naive.conj().T)


def test_spin_current_operator_rejects_a_windowed_spin_operator():
    """The commonest way to get this wrong: passing a band-windowed S
    (what EigenstateSession.spin_operator(k, ist0, ist1) returns) against
    a full-nstsv pmat. Caught by shape, with a message pointing at the
    one-diagonalisation rule."""
    pmat = np.zeros((3, 4, 4), dtype=complex)
    with pytest.raises(ValueError, match="same eigenbasis as pmat"):
        spin_hall.spin_current_operator(pmat, np.eye(2, dtype=complex))


def test_spin_berry_curvature_rejects_bad_directions():
    energies, pmat = _dirac_states(0.1, 0.1, delta=0.5)
    with pytest.raises(ValueError, match="distinct Cartesian axes"):
        spin_hall.spin_berry_curvature(
            energies, pmat, np.eye(2, dtype=complex), 1, 1, directions=(1, 1)
        )


# --- the intra-window cancellation the window sum relies on ---


def test_window_sum_equals_the_full_sum_over_all_other_states():
    """parsers.optical.kubo_sum drops pairs with both states inside the
    window. For a window-SUMMED quantity that is exact for any Hermitian
    J and v -- the reversed pair's numerator is the complex conjugate
    over the same denominator, so the imaginary parts cancel -- and this
    checks it directly against the textbook form of the formula, which
    sums over ALL m != n and would otherwise have to be regularized near
    an intra-window degeneracy.

    Built on a random Hermitian model with well-separated energies, so
    the brute-force reference is itself numerically safe.
    """
    rng = np.random.default_rng(3)
    n, ist0, ist1 = 6, 1, 3
    energies = np.sort(rng.normal(scale=2.0, size=n))
    def _herm():
        m = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        return m + m.conj().T
    pmat = np.array([_herm() for _ in range(3)])
    s = _herm()
    j = spin_hall.spin_current_operator(pmat, s, direction=1)
    v = pmat[1]

    brute = 0.0
    for nn in range(ist0 - 1, ist1):
        for mm in range(n):
            if mm == nn:
                continue
            brute += -2.0 * np.imag(
                j[nn, mm] * v[mm, nn] / (energies[nn] - energies[mm]) ** 2
            )
    got = spin_hall.spin_berry_curvature(energies, pmat, s, ist0, ist1)
    assert got == pytest.approx(brute, rel=1e-10)


# --- the Brillouin-zone integral ---


def test_spin_hall_conductivity_is_the_mean_curvature_over_the_cell_volume():
    curvatures = [1.0, 3.0, 5.0, 7.0]
    assert spin_hall.spin_hall_conductivity(curvatures, cell_volume=2.0) == pytest.approx(2.0)


def test_spin_hall_conductivity_weights_only_matter_up_to_normalization():
    """A symmetry-reduced mesh supplies multiplicities, whose absolute
    scale is arbitrary -- only the ratios may matter."""
    c = [1.0, 3.0, 5.0]
    a = spin_hall.spin_hall_conductivity(c, 1.0, weights=[1, 2, 3])
    b = spin_hall.spin_hall_conductivity(c, 1.0, weights=[10, 20, 30])
    assert a == pytest.approx(b)
    assert a == pytest.approx((1 * 1 + 2 * 3 + 3 * 5) / 6.0)


def test_spin_hall_conductivity_guards():
    with pytest.raises(ValueError, match="no curvature samples"):
        spin_hall.spin_hall_conductivity([], 1.0)
    with pytest.raises(ValueError, match="cell_volume must be positive"):
        spin_hall.spin_hall_conductivity([1.0], 0.0)
    with pytest.raises(ValueError, match="weights shape"):
        spin_hall.spin_hall_conductivity([1.0, 2.0], 1.0, weights=[1.0])
