"""Gauge-invariance and physics-grounded tests for the non-Abelian
Wilson-loop / Wannier-charge-center (WCC) / Z2 arithmetic in
elkpy.parsers.wilson, on synthetic data -- no Elk run needed.

Same spirit as test_berry_gauge_invariance.py: the WCC construction
(arXiv:1101.2011 eq. 5-6) is built to be exactly gauge-invariant under the
arbitrary per-k unitary freedom of the underlying wavefunctions, and we can
apply a synthetic gauge transformation directly to link-overlap matrices to
confirm that.

For the Z2 crossing-count itself (arXiv:1102.5600 eqs. 16-18), gauge
invariance alone is a weak check (a completely broken crossing-counter
could still be gauge invariant if it just always returns 0). The decisive
test instead cross-validates the WHOLE wilson.py pipeline against a
physically grounded, independently-checkable model: a Sz-conserving,
time-reversal-symmetric two-copy Qi-Wu-Zhang (QWZ) lattice model (spin-up =
one QWZ Chern insulator, spin-down = its complex-conjugate partner, giving
exactly the opposite Chern number by construction -- complex conjugation of
a Bloch Hamiltonian flips the sign of Berry curvature). Whenever Sz is
conserved, the Kane-Mele Z2 invariant is the spin Chern number mod 2 (Kane
& Mele, PRL 95, 146802 (2005)) -- a standard, safe theoretical fact, so
Z2 = 1 iff the single-spin-sector Chern number is odd. The single-sector
Chern number here is computed independently via elkpy.parsers.berry's own
(already Chern-number-tested, see test_calculation_berry.py) plaquette-flux
construction, so this test does not just re-derive its own expectation --
it cross-checks wilson.py's Z2 against a DIFFERENT, already-trusted code
path applied to the SAME underlying wavefunctions.
"""

import numpy as np
import pytest

from elkpy.parsers import berry, wilson

# --- gauge invariance / sign-convention pins for the raw WCC construction ---


def _random_unitary(rng, n):
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    return q * (d / np.abs(d))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_wannier_centers_are_gauge_invariant(seed):
    rng = np.random.default_rng(seed)
    nst, n_links = 3, 6
    links = [
        np.eye(nst) + 0.3 * (rng.normal(size=(nst, nst)) + 1j * rng.normal(size=(nst, nst)))
        for _ in range(n_links)
    ]
    gauges = [_random_unitary(rng, nst) for _ in range(n_links)]
    gauged_links = [
        gauges[i].conj().T @ links[i] @ gauges[(i + 1) % n_links] for i in range(n_links)
    ]

    before = wilson.wilson_loop_wannier_centers(links)
    after = wilson.wilson_loop_wannier_centers(gauged_links)
    np.testing.assert_allclose(before, after, atol=1e-10)


def test_identity_links_give_zero_wannier_centers():
    """A perfectly adiabatic loop (every link exactly the identity) must
    give exactly zero WCCs, by construction (D = identity)."""
    nst, n_links = 3, 5
    links = [np.eye(nst) for _ in range(n_links)]
    np.testing.assert_allclose(wilson.wilson_loop_wannier_centers(links), 0.0, atol=1e-12)


def test_single_band_wannier_center_matches_link_phase_product():
    """Pins the sign/formula for the 1-band (Abelian) special case: with a
    single occupied band, a single link carrying phase theta0 (rest exactly
    the identity) gives a WCC of exactly theta0 -- the 1x1 case of eq. 5-6
    where _unitarize reduces to parsers.berry._link_variable's
    det(M)/|det(M)|."""
    theta0 = 0.7
    links = [np.array([[1.0 + 0j]]) for _ in range(6)]
    links[2] = np.array([[np.exp(1j * theta0)]])
    result = wilson.wilson_loop_wannier_centers(links)
    assert result == pytest.approx([theta0], abs=1e-12)


def test_largest_gap_center_finds_the_bigger_arc():
    """Direct pin of _largest_gap_center: for two WCCs at -0.5 and 0.2, the
    short arc between them (length 0.7, going the direct way) is smaller
    than the long arc (length 2*pi - 0.7, going the other way around) -- the
    returned center must sit in the long arc, equidistant (going the long
    way) from both points."""
    center = wilson._largest_gap_center(np.array([-0.5, 0.2]))

    def circ_dist_ccw(a, b):
        return (b - a) % (2 * np.pi)

    d1 = circ_dist_ccw(0.2, center)
    d2 = circ_dist_ccw(center, -0.5 + 2 * np.pi)
    assert d1 == pytest.approx(d2, abs=1e-9)
    # not inside the short arc (-0.5, 0.2)
    assert not (-0.5 < center < 0.2)


def test_orientation_sign_convention():
    """Direct pin of _orientation (arXiv:1102.5600 eq. 17): c=0.5 lies on
    the direct (short, increasing) arc from a=0 to b=1, giving a negative
    sign; c=1.5 and c=-0.5 lie outside that arc (on the complementary one),
    giving a positive sign."""
    assert wilson._orientation(0.0, 1.0, 0.5) < 0
    assert wilson._orientation(0.0, 1.0, 1.5) > 0
    assert wilson._orientation(0.0, 1.0, -0.5) > 0


# --- physically grounded Z2 validation: Sz-conserving two-copy QWZ model ---

_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def _h_up(kx, ky, mass):
    """Qi-Wu-Zhang 2-band lattice Chern insulator (up-spin sector)."""
    return np.sin(kx) * _SX + np.sin(ky) * _SY + (mass - np.cos(kx) - np.cos(ky)) * _SZ


def _h_down(kx, ky, mass):
    """Complex-conjugate partner of _h_up -- exactly the opposite Chern
    number (complex conjugation of a Bloch Hamiltonian flips the sign of
    Berry curvature), giving an Sz-conserving, time-reversal-symmetric
    4-band (2 orbital x 2 spin) model overall."""
    return np.sin(kx) * _SX - np.sin(ky) * _SY + (mass - np.cos(kx) - np.cos(ky)) * _SZ


def _lower_band_evec(h):
    _, v = np.linalg.eigh(h)
    return v[:, 0]


def _spin_chern_number(hfun, mass, n=24):
    """The up-sector's own Chern number, via elkpy.parsers.berry's
    already-tested plaquette-flux construction (independent of wilson.py) --
    the ground truth this test's Z2 result is checked against."""
    evecs = {
        (i1, i2, 0): _lower_band_evec(hfun(2 * np.pi * i1 / n, 2 * np.pi * i2 / n, mass))
        for i1 in range(n)
        for i2 in range(n)
    }
    overlaps = {}
    for i1 in range(n):
        for i2 in range(n):
            k = (i1, i2, 0)
            k1, k2 = ((i1 + 1) % n, i2, 0), (i1, (i2 + 1) % n, 0)
            overlaps[k] = (
                np.array([[np.vdot(evecs[k], evecs[k1])]]),
                np.array([[np.vdot(evecs[k], evecs[k2])]]),
            )
    parsed = {"ngridk": (n, n, 1), "directions": (1, 2), "overlaps": overlaps}
    return berry.compute_berry_curvature(parsed)["chern_number"][0]


def _kane_mele_like_z2(mass, nkx=41, nt=21):
    """Z2 of the two occupied bands (lower band of each spin sector) of the
    Sz-conserving double-QWZ model above, via wilson.py's own pipeline --
    exactly the construction Calculation.get_z2_invariant() uses, just fed
    from this hand-built model's wavefunctions instead of Elk's."""
    nky_full = 2 * (nt - 1)
    theta_by_step = []
    for j in range(nt):
        ky = 2 * np.pi * j / nky_full
        states = [
            (
                _lower_band_evec(_h_up(2 * np.pi * i / nkx, ky, mass)),
                _lower_band_evec(_h_down(2 * np.pi * i / nkx, ky, mass)),
            )
            for i in range(nkx)
        ]
        link_overlaps = []
        for i in range(nkx):
            up_a, down_a = states[i]
            up_b, down_b = states[(i + 1) % nkx]
            # spin sectors are exactly orthogonal (block-diagonal model) --
            # the occupied-manifold overlap matrix is exactly diagonal.
            link_overlaps.append(
                np.array(
                    [[np.vdot(up_a, up_b), 0.0], [0.0, np.vdot(down_a, down_b)]]
                )
            )
        theta_by_step.append(wilson.wilson_loop_wannier_centers(link_overlaps))
    return wilson.z2_from_wannier_centers(theta_by_step)


@pytest.mark.parametrize(
    "mass,expect_z2",
    [(-0.5, 1), (1.0, 1), (3.0, 0), (4.5, 0)],
)
def test_z2_matches_spin_chern_number_parity(mass, expect_z2):
    """Kane & Mele, PRL 95, 146802 (2005): whenever Sz is conserved, Z2 is
    the spin Chern number mod 2. mass=-0.5 and mass=1.0 sit in this model's
    topological regime (chern_up = -1 and +1 respectively, both odd);
    mass=3.0 and 4.5 sit in the trivial regime (chern_up = 0) -- see this
    module's docstring for why cross-checking against parsers.berry's own
    (already-tested) Chern number, rather than hand-asserting a Z2 value, is
    the decisive validation here."""
    chern_up = _spin_chern_number(_h_up, mass)
    assert round(chern_up) % 2 == expect_z2  # sanity: the model is in the regime this test claims
    assert _kane_mele_like_z2(mass) == expect_z2
