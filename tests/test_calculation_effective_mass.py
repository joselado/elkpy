"""The k.p sum-rule effective mass against Elk's own task 25, on bulk Si,
with a real compiled elk binary.

This is the cross-check the feature exists for. The two routes to
d^2 eps_n/dk_a dk_b share nothing but the self-consistent density:

  * Calculation.get_effective_mass() (task 25, src/effmass.f90) fits a
    polynomial to EIGENVALUES on a small k-mesh around the point and
    differentiates it -- no matrix elements anywhere.
  * Calculation.get_effective_mass_sum_rule() /
    parsers.optical.effective_mass_tensor() sums momentum MATRIX ELEMENTS
    over intermediate states at the single k-point -- no k-derivative
    anywhere.

Their agreement is limited by a physical truncation, not by numerics, and
that is the interesting content: the sum's 1/(eps_n - eps_m) weight decays
only as the FIRST power of the energy denominator (unlike the 1/(delta
eps)^2 Kubo geometric sums of docs/design.md #22), so the states Elk does
not compute -- everything above nempty, plus the core states, plus the
continuum the linearized LAPW basis cannot represent -- still matter.
The tests below therefore assert the CONVERGENCE BEHAVIOUR as much as the
value: a truncated sum overestimates the curvature and falls monotonically
towards the finite-difference answer as more states are retained.

Skipped if the elk binary hasn't been built, same as test_calculation_si.py.
"""

import numpy as np
import pytest

from elkpy import config
from elkpy.parsers import optical
from elkpy.structure import Structure

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)

SI_AVEC = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
SI_SPECIES = {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}

GAMMA = (0.0, 0.0, 0.0)
GENERIC = (0.2, 0.15, 0.1)  # low symmetry: every band non-degenerate


@pytest.fixture(scope="module")
def si(tmp_path_factory):
    """Bulk Si with nempty far above Elk's default -- the k.p sum needs
    intermediate states to run over, and its slow 1/(delta eps) tail makes
    the default useless here."""
    workdir = tmp_path_factory.mktemp("effmass_kp")
    calc = Structure(SI_AVEC, SI_SPECIES).get_calculation(
        workdir / "si", xc="PW", ngridk=(4, 4, 4), rgkmax=7.0,
        extra_blocks={"nempty": [40]},
    )
    calc.get_energy()
    with calc.eigenstate_session(label="momentum") as session:
        moments = {GAMMA: session.momentum(GAMMA), GENERIC: session.momentum(GENERIC)}
    # task 25 is a separate elk run per k-point (27 diagonalisations each);
    # do both once here rather than in every test that needs a reference
    finite_difference = {
        GAMMA: calc.get_effective_mass(GAMMA),
        GENERIC: calc.get_effective_mass(GENERIC),
    }
    return calc, moments, finite_difference


def test_sum_rule_matches_task_25_at_gamma(si):
    """Si's lowest valence band at Gamma (the non-degenerate Gamma_1, s
    bonding) is the cleanest case: isolated by 12 eV from everything else,
    so no degeneracy ambiguity on either side.

    Tolerance is 6%, and that is a physical statement, not slack: with all
    85 computed states retained the sum rule gives 0.888 against task 25's
    0.861, i.e. it recovers about 80% of the interband repulsion that
    lowers the curvature below the free-electron value of 1. The rest sits
    in states this basis does not have.
    """
    calc, moments, finite_difference = si
    fd = finite_difference[GAMMA]
    m = moments[GAMMA]
    kp = optical.effective_mass_tensor(m.energies, m.pmat, ist=1)

    # task 25's "matrix of eigenvalue derivatives" IS the inverse mass;
    # its "effective mass tensor" is that matrix inverted
    reference = fd[0]["derivative_tensor"]
    assert fd[0]["eigenvalue"] == pytest.approx(kp["energy"], abs=1e-6)
    assert np.allclose(np.diag(reference), np.diag(reference)[0], rtol=1e-3)  # cubic
    assert np.allclose(kp["inverse_mass"], reference, rtol=0.06, atol=1e-3)
    assert np.allclose(kp["mass"], fd[0]["tensor"], rtol=0.06, atol=1e-3)


def test_sum_rule_matches_task_25_at_a_generic_k_point(si):
    """The same comparison where the tensor is not forced isotropic by
    symmetry: at a low-symmetry k-point the OFF-diagonal components are
    nonzero and must match too, which the cubic point at Gamma cannot
    test (its off-diagonals vanish for both routes trivially)."""
    calc, moments, finite_difference = si
    fd = finite_difference[GENERIC]
    m = moments[GENERIC]
    kp = optical.effective_mass_tensor(m.energies, m.pmat, ist=1)
    reference = fd[0]["derivative_tensor"]
    assert abs(reference[0, 1]) > 0.01  # genuinely off-diagonal
    assert np.allclose(kp["inverse_mass"], reference, rtol=0.06, atol=2e-3)


def test_truncated_sum_overestimates_and_falls_monotonically(si):
    """The convergence behaviour IS the result. Every state omitted by the
    truncation lies ABOVE band n, so its omitted term
    -2|p^a_nm|^2/|eps_n - eps_m| is negative: a truncated (1/m*)^{aa} is
    an overestimate and must fall as more states are retained, towards
    (but here not reaching) the finite-difference value.

    Retaining only the four occupied bands gives exactly the free-electron
    1 -- the interband repulsion is entirely a coupling to empty states,
    and at Gamma the coupling to the occupied triplet is symmetry-forbidden
    (see test_dipole_forbidden_intermediate_states_contribute_exactly_zero).
    """
    calc, moments, finite_difference = si
    m = moments[GAMMA]
    trend = [
        optical.effective_mass_tensor(m.energies, m.pmat, 1, nstates=n)["inverse_mass"][0, 0]
        for n in (4, 8, 20, 40, len(m.energies))
    ]
    assert trend[0] == pytest.approx(1.0, abs=1e-9)
    assert all(a >= b - 1e-9 for a, b in zip(trend, trend[1:])), trend
    assert trend[-1] < trend[0]
    reference = finite_difference[GAMMA][0]["derivative_tensor"][0, 0]
    assert trend[-1] > reference  # still approaching from above
    assert abs(trend[-1] - reference) < abs(trend[1] - reference)


def test_dipole_forbidden_intermediate_states_contribute_exactly_zero(si):
    """The decomposition a finite-difference mass cannot give: which
    interband coupling produces the mass.

    At Gamma, diamond Si's states have definite parity about the bond
    centre and the momentum operator is odd, so a transition between two
    even states is forbidden. Band 1 (Gamma_1, even) therefore gets
    nothing at all from the even valence triplet Gamma_25' (bands 2-4) --
    exactly zero, at machine precision, not merely small -- and its entire
    curvature comes from the odd conduction triplet Gamma_15 (bands 5-7)
    and its higher analogues. This is the textbook selection rule behind
    Si's optical spectrum (Gamma_25' -> Gamma_15 allowed, Gamma_1 ->
    Gamma_25' forbidden), and a numerical accident could not produce a
    1e-30 matrix element.
    """
    _, moments, _ = si
    m = moments[GAMMA]
    contributions = optical.effective_mass_tensor(
        m.energies, m.pmat, 1, decompose=True
    )["contributions"]
    assert np.abs(contributions[1:4]).max() < 1e-12
    triplet = contributions[4:7]
    assert np.abs(np.diag(triplet[0])).min() > 0.01
    # the three partners of a degenerate triplet contribute equally
    assert np.allclose(np.trace(triplet[0]), np.trace(triplet[1]), rtol=1e-6)
    # and they dominate: > half of the whole deviation from the free-electron 1
    total_deviation = contributions.sum(axis=0)[0, 0]
    assert abs(np.sum([c[0, 0] for c in triplet])) > 0.5 * abs(total_deviation)


def test_oscillator_strength_sum_is_delta_minus_the_inverse_mass(si):
    """f^{ab} = delta_ab - (1/m*)^{ab} holds identically at one k-point,
    which is exactly why 'f = 1 per electron' is a BZ-averaged statement
    rather than a pointwise one. Its value here is as a truncation
    diagnostic: Tr f / 3 = 0.11 at Gamma against task 25's 0.14 says the
    retained states carry about 80% of the oscillator strength."""
    calc, moments, finite_difference = si
    m = moments[GAMMA]
    f = optical.oscillator_strength_sum(m.energies, m.pmat, 1)
    kp = optical.effective_mass_tensor(m.energies, m.pmat, 1)
    assert np.allclose(f, np.eye(3) - kp["inverse_mass"], atol=1e-12)
    reference = finite_difference[GAMMA][0]["derivative_tensor"]
    exact = float(np.trace(np.eye(3) - reference)) / 3
    assert 0.7 * exact < float(np.trace(f)) / 3 < exact


def test_degenerate_band_raises(si):
    """Si's valence-band top at Gamma is the three-fold degenerate
    Gamma_25': a single member's curvature is not defined there, and
    neither route computes anything meaningful. Fail loud."""
    _, moments, _ = si
    m = moments[GAMMA]
    with pytest.raises(ValueError, match="degenerate"):
        optical.effective_mass_tensor(m.energies, m.pmat, ist=2)


def test_calculation_wrapper_matches_the_parser(si):
    """get_effective_mass_sum_rule() is the one-off convenience wrapper --
    same result as driving eigenstate_session()/parsers.optical directly."""
    calc, moments, _ = si
    m = moments[GAMMA]
    direct = optical.effective_mass_tensor(m.energies, m.pmat, 1)
    wrapped = calc.get_effective_mass_sum_rule(GAMMA, ist=1)
    assert wrapped["k"] == GAMMA
    assert np.allclose(wrapped["inverse_mass"], direct["inverse_mass"], rtol=1e-6)
