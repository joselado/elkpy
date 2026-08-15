"""Integration tests for Calculation.get_momentum_matrix() /
EigenstateSession.momentum() (task 9002's MOMENTUM query,
patches/0007-momentum-matrix-elements.patch) and the quantities
elkpy.parsers.optical builds on it, against a real compiled elk binary.

Skipped if the elk binary hasn't been built, same as test_calculation_si.py.

Note on what is NOT tested here: Hermiticity of pmat. Unlike the
projection/angular-momentum operators, genpmatk enforces it by
construction (it fills the upper triangle, sets the lower by conjugation
and forces the diagonal real), so asserting it would say nothing about
this export path. The checks with teeth are instead: the diagonal against
the eigenvalue slope (Hellmann-Feynman -- an entirely separate code path,
eigenvalues vs. matrix elements), and the off-diagonal geometry against
the already-trusted Wilson-loop curvature.
"""

import re

import numpy as np
import pytest

from elkpy import config
from elkpy.parsers import optical
from elkpy.structure import Structure

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)

A, VACUUM = 4.743210000, 20.0  # Bohr, same hBN slab as notebooks/05_berry_curvature.ipynb
HBN_AVEC = [(A, 0.0, 0.0), (-A / 2, A * 3**0.5 / 2, 0.0), (0.0, 0.0, VACUUM)]
HBN_SPECIES = {"B": [(1 / 3, 2 / 3, 0.5)], "N": [(2 / 3, 1 / 3, 0.5)]}

K = (1 / 3, 1 / 3, 0.0)
KPRIME = (-1 / 3, -1 / 3, 0.0)


@pytest.fixture(scope="module")
def hbn(tmp_path_factory):
    """Monolayer h-BN, with nempty raised well above Elk's default so the
    Kubo sums below have unoccupied states to run over."""
    workdir = tmp_path_factory.mktemp("momentum")
    calc = Structure(HBN_AVEC, HBN_SPECIES).get_calculation(
        workdir / "hbn", xc="PW", ngridk=(6, 6, 1), rgkmax=7.0,
        extra_blocks={"nempty": [12]},
    )
    calc.get_energy()
    # occupied-band count from EIGVAL.OUT's own occupation numbers, not an
    # assumed electron count -- core electrons aren't among the valence
    # bands nstsv indexes at all (docs/design.md #13's pitfall on this same
    # structure).
    first_block = (calc.workdir / "EIGVAL.OUT").read_text().split("k-point")[1]
    state_line = re.compile(r"^\s*\d+\s+(\S+)\s+(\S+)\s*$")
    occ = [float(m.group(2)) for m in map(state_line.match, first_block.splitlines()) if m]
    return calc, 1, sum(o > 1.0 for o in occ)


def test_band_velocity_matches_the_eigenvalue_slope(hbn):
    """Hellmann-Feynman: for a local Kohn-Sham potential the diagonal of
    the momentum matrix IS the band slope, v_n = dE_n/dk (hbar = m_e = 1).
    The right-hand side comes from eigenvalues alone, which share no code
    with the matrix-element machinery -- so this is the cheap check with
    real teeth on the whole export path (matching, genwfsv expansion,
    genpmatk reduction, token protocol, parser packing) that Hermiticity
    cannot provide.

    Stepping in FRACTIONAL coordinates differentiates along a reciprocal
    lattice vector, so the slope is v . b_i, not the Cartesian component
    v_i -- one dot product, and it exercises all three components at once.
    """
    calc, ist0, ist1 = hbn
    bvec = calc._reciprocal_vectors()
    k0 = (0.13, 0.21, 0.0)  # generic, low-symmetry: no degeneracies
    band = ist1  # valence-band top, non-degenerate away from K
    delta = 1e-3

    with calc.eigenstate_session(label="velocity") as session:
        v_cart = optical.band_velocity(session.momentum(k0).pmat, band)
        for axis in range(3):
            step = [0.0, 0.0, 0.0]
            step[axis] = delta
            kp = tuple(k0[i] + step[i] for i in range(3))
            km = tuple(k0[i] - step[i] for i in range(3))
            e_plus = session.get_eigenstates(kp).energies[band - 1]
            e_minus = session.get_eigenstates(km).energies[band - 1]
            slope = (e_plus - e_minus) / (2 * delta)
            assert float(v_cart @ bvec[axis]) == pytest.approx(slope, rel=2e-2, abs=1e-4)


def test_hbn_valley_circular_dichroism(hbn):
    """Valley-selective circular dichroism: the band-edge transition at
    the zone corner couples to ONE circular polarization only, with
    opposite handedness at the two inequivalent valleys -- eta(K) =
    -eta(K'), |eta| = 1 (Yao, Xiao & Niu, PRB 77, 235406 (2008); Xiao,
    Liu, Feng, Xu & Yao, PRL 108, 196802 (2012); Cao et al., Nat. Commun.
    3, 887 (2012)).

    The RELATIVE sign is the time-reversal-enforced, published
    sign-of-the-effect prediction -- the same K/K' spirit as the existing
    Berry-curvature, atom-projection and spin-operator tests. |eta| = 1
    is forced by the three-fold rotation symmetry at the corner; asserted
    at 0.98 rather than exactly 1 because that is a two-band k.p result
    and this is all-electron DFT with the full band manifold (measured:
    1.000000 to six figures, comfortably inside).

    The valence-band top and conduction-band bottom are each
    non-degenerate here, so the single-pair transition is the right
    object; lumping in another band would average distinct selectivities
    and dilute eta.
    """
    calc, ist0, ist1 = hbn
    with calc.eigenstate_session(label="dichroism") as session:
        eta = {}
        for name, kpoint in (("K", K), ("K'", KPRIME)):
            m = session.momentum(kpoint)
            eta[name] = optical.circular_polarization(
                m.pmat, valence=ist1, conduction=ist1 + 1, directions=(1, 2)
            )["eta"]

    assert abs(eta["K"]) > 0.98
    assert abs(eta["K'"]) > 0.98
    assert eta["K"] == pytest.approx(-eta["K'"], rel=1e-3)


def test_kubo_curvature_matches_wilson_loop(hbn):
    """The headline cross-check: the Berry curvature of h-BN's occupied
    manifold at K/K' from the Kubo sum over velocity matrix elements
    (this feature) against get_berry_curvature_path()'s Wilson loop over
    wavefunction overlaps (docs/design.md #13). The two share no Python
    arithmetic and, on the Fortran side, not even the same machinery --
    genpmatk's gradient/quadrature vs genolpq's real-space overlap.

    This is the check that fixed elkpy's Berry-phase sign convention. It
    originally failed with a clean factor of -1 (magnitudes agreeing to
    1.2%), which -- together with the synthetic pins in
    tests/test_parsers_optical.py -- identified a missing
    King-Smith--Vanderbilt/Resta negation in parsers.berry's flux step.
    Both routes now use the standard A = i<u|grad_k u>, Omega = curl A
    (Xiao, Chang & Niu, RMP 82, 1959 (2010)); see
    parsers.berry._berry_phase and docs/design.md #22.

    Because the same factor of -1 showed up in pure Python AND end-to-end
    here, the discrepancy was localized entirely in that Python step,
    which confirmed the Fortran moverlap/genolpq conjugation convention --
    previously resting only on a zgemv BLAS-semantics derivation with no
    runtime test.

    Residual disagreement is a few percent, not machine precision: the
    Kubo sum truncates at nstsv and omits core states, and the Wilson loop
    carries its own O(dk^2) error near a sharp valley peak.
    """
    calc, ist0, ist1 = hbn
    wilson = calc.get_berry_curvature_path(
        [K, KPRIME], ist0, ist1, directions=(1, 2), dk=0.01, label="wilson_ref"
    )
    with calc.eigenstate_session(label="kubo") as session:
        kubo = [
            optical.kubo_berry_curvature(
                *(lambda m: (m.energies, m.pmat))(session.momentum(kpoint)),
                ist0, ist1, directions=(1, 2),
            )
            for kpoint in (K, KPRIME)
        ]

    # both methods must see a real, valley-antisymmetric curvature, not noise
    assert abs(wilson[0]["curvature"]) > 1.0
    assert kubo[0] == pytest.approx(-kubo[1], rel=1e-3)
    for kubo_value, wilson_point in zip(kubo, wilson):
        assert kubo_value == pytest.approx(wilson_point["curvature"], rel=0.05)


def test_kubo_curvature_is_stable_in_nempty(hbn):
    """The Kubo sum over states truncates at nstsv, which `nempty` sets --
    the classic reason a Kubo curvature disagrees with a Wilson-loop one.
    This checks the result is actually converged in that parameter rather
    than assuming it: the module fixture's nempty=12 against a separate
    ground state at nempty=20.
    """
    calc, ist0, ist1 = hbn
    denser = Structure(HBN_AVEC, HBN_SPECIES).get_calculation(
        calc.workdir.parent / "hbn_nempty20", xc="PW", ngridk=(6, 6, 1), rgkmax=7.0,
        extra_blocks={"nempty": [20]},
    )
    denser.get_energy()

    with calc.eigenstate_session(label="nempty12") as session:
        m = session.momentum(K)
        low = optical.kubo_berry_curvature(m.energies, m.pmat, ist0, ist1)
    with denser.eigenstate_session(label="nempty20") as session:
        m = session.momentum(K)
        high = optical.kubo_berry_curvature(m.energies, m.pmat, ist0, ist1)

    assert high == pytest.approx(low, rel=0.02)


def test_kubo_metric_matches_get_quantum_geometry(hbn):
    """The quantum metric from the same Kubo sum, against
    get_quantum_geometry()'s finite-difference overlap stencil
    (docs/design.md #15) at K. Unlike the curvature there is no sign
    ambiguity here -- the metric is a real, positive-semi-definite Gram
    matrix in both constructions.

    The looser tolerance is expected on both sides: the Kubo metric
    truncates at nstsv (its diagonal converges from below), and the
    finite-difference metric still carries O(dk^2) error plus genolpq's
    Loewdin-corrected truncation floor.
    """
    calc, ist0, ist1 = hbn
    fd = calc.get_quantum_geometry(
        [K], ist0, ist1, directions=(1, 2), dk=0.01, label="qg_ref"
    )[0]
    with calc.eigenstate_session(label="kubo_metric") as session:
        m = session.momentum(K)
        kubo = optical.kubo_quantum_metric(m.energies, m.pmat, ist0, ist1, directions=(1, 2))

    assert np.all(np.linalg.eigvalsh(kubo) >= -1e-10)
    # diagonal components: the physically meaningful, well-conditioned pair
    for i in (0, 1):
        assert kubo[i, i] == pytest.approx(fd["g"][i, i], rel=0.15)


def test_momentum_band_window_is_applied_in_python(hbn):
    """ist0/ist1 are optional here (the Fortran query has no window at
    all -- genpmatk's array is hard-dimensioned nstsv) and slice both the
    energies and all three Cartesian blocks consistently."""
    calc, ist0, ist1 = hbn
    with calc.eigenstate_session(label="window") as session:
        full = session.momentum(K)
        windowed = session.momentum(K, ist0=2, ist1=5)

    assert full.pmat.shape == (3, len(full.energies), len(full.energies))
    assert windowed.pmat.shape == (3, 4, 4)
    np.testing.assert_allclose(windowed.energies, full.energies[1:5])
    np.testing.assert_allclose(windowed.pmat, full.pmat[:, 1:5, 1:5], atol=1e-10)


def test_get_momentum_matrix_along_a_path(hbn):
    """The kpath= convenience wrapper returns the same list-of-dicts shape
    the other session wrappers do, one entry per point plus a distance."""
    calc, ist0, ist1 = hbn
    points = calc.get_momentum_matrix(kpath="GK", npoints=4, label="mom_path")
    assert len(points) == 4
    assert {"k", "energies", "pmat", "distance"} <= set(points[0])
    assert points[0]["distance"] < points[-1]["distance"]
