"""The Cholesky-reduced eigensolve wired to the safe-K projector rule, on REAL
Elk matrices (JAX port, Phase 1f -- docs/jax_port_phase1.md).

What this closes. `elkjax.projector`'s safe-K rule was written for Phase 0b and
had only ever been exercised on a SYNTHETIC overlap with a prescribed condition
number; `docs/continue_here.md` section 3 calls that the biggest hole left in
Phase 0. `elkjax.hamiltonian.first_variational_eigenvalues` meanwhile closed
with a plain `eigvalsh`, so at a multiplet it failed exactly the way 0b
describes. Patch 0013's LAPW export supplies what was missing: real H and O at
any k, and therefore a real kappa(O) to set the tolerance from.

Why bulk silicon at Gamma. The disputed configuration of Phase 0b is a hard
occupied window with a degenerate multiplet fully ENCLOSED and the window
boundary gapped. Diamond Si at Gamma is that configuration by symmetry rather
than by construction: the Gamma_25' valence triplet sits inside the four
occupied first-variational bands, and the boundary to the conduction band is
open by ~0.09 Ha. A generic k on the same ground state is the control -- no
degeneracy, so both AD routes must agree there or the comparison means nothing.

Skipped without the elk binary, and without jax.
"""

import os

import numpy as np
import pytest

from elkpy import config
from elkpy.structure import Structure

jax = pytest.importorskip("jax")
elkjax = pytest.importorskip("elkjax")

from elkjax import hamiltonian as ham, memory, projector as pj, reference as ref  # noqa: E402
import jax.numpy as jnp  # noqa: E402

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)

memory.limit_address_space(8.0)

SI_AVEC = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
SI_SPECIES = {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}
GAMMA = (0.0, 0.0, 0.0)
GENERIC = (0.1, 0.2, 0.05)       # no symmetry, no degeneracy


@pytest.fixture(scope="module")
def silicon(tmp_path_factory):
    """One converged ground state, exported at Gamma and at a generic k.

    Both exports come from the SAME session, so the two cases differ only in
    the k-point -- the point of the control.
    """
    from elkpy.calculation import Calculation

    workdir = tmp_path_factory.mktemp("lapw_projector") / "si"
    calculation = Calculation(
        structure=Structure(avec=SI_AVEC, species=SI_SPECIES),
        workdir=workdir, ngridk=(4, 4, 4), rgkmax=7.0)
    calculation.ensure_ground_state()
    efermi = float((calculation.workdir / "EFERMI.OUT").read_text().split()[0])
    with calculation.eigenstate_session() as session:
        out = {"gamma": session.lapw_problem(GAMMA),
               "generic": session.lapw_problem(GENERIC)}
    for export in out.values():
        export["_nocc"] = ham.occupied_band_count(export["evalfv"], efermi)
    return out


def _reduced(export):
    """H~ at the exported k, from the JAX pipeline, plus O and the tolerance."""
    h, o = ham.eigenproblem_at(export, np.asarray(export["vkc"]))
    reduced, _ = ham.cholesky_reduce(h, o)
    return np.asarray(reduced), np.asarray(o), ham.projector_tolerance(reduced, o)


def _observable(n):
    return np.diag(np.linspace(-1.0, 1.0, n)).astype(complex)


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


# --------------------------------------------------------------- the premise


def test_gamma_encloses_a_real_multiplet_with_a_gapped_boundary(silicon):
    """The fixture's premise, asserted rather than assumed.

    Everything below is a statement about a hard window with a degeneracy
    inside it and a gap at its edge. If a future Elk version, species file or
    lattice constant moved either, every other test here would keep passing
    while measuring nothing -- so this fails first and by name.
    """
    export = silicon["gamma"]
    nocc = export["_nocc"]
    assert nocc == 4                        # Si: 8 valence electrons, nspinor=1
    reduced, _, tol = _reduced(export)
    evals = np.asarray(jnp.linalg.eigvalsh(reduced))

    inside = np.diff(evals[:nocc]).min()
    boundary = evals[nocc] - evals[nocc - 1]
    assert inside < 1e-8, "Gamma_25' is no longer degenerate to the level tested"
    assert boundary > 1e-2, "the occupied window boundary is no longer gapped"
    # and the control really is a control
    generic = silicon["generic"]
    g_reduced, _, _ = _reduced(generic)
    g_evals = np.asarray(jnp.linalg.eigvalsh(g_reduced))
    assert np.diff(g_evals[:generic["_nocc"] + 1]).min() > 1e-2


# --------------------------------------------------------------- A: forward


@pytest.mark.parametrize("case", ["gamma", "generic"])
def test_projector_reproduces_elks_own_occupied_subspace(silicon, case):
    """The study's Phase 1 forward criterion, as a projector rather than
    coefficients.

    `evecfv` is arbitrary inside a degenerate multiplet, so comparing columns
    would fail at Gamma for a gauge reason and mean nothing. Elk's normalised
    eigenvectors map into the reduced basis as Y = L^H C, orthonormal because
    C^H O C = 1, and Y Y^H is the projector onto exactly the subspace Elk
    found.

    Phase 0's first carried-forward finding is that a green gradient test does
    not validate a transcription. This is the forward check that stands beside
    the gradient checks below.
    """
    export = silicon[case]
    nocc = export["_nocc"]
    reduced, _, tol = _reduced(export)
    p_jax = np.asarray(pj.hard_window_projector(jnp.asarray(reduced), nocc, tol))
    p_elk = ham.elk_occupied_projector(export, nocc)
    assert np.linalg.norm(p_jax - p_elk) < 1e-8
    assert np.linalg.norm(p_jax @ p_jax - p_jax) < 1e-10
    assert np.linalg.norm(p_jax - p_jax.conj().T) < 1e-12


# --------------------------------------------------- B: the rule on real H~


@pytest.mark.parametrize("case", ["gamma", "generic"])
def test_safe_rule_against_the_closed_form_on_elks_own_matrix(silicon, case):
    """d/dt Tr[P(H~ + t D) M] along random Hermitian D, against Daleckii-Krein.

    The reference is the closed form, not finite differences: study section
    8(b) measures FD of a sorted spectrum returning the branch average at a
    multiplet. Central FD is carried alongside anyway, because for the
    PROJECTOR with a gapped boundary the loss is smooth in H and FD is
    therefore a legitimate second opinion -- the control that separates "AD is
    broken" from "the test is broken".

    Forward and reverse mode are both checked and must agree: for a
    scalar-in, scalar-out function they are the same number, so a disagreement
    is proof on its own (Phase 0's carried-forward finding, which cost a wrong
    conclusion to learn).
    """
    export = silicon[case]
    nocc = export["_nocc"]
    reduced, _, tol = _reduced(export)
    n = reduced.shape[0]
    m = _observable(n)
    occ = ref.hard_occupations(n, nocc)
    hj, mj = jnp.asarray(reduced), jnp.asarray(m)

    def loss(p):
        return jnp.real(jnp.trace(p @ mj))

    def np_loss(matrix):
        _, v = np.linalg.eigh(matrix)
        return float(np.real(np.trace(v[:, :nocc] @ v[:, :nocc].conj().T @ m)))

    for j in range(3):
        d = ref.random_hermitian_direction(n, 800 + j)
        dj = jnp.asarray(d)
        exact = ref.directional_derivative(reduced, d, m, occ, tol=tol)
        safe = lambda t: loss(pj.hard_window_projector(hj + t * dj, nocc, tol))
        forward = float(jax.jvp(safe, (0.0,), (1.0,))[1])
        reverse = float(jax.grad(safe)(0.0))
        step = 1e-5
        fd = (np_loss(reduced + step * d) - np_loss(reduced - step * d)) / (2 * step)

        assert _rel(reverse, exact) < 1e-9
        assert _rel(forward, reverse) < 1e-12
        assert _rel(fd, exact) < 1e-5


def test_naive_ad_fails_at_the_real_multiplet_and_not_at_the_control(silicon):
    """The rule is NEEDED, measured on Elk's own matrix rather than argued.

    At Gamma the enclosed triplet is split at the level of the assembly's own
    rounding, so JAX's default eigenvector rule divides by it; at the generic
    k every gap is O(1e-2) Ha and there is nothing to divide by. Requiring the
    naive route to FAIL at Gamma and to SUCCEED at the control is what makes
    this a measurement of the rule rather than of the fixture: a test that only
    demanded the safe route be accurate would pass just as well if the hazard
    had quietly gone away.

    Which face the failure wears -- a large finite number or a NaN -- is the
    eigensolver's choice, not a property of the bug (Phase 0b experiment H),
    so both are accepted.
    """
    results = {}
    for case in ("gamma", "generic"):
        export = silicon[case]
        nocc = export["_nocc"]
        reduced, _, tol = _reduced(export)
        n = reduced.shape[0]
        m = _observable(n)
        occ = ref.hard_occupations(n, nocc)
        hj, mj = jnp.asarray(reduced), jnp.asarray(m)
        worst = 0.0
        for j in range(3):
            d = ref.random_hermitian_direction(n, 900 + j)
            dj = jnp.asarray(d)
            exact = ref.directional_derivative(reduced, d, m, occ, tol=tol)
            naive = lambda t: jnp.real(jnp.trace(
                pj.naive_hard_window_projector(hj + t * dj, nocc) @ mj))
            value = float(jax.grad(naive)(0.0))
            worst = np.inf if not np.isfinite(value) else max(worst, _rel(value, exact))
        results[case] = worst
    assert results["gamma"] > 1e-3, (
        f"naive AD was accurate at the multiplet ({results['gamma']:.2e}): the "
        "hazard this rule exists for is no longer being exercised")
    assert results["generic"] < 1e-6


# ------------------------------------------------------- C: the k-derivative


@pytest.mark.parametrize("case", ["generic", "gamma"])
def test_k_gradient_through_the_projector(silicon, case):
    """The whole pipeline differentiated in k, at both points.

    The reference composes two exact pieces: dH~/dk from a matrix-valued `jvp`
    -- legitimate because every step from `match` through the Cholesky is
    analytic in k, so its tangent carries no degeneracy -- fed to the closed
    form for dP. Central FD of the same loss is the independent control.

    Gamma is the case that could not be run at all until `match` was rewritten
    through solid harmonics: its basis contains G+k = 0, where the harmonic's
    direction is undefined and |G+k| is sqrt' at zero. It is also the only case
    that exercises what this whole file is about, since the multiplet is here
    -- an INDIVIDUAL eigenvalue's k-derivative does not exist inside the
    Gamma_25' triplet, while the occupied projector's does.
    """
    export = silicon[case]
    nocc = export["_nocc"]
    kc = np.asarray(export["vkc"])
    reduced, _, tol = _reduced(export)
    n = reduced.shape[0]
    m = _observable(n)
    occ = ref.hard_occupations(n, nocc)
    mj = jnp.asarray(m)

    def reduced_at(kvec):
        return ham.cholesky_reduce(*ham.eigenproblem_at(export, kvec))[0]

    dk = np.array([0.3, -0.5, 0.8])
    dk /= np.linalg.norm(dk)
    dh = np.asarray(jax.jvp(reduced_at, (jnp.asarray(kc),), (jnp.asarray(dk),))[1])
    assert np.isfinite(dh).all()
    dh = 0.5 * (dh + dh.conj().T)
    exact = ref.directional_derivative(reduced, dh, m, occ, tol=tol)

    def safe(t):
        p = pj.hard_window_projector(
            reduced_at(jnp.asarray(kc) + t * jnp.asarray(dk)), nocc, tol)
        return jnp.real(jnp.trace(p @ mj))

    reverse = float(jax.grad(safe)(0.0))
    forward = float(jax.jvp(safe, (0.0,), (1.0,))[1])

    def np_loss(kvec):
        matrix = np.asarray(reduced_at(jnp.asarray(kvec)))
        _, v = np.linalg.eigh(matrix)
        return float(np.real(np.trace(v[:, :nocc] @ v[:, :nocc].conj().T @ m)))

    step = 1e-4
    fd = (np_loss(kc + step * dk) - np_loss(kc - step * dk)) / (2 * step)

    assert _rel(reverse, exact) < 1e-8
    assert _rel(forward, reverse) < 1e-10
    assert _rel(fd, exact) < 1e-4


def test_the_naive_route_also_fails_on_the_k_derivative_at_gamma(silicon):
    """The rule is what makes the Gamma k-derivative work, not just the pole fix.

    Removing `match`'s two poles made the TANGENT of the assembly finite at
    Gamma; it did nothing about the eigensolve downstream of it. Along a real
    k-direction, through the real multiplet, the naive projector still fails --
    which is what stops "we fixed the pole" from being mistaken for "the
    k-derivative is now fine".
    """
    export = silicon["gamma"]
    nocc = export["_nocc"]
    kc = np.asarray(export["vkc"])
    reduced, _, tol = _reduced(export)
    n = reduced.shape[0]
    m = _observable(n)
    occ = ref.hard_occupations(n, nocc)
    mj = jnp.asarray(m)

    def reduced_at(kvec):
        return ham.cholesky_reduce(*ham.eigenproblem_at(export, kvec))[0]

    dk = np.array([0.3, -0.5, 0.8])
    dk /= np.linalg.norm(dk)
    dh = np.asarray(jax.jvp(reduced_at, (jnp.asarray(kc),), (jnp.asarray(dk),))[1])
    exact = ref.directional_derivative(
        reduced, 0.5 * (dh + dh.conj().T), m, occ, tol=tol)

    def naive(t):
        p = pj.naive_hard_window_projector(
            reduced_at(jnp.asarray(kc) + t * jnp.asarray(dk)), nocc)
        return jnp.real(jnp.trace(p @ mj))

    value = float(jax.grad(naive)(0.0))
    assert (not np.isfinite(value)) or _rel(value, exact) > 1e-3


# ------------------------------------------------------------- D: the refusal


def test_occupied_window_refuses_an_unresolvable_boundary(silicon):
    """A window boundary INSIDE the triplet must be refused, not answered.

    The study asks for "an explicit refusal (not a returned number)" once the
    sampled gap falls below the tolerance, and notes that "agreement or a clean
    refusal" without a threshold at which refusal is REQUIRED is unfalsifiable.
    Cutting Si's Gamma_25' triplet is that threshold, reached without touching
    the physics: the same matrix, the same ground state, one fewer occupied
    band.

    The pair of assertions matters more than either alone. nocc=4 (the whole
    triplet inside) is accepted and nocc=3 (the boundary between two states of
    the triplet) is refused, so this measures the criterion rather than a
    blanket refusal.
    """
    export = silicon["gamma"]
    kc = np.asarray(export["vkc"])
    tol, gap = ham.occupied_window(export, kc, 4)
    assert gap > tol

    with pytest.raises(ValueError, match="not differentiable"):
        ham.occupied_window(export, kc, 3)

    with pytest.raises(ValueError, match="not a proper window"):
        ham.occupied_window(export, kc, 0)


def test_the_refusal_detects_unresolvability_and_not_symmetry(silicon):
    """The limitation of the criterion, measured and pinned.

    Elk's own matrices split the Gamma_25' triplet UNEVENLY: two of the three
    agree to ~5e-15 Ha while the third sits ~3.5e-9 Ha away -- seventy times
    ABOVE the resolution eps kappa(O) ||H~||. So cutting the triplet at nocc=3
    is refused and cutting the SAME triplet at nocc=2 is accepted, and the
    accepted one returns a finite derivative of a subspace that is not
    physically separable at all.

    The threshold is therefore necessary but not sufficient: it is a statement
    about what the arithmetic can resolve, not about what the symmetry group
    allows. A caller who knows the point group still has to window the whole
    degenerate group together (the same mitigation `docs/design.md` section 13
    already applies to Berry curvature).
    """
    export = silicon["gamma"]
    kc = np.asarray(export["vkc"])
    reduced, _, tol = _reduced(export)
    evals = np.asarray(jnp.linalg.eigvalsh(reduced))

    assert evals[3] - evals[2] < tol            # refused, below the resolution
    assert evals[2] - evals[1] > 10 * tol       # accepted, and still degenerate
    accepted_tol, gap = ham.occupied_window(export, kc, 2)
    assert gap == pytest.approx(float(evals[2] - evals[1]), rel=1e-9)
    assert gap > accepted_tol


def test_the_tolerance_is_measured_on_this_run_not_assumed(silicon):
    """kappa(O) from a dense eigvalsh, and the reduced norm -- both required.

    Phase 0b(ii) measured that section 8(b)'s cheap Cholesky-diagonal estimate
    of kappa(O) is not merely a low bound but UNINFORMATIVE on real overlaps,
    and that the norm in the tolerance should be that of the reduced matrix,
    which is several times larger than ||H||. Both are asserted here so a
    future simplification back to either shortcut fails rather than silently
    shrinking the threshold.
    """
    export = silicon["gamma"]
    reduced, o, tol = _reduced(export)
    eigenvalues = np.linalg.eigvalsh(o)
    kappa = eigenvalues.max() / eigenvalues.min()
    chol = np.linalg.cholesky(o)
    pivots = np.abs(np.diag(chol)) ** 2
    cheap = pivots.max() / pivots.min()

    assert kappa > 1e3                       # a real LAPW overlap is ill-conditioned
    assert cheap < kappa / 100               # and the cheap estimate misses it badly
    assert np.linalg.norm(reduced, 2) > 1.5 * np.linalg.norm(
        np.asarray(export["hmat"]), 2)
    assert tol == pytest.approx(
        np.finfo(np.float64).eps * kappa * np.linalg.norm(reduced, 2), rel=1e-12)
