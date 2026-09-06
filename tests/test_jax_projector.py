"""Phase 0b of the Elk-to-JAX port: the safe-K spectral projector rule.

Study §6 item 0b and §8(b) of `docs/jax_port.md`, plus the unresolved measurement
disagreement recorded in `docs/continue_here.md` §3 -- which these tests settle.

Every assertion here uses the closed-form derivative of `elkjax.reference` as its
reference, not finite differences.  `test_finite_differences_agree_with_the_closed_form`
is what earns that: it shows FD *is* reliable for this observable (a gapped window
boundary makes Tr[P M] smooth in H however degenerate the interior is), so the closed
form is cross-checked rather than merely asserted, and the naive route's disagreement
with it cannot be dismissed as FD noise.
"""

import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import elkjax  # noqa: E402  (sets jax_enable_x64 on import)
from elkjax import memory, phase0b, projector as pj, reference as ref  # noqa: E402

# Nothing here needs more than ~2 GB; the cap turns a mistake into a prompt failure
# rather than into swap on a 39 GB box.  See CLAUDE.md, "JAX port".
memory.limit_address_space(16.0)

SLOW = pytest.mark.skipif(
    not os.environ.get("ELKPY_RUN_SLOW_TESTS"),
    reason="set ELKPY_RUN_SLOW_TESTS=1 to run the n>=400 cases",
)

# a six-state spectrum with a two-fold degeneracy strictly INSIDE a window of three
ENCLOSED = np.array([-2.0, 1.0, 1.0, 3.0, 4.0, 5.0])
NOCC = 3
OBSERVABLE = np.diag(np.arange(6.0)).astype(complex)


def _loss_np(h, nocc=NOCC, m=OBSERVABLE):
    _, v = np.linalg.eigh(h)
    p = v[:, :nocc] @ v[:, :nocc].conj().T
    return float(np.real(np.trace(p @ m)))


def _jax_modes(fn):
    return float(jax.jvp(fn, (0.0,), (1.0,))[1]), float(jax.grad(fn)(0.0))


def _directional(h, d, nocc=NOCC, tol=1e-11, m=OBSERVABLE):
    hj, dj, mj = jnp.array(h), jnp.array(d), jnp.array(m)
    loss = lambda p: jnp.real(jnp.trace(p @ mj))
    naive = lambda t: loss(pj.naive_hard_window_projector(hj + t * dj, nocc))
    safe = lambda t: loss(pj.hard_window_projector(hj + t * dj, nocc, tol))
    return _jax_modes(naive), _jax_modes(safe)


def test_x64_is_enabled():
    # An all-electron spectrum spans ~2500 Ha; float32 is not an option (study §1).
    assert jnp.zeros(1).dtype == jnp.float64


def test_the_realistic_assembly_splits_the_degeneracy():
    # A diagonal test matrix is degenerate bitwise; a real Hamiltonian is not.  Unit
    # tests reach for the first shape and real work meets the second (study §8b).
    w = np.linalg.eigvalsh(ref.hermitian_from_spectrum(ENCLOSED, 0))
    assert 0.0 < w[2] - w[1] < 1e-13


def test_finite_differences_agree_with_the_closed_form():
    """FD is trustworthy for this observable, which is what makes the rest decisive."""
    occ = ref.hard_occupations(6, NOCC)
    for seed in range(3):
        h = ref.hermitian_from_spectrum(ENCLOSED, seed)
        for k in range(4):
            d = ref.random_hermitian_direction(6, 100 + k)
            exact = ref.directional_derivative(h, d, OBSERVABLE, occ)
            for step in (1e-4, 1e-5, 1e-6):
                fd = (_loss_np(h + step * d) - _loss_np(h - step * d)) / (2 * step)
                assert abs(fd - exact) < 1e-6 * max(abs(exact), 1.0)


def test_safe_rule_matches_the_closed_form_with_the_multiplet_enclosed():
    occ = ref.hard_occupations(6, NOCC)
    for seed in range(3):
        h = ref.hermitian_from_spectrum(ENCLOSED, seed)
        for k in range(6):
            d = ref.random_hermitian_direction(6, 100 + k)
            exact = ref.directional_derivative(h, d, OBSERVABLE, occ)
            (_, _), (fwd, rev) = _directional(h, d)
            assert abs(fwd - exact) < 1e-11 * max(abs(exact), 1.0)
            assert abs(rev - exact) < 1e-11 * max(abs(exact), 1.0)


def test_naive_rule_is_wrong_with_the_multiplet_enclosed():
    """The negative test, required to FAIL, per study §6's Phase 1 pattern.

    This is the case `docs/continue_here.md` §3 reported as safe.  It is not: that check
    used a single real diagonal direction in reverse mode only.  A general Hermitian
    direction breaks it, and forward and reverse mode disagree with each other -- which
    for a scalar-in, scalar-out function is by itself proof that one of them is wrong.
    """
    occ = ref.hard_occupations(6, NOCC)
    worst_forward = worst_reverse = worst_mode_split = 0.0
    for seed in range(3):
        h = ref.hermitian_from_spectrum(ENCLOSED, seed)
        for k in range(8):
            d = ref.random_hermitian_direction(6, 100 + k)
            exact = ref.directional_derivative(h, d, OBSERVABLE, occ)
            scale = max(abs(exact), 1e-30)
            (nf, nr), _ = _directional(h, d)
            worst_forward = max(worst_forward, abs(nf - exact) / scale)
            worst_reverse = max(worst_reverse, abs(nr - exact) / scale)
            worst_mode_split = max(worst_mode_split, abs(nf - nr) / scale)
    assert worst_forward > 1.0        # measured 1.1e+1 over 3 seeds x 21 directions
    assert worst_reverse > 1.0        # measured 3.4e+0
    assert worst_mode_split > 1.0


def test_the_benign_direction_that_misled_the_earlier_check():
    """A single real diagonal entry: naive REVERSE mode is nearly right here.

    Pinned because it explains the earlier disagreement rather than merely overruling
    it.  The reverse-mode agreement is a rounding accident, so it is asserted only
    relative to forward mode on the same matrix -- an absolute bound there would pin
    this BLAS build, not a fact.  Measured: reverse 1e-8 relative, forward 1.3e-2.
    """
    occ = ref.hard_occupations(6, NOCC)
    h = ref.hermitian_from_spectrum(ENCLOSED, 0)
    d = np.diag([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype(complex)
    exact = ref.directional_derivative(h, d, OBSERVABLE, occ)
    (nf, nr), (_, sr) = _directional(h, d)
    assert abs(nf - exact) / abs(exact) > 1e-3
    assert abs(nr - exact) < 0.1 * abs(nf - exact)
    assert abs(sr - exact) / abs(exact) < 1e-11


def test_why_the_terms_do_not_cancel_bitwise():
    """The mechanism, not just the symptom.

    The earlier check expected the two divergent contributions to cancel bitwise,
    because they are exact negatives.  They are -- but only if
    ``A = v^H dH v`` is *bitwise* Hermitian, and it is not: JAX's ``_eigh_jvp_rule``
    forms ``vdag_adot_v`` with no symmetrisation, so the surviving term is
    ``(A_ij - conj(A_ji)) / (lambda_i - lambda_j)``, a rounding error over a rounding
    error.  Measured here: the ratio is O(0.1), which is the size of the observed
    failure.
    """
    for seed in range(3):
        h = ref.hermitian_from_spectrum(ENCLOSED, seed)
        d = ref.random_hermitian_direction(6, 101)
        evals = np.asarray(jnp.linalg.eigvalsh(jnp.array(h)))
        _, v = np.linalg.eigh(h)
        a = np.asarray(jnp.array(v).conj().T @ jnp.array(d) @ jnp.array(v))
        asymmetry = np.linalg.norm(a - a.conj().T)
        assert asymmetry < 1e-14                       # A is Hermitian to rounding
        assert asymmetry > 0.0                         # ... but not bitwise
        assert asymmetry / (evals[2] - evals[1]) > 1e-2


def test_multiplet_summed_eigenvalues_are_exact_and_free():
    """Study §8(b)'s first row: no 1/(lambda_i - lambda_j) appears, so nothing to fix.

    The control that separates "AD through eigh is broken" from "this observable is".
    """
    occ = ref.hard_occupations(6, NOCC)
    h = ref.hermitian_from_spectrum(ENCLOSED, 0)
    d = ref.random_hermitian_direction(6, 101)
    exact = ref.multiplet_sum_derivative(h, d, occ)
    hj, dj = jnp.array(h), jnp.array(d)
    fn = lambda t: jnp.sum(jnp.linalg.eigvalsh(hj + t * dj)[:NOCC])
    fwd, rev = _jax_modes(fn)
    assert abs(fwd - exact) < 1e-12 * max(abs(exact), 1.0)
    assert abs(rev - exact) < 1e-12 * max(abs(exact), 1.0)


def test_padding_block_is_a_nan_for_the_naive_rule_and_finite_for_the_safe_one():
    """Study §8(b): do not manufacture degeneracies in the padding.

    H_pad = E_big * 1 is bitwise degenerate at every k-point by construction, which a
    Phase 1 developer meets on day one with a perfectly non-degenerate physical block.
    """
    result = phase0b.experiment_d()
    assert np.isnan(result["naive_rev"])
    assert abs(result["safe_rev"] - result["exact"]) < 1e-11 * abs(result["exact"])


def test_reassembly_jitter_moves_the_naive_gradient_and_not_the_safe_one():
    """Perturb H at 1e-16: the answer moves by 1e-16, the naive gradient by ~60%."""
    result = phase0b.experiment_c()
    assert result["spread_exact"] < 1e-12
    assert result["spread_naive"] > 1e-2      # measured 6.4e-1
    assert result["spread_safe"] < 1e-12      # measured 6.4e-15


def test_smeared_occupations_need_the_rule_too():
    rows = phase0b.experiment_e()
    rel = lambda a, b: abs(a - b) / max(abs(b), 1e-30)
    assert max(rel(r["naive_rev"], r["exact"]) for r in rows) > 1e-2
    assert max(rel(r["safe_rev"], r["exact"]) for r in rows) < 1e-11


def test_a_degeneracy_across_the_window_boundary_is_refused():
    """No differentiable occupied subspace exists there, so a number would be a lie.

    The standing mitigation is the one CLAUDE.md §13 already applies to Berry
    curvature: window the whole degenerate group together, which the second half of
    this test does.
    """
    straddling = np.array([-2.0, 1.0, 1.0, 1.0, 4.0, 5.0])  # window cuts the triplet
    h = ref.hermitian_from_spectrum(straddling, 0)
    d = ref.random_hermitian_direction(6, 100)
    (_, _), (_, refused) = _directional(h, d, nocc=2, tol=1e-9)
    assert np.isnan(refused)
    (_, _), (_, enclosed) = _directional(h, d, nocc=4, tol=1e-9)
    assert np.isfinite(enclosed)
    exact = ref.directional_derivative(h, d, OBSERVABLE, ref.hard_occupations(6, 4))
    assert abs(enclosed - exact) < 1e-11 * max(abs(exact), 1.0)


def test_the_default_tolerance_resolves():
    """`hard_window_projector(h, nocc)` with `tol` defaulted must still differentiate.

    `custom_jvp` + `nondiff_argnums` is fussy about defaulted static arguments, and a
    caller who omits `tol` is the common case.
    """
    h = jnp.array(ref.hermitian_from_spectrum(np.array([-2.0, -1.0, 0.5, 2.0, 4.0, 5.0]), 0))
    d = jnp.array(ref.random_hermitian_direction(6, 1))
    m = jnp.array(OBSERVABLE)
    fn = lambda t: jnp.real(jnp.trace(pj.hard_window_projector(h + t * d, NOCC) @ m))
    assert np.isfinite(float(jax.grad(fn)(0.0)))


def test_the_sign_projector_refuses_an_ungapped_window():
    """`sign_projector` fails SILENTLY if mu lands inside a multiplet, so it is guarded.

    Nothing in the Newton-Schulz iteration notices that mu is not in a gap; the returned
    operator is simply not the projector onto anything.  Same role as
    `elkpy.parsers.symmetry.check_window_gap`.
    """
    h = jnp.array(ref.hermitian_from_spectrum(ENCLOSED, 0))
    assert pj.check_sign_window(h, NOCC) > 1.0          # the gapped window is fine
    with pytest.raises(ValueError, match="inside a multiplet"):
        pj.check_sign_window(h, 2)                      # ... this one splits the pair
    with pytest.raises(ValueError, match="not a proper window"):
        pj.check_sign_window(h, 0)


def test_kernel_agrees_with_the_numpy_reference():
    evals = np.linalg.eigvalsh(ref.hermitian_from_spectrum(ENCLOSED, 0))
    occ, docc = ref.fermi_dirac(evals, 2.0, 0.3)
    expected = ref.divided_difference_kernel(evals, occ, docc, 1e-9)
    got = np.asarray(pj.divided_difference_kernel(
        jnp.array(evals), jnp.array(occ), jnp.array(docc), 1e-9))
    assert np.allclose(got, expected, rtol=0, atol=1e-13)


def test_documented_memory_figures():
    """Pins the numbers CLAUDE.md's "JAX port" section quotes."""
    assert memory.matrix_bytes(3000) == 144 * 1000 ** 2  # 144 MB per complex128 matrix
    total = memory.kpoint_set_bytes(**memory.PRODUCTION_SHAPE) / memory.GB
    assert 26.0 < total < 27.0        # H + S over 100 k-points, in GiB
    assert memory.kpoint_set_bytes(**memory.LOCAL_BUDGET) / memory.GB < 0.3


@SLOW
def test_error_grows_as_the_enclosed_pair_closes():
    """Study §8(b) tabulates this for individual eigenvalues; here it is for P."""
    rows = phase0b.experiment_g()
    rel = lambda r, key: abs(r[key] - r["exact"]) / abs(r["exact"])
    assert rel(rows[0], "naive_rev") < 1e-9     # split 1e-4: naive is fine
    assert rel(rows[-1], "naive_rev") > 1e-2    # bitwise: measured 2.3e-1
    assert all(rel(r, "safe_rev") < 1e-10 for r in rows)


@SLOW
def test_tolerance_plateau_under_smearing():
    """The gradient must be flat over at least two decades of `tol` (study §8b)."""
    result = phase0b.experiment_f()
    values = [v for _, v in result["sweep"]]
    plateau = values[1:]
    spread = (max(plateau) - min(plateau)) / abs(np.mean(plateau))
    assert spread < 1e-10
    assert abs(plateau[0] - result["exact"]) < 1e-9 * abs(result["exact"])


@SLOW
def test_failure_mode_depends_on_the_eigensolver():
    """LAPACK and XLA do not agree on the splitting of the same engineered pair.

    Which failure mode the naive route then takes -- a wrong number, or a NaN when the
    backend happens to return the pair bitwise equal -- follows from that, so only the
    disagreement itself and the failure are asserted.  Measured on this build: XLA
    splits below LAPACK on every size and returns exactly zero at n=1000.
    """
    rows = phase0b.experiment_h()
    assert any(r["split_xla"] != r["split_lapack"] for r in rows)
    for r in rows:
        rel = abs(r["naive_rev"] - r["exact"]) / abs(r["exact"])
        assert np.isnan(r["naive_rev"]) or rel > 1e-3
        assert abs(r["safe_rev"] - r["exact"]) < 1e-9 * abs(r["exact"])


# ---------------------------------------------------------------------------
# Phase 1i: the smeared kernel's exact oracle, and the self-consistent Fermi level.
# The real-matrix half of this is `tests/test_calculation_lapw_smearing.py`.
# ---------------------------------------------------------------------------


def test_the_exact_fermi_kernel_is_f_prime_on_the_diagonal():
    """The closed form must reduce to f' exactly, or it is not the same function.

    This is the whole claim of `fermi_divided_difference_kernel`: one analytic
    expression covering both branches, so it can arbitrate between them rather than
    being a third opinion.
    """
    evals = np.array([-0.5, -1e-9, 1e-9, 0.3, 5.0])
    for width in (1e-3, 1e-2, 1e-1):
        kernel = ref.fermi_divided_difference_kernel(evals, 0.0, width)
        _, docc = ref.fermi_dirac(evals, 0.0, width)
        assert np.abs(np.diag(kernel) - docc).max() < 1e-14 * max(np.abs(docc).max(), 1.0)


def test_the_direct_quotient_loses_accuracy_as_the_smearing_widens():
    """Cancellation in `(f_i - f_j)` costs ~eps*w/dlambda, so BROADER smearing is worse.

    That is the opposite of the intuition that a smoother occupation is safer, and it
    is why `tol` cannot be a fixed number: what it has to beat is set by the width as
    well as by the splitting.  Asserted as a growth rate, not a threshold.
    """
    split = 1e-15
    evals = np.array([-1.0, -0.5, -0.5 * split, 0.5 * split, 0.5, 1.0])
    errors = []
    for width in (1e-3, 1e-2, 1e-1):
        exact = ref.fermi_divided_difference_kernel(evals, 0.0, width)[2, 3]
        f, _ = ref.fermi_dirac(evals, 0.0, width)
        quotient = (f[2] - f[3]) / (evals[2] - evals[3])
        errors.append(abs(quotient - exact) / abs(exact))
    assert errors[2] > 30 * errors[0], errors        # measured 8.9e-5, 8.0e-4, 2.1e-2
    assert errors[0] < 1e-3 and errors[2] > 1e-2, errors


def test_the_fermi_level_rule_matches_finite_differences():
    """Study §8(b)'s dmu/deps_i = w_i f'_i / sum_j w_j f'_j, tested for the first time.

    `docs/continue_here.md` §3 lists the closed form as untested.  The primal is a
    bisection and is never differentiated -- unrolling a root find is the Phase 0a
    lesson about unrolled solvers in miniature -- so the JVP is the whole content.
    """
    n = 12
    h = ref.hermitian_from_spectrum(np.linspace(-1.0, 1.0, n), 3)
    hj = jnp.asarray(h)
    for j in range(3):
        d = ref.random_hermitian_direction(n, 700 + j)
        dj = jnp.asarray(d)
        ad = float(jax.jvp(lambda t: pj.fermi_level(hj + t * dj, 6.0, 0.1),
                           (0.0,), (1.0,))[1])
        step = 1e-5
        fd = (ref.fermi_level(np.linalg.eigvalsh(h + step * d), 6.0, 0.1)
              - ref.fermi_level(np.linalg.eigvalsh(h - step * d), 6.0, 0.1)) / (2 * step)
        assert abs(ad - fd) < 1e-7 * max(abs(fd), 1e-3), (ad, fd)


def test_a_multiplet_at_the_fermi_level_does_not_break_the_mu_rule():
    """The rule uses A_jj, which is gauge-dependent -- but only its f'-weighted SUM.

    Inside a degenerate group f' is constant, so that sum is f' Tr[A] over the group,
    invariant under the arbitrary unitary `eigh` picks there.  Tested by rotating the
    degenerate block explicitly and requiring dmu to be unchanged; a per-state
    quantity, which is what the naive eigenvalue derivative would need, is NOT
    invariant under the same rotation, and that is asserted alongside so the check
    cannot pass on a rotation that happens to do nothing.
    """
    evals = np.array([-1.0, 0.0, 0.0, 0.0, 1.0, 2.0])
    h = ref.hermitian_from_spectrum(evals, 4)
    d = ref.random_hermitian_direction(6, 77)
    lam, v = np.linalg.eigh(h)
    mu = ref.fermi_level(lam, 3.0, 0.05)          # NOT 0: the multiplet is only half in
    _, docc = ref.fermi_dirac(lam, mu, 0.05)

    def dmu(vecs):
        a = np.real(np.diag(vecs.conj().T @ d @ vecs))
        return float(np.sum(docc * a) / np.sum(docc)), a

    plain, a_plain = dmu(v)
    block = np.eye(6, dtype=complex)
    block[1:4, 1:4] = np.linalg.qr(ref.random_hermitian_direction(3, 78) + 3j * np.eye(3))[0]
    rotated, a_rotated = dmu(v @ block)
    assert abs(plain) > 1e-6, plain
    assert abs(rotated - plain) < 1e-12 * abs(plain)
    assert np.abs(a_rotated[1:4] - a_plain[1:4]).max() > 1e-3, "the rotation did nothing"
    # and the AD rule reproduces it on the same matrix
    ad = float(jax.jvp(lambda t: pj.fermi_level(jnp.asarray(h) + t * jnp.asarray(d),
                                                3.0, 0.05), (0.0,), (1.0,))[1])
    assert abs(ad - plain) < 1e-10 * abs(plain), (ad, plain)


def _loss_fixed_n(h, nelec, width, m):
    """Tr[P M] with mu RE-SOLVED for this matrix -- the constraint, not the formula."""
    evals, evecs = np.linalg.eigh(np.asarray(h))
    f, _ = ref.fermi_dirac(evals, ref.fermi_level(evals, nelec, width), width)
    return float(np.real(np.trace(((evecs * f) @ evecs.conj().T) @ m)))


def test_fixed_number_adds_a_term_fixed_mu_omits():
    """`fixed_number_projector` is the composition, and the extra term is not small.

    At a level pinned to the Fermi energy the chemical-potential response is of the
    same order as the whole fixed-mu derivative, so a test that only checked "AD agrees
    with the closed form" could pass with the term dropped from BOTH.  Hence the second
    reference: a central difference of a loss whose mu is RE-SOLVED at each displaced
    matrix, which carries the constraint rather than the formula being tested.
    """
    n = 12
    h = ref.hermitian_from_spectrum(np.linspace(-1.0, 1.0, n), 5)
    hj, mj = jnp.asarray(h), jnp.asarray(np.diag(np.linspace(-1, 1, n)).astype(complex))
    m = np.asarray(mj)
    nelec, width, step = 6.0, 0.1, 1e-5
    separated = False
    for j in range(3):
        d = ref.random_hermitian_direction(n, 900 + j)
        dj = jnp.asarray(d)
        loss = lambda t: jnp.real(jnp.trace(
            pj.fixed_number_projector(hj + t * dj, nelec, width) @ mj))
        ad = float(jax.grad(loss)(0.0))
        exact = float(np.real(np.trace(
            ref.dprojector_fermi(h, d, ref.fermi_level(np.linalg.eigvalsh(h),
                                                       nelec, width),
                                width, dmu="selfconsistent") @ m)))
        fixed_mu = float(np.real(np.trace(
            ref.dprojector_fermi(h, d, ref.fermi_level(np.linalg.eigvalsh(h),
                                                       nelec, width), width) @ m)))
        fd = (_loss_fixed_n(h + step * d, nelec, width, m)
              - _loss_fixed_n(h - step * d, nelec, width, m)) / (2 * step)
        assert abs(ad - exact) < 1e-10 * max(abs(exact), 1.0), (ad, exact)
        assert abs(float(jax.jvp(loss, (0.0,), (1.0,))[1]) - ad) < 1e-12 * max(abs(ad), 1.0)
        assert abs(fd - ad) < 1e-5 * max(abs(ad), 1.0), (fd, ad)
        if abs(fixed_mu - exact) > 0.05 * abs(exact):
            separated = True
    assert separated, "the mu term is negligible here, so this fixture proves nothing"


def test_a_gapped_spectrum_refuses_a_self_consistent_fermi_level():
    """sum f' -> 0 in a gap: the constraint stops determining mu and dmu is 0/0.

    A physical singularity, not a numerical one, so the honest answer is a refusal --
    the same host-side shape as `check_sign_window` and `occupied_window`.
    """
    gapped = jnp.asarray(ref.hermitian_from_spectrum(
        np.array([-2.0, -1.9, -1.8, 1.8, 1.9, 2.0]), 11))
    with pytest.raises(ValueError, match="does not determine the Fermi level"):
        pj.check_fermi_level_determined(gapped, 3.0, 1e-3)
    # the same spectrum at a width comparable to the gap is determined again
    mu, response = pj.check_fermi_level_determined(gapped, 3.0, 1.0)
    assert response > 1e-2 and -1.9 < mu < 1.9


# ---------------------------------------------------------------------------
# Phase 1i, second pass: the cancellation-free kernel INSIDE the JVP.
#
# `smeared_projector` now uses `projector.fermi_kernel`, which shares its closed
# form with `reference.fermi_divided_difference_kernel`.  That makes the NumPy
# oracle an INDEPENDENT IMPLEMENTATION but no longer an independent FORMULA, so
# every test below carries either a central finite difference (legitimate here --
# P = f(H) is smooth, so there is no branch exchange to average over) or
# `direct_quotient_projector`, which keeps the literal quotient and is therefore
# still formula-independent wherever it is accurate.
# ---------------------------------------------------------------------------


def test_the_jax_kernel_matches_the_numpy_oracle_and_stays_finite():
    """Two implementations of one closed form, over an all-electron energy spread.

    The spread matters: `cosh(u)` overflows at |u| ~ 710, and a real LAPW spectrum
    runs from core levels thousands of Ha below mu to basis states thousands above,
    so at Elk's default swidth |u| reaches 1e6.  `jnp.where` evaluates both branches,
    so an unguarded `inf/inf` in the discarded one would reach the gradient as NaN.
    """
    evals = np.array([-3000.0, -0.5, -1e-15, 1e-15, 0.3, 5.0, 4000.0])
    for width in (1e-3, 1e-2, 1e-1):
        got = np.asarray(pj.fermi_kernel(jnp.asarray(evals), 0.0, width))
        assert np.isfinite(got).all(), width
        expected = ref.fermi_divided_difference_kernel(evals, 0.0, width)
        assert np.abs(got - expected).max() < 1e-14 * max(1.0 / width, 1.0)


def test_the_tolerance_is_now_inert_for_smeared_occupations():
    """Not a plateau -- the gradient does not depend on `tol` AT ALL.

    Study §8(b) asks for flatness over two decades; with the closed form in the JVP
    there is nothing left for `tol` to select, so the assertion is equality to the last
    bit across fourteen decades including 0 and a value larger than the whole spectrum.
    That is the difference between a threshold tuned to be harmless and a threshold
    removed, and it is why §1i's 2e-9 floor on real Si matrices goes away.
    """
    evals = np.array([-2.0, -1.0, -1e-15, 1e-15, 1.0, 2.0])
    h = jnp.asarray(ref.hermitian_from_spectrum(evals, 12))
    d = jnp.asarray(ref.random_hermitian_direction(6, 13))
    m = jnp.asarray(OBSERVABLE)
    values = []
    for tol in (0.0, 1e-16, 1e-12, 1e-9, 1e-6, 1e-3, 1e2):
        fn = lambda t: jnp.real(jnp.trace(
            pj.smeared_projector(h + t * d, 0.0, 1e-2, tol) @ m))
        values.append(float(jax.grad(fn)(0.0)))
    assert abs(values[0]) > 1e-6, values
    assert all(v == values[0] for v in values), values


def test_the_closed_form_beats_the_literal_quotient_at_a_roundoff_splitting():
    """The two routes, arbitrated by a finite difference rather than by each other.

    At a 1e-15 splitting and Elk's default width the literal quotient's numerator is
    bitwise zero, so it misses that pair's whole contribution; the closed form does
    not.  Central FD is the independent judge -- it is legitimate here because
    P = f(H) is a smooth matrix function, so unlike §1h there is no branch exchange
    to average over.

    Compared as MATRICES, not as a scalar loss.  A trace against one observable dilutes
    a single kernel entry among all the others -- measured, the same comparison through
    Tr[dP M] shows only 2.6e-5 -- so a scalar test would understate a failure that is
    100% of the entry responsible for it.  That dilution is itself the reason §1i's real
    silicon derivative was off by 3.7e-3 rather than by 100%.
    """
    evals = np.array([-2.0, -1.0, -0.5e-15, 0.5e-15, 1.0, 2.0])
    h = ref.hermitian_from_spectrum(evals, 14)
    d = ref.random_hermitian_direction(6, 15)
    hj, dj = jnp.asarray(h), jnp.asarray(d)
    mu, width, step = 0.0, 1e-3, 1e-7

    def projector(matrix):
        lam, vecs = np.linalg.eigh(matrix)
        f, _ = ref.fermi_dirac(lam, mu, width)
        return (vecs * f) @ vecs.conj().T

    fd = (projector(h + step * d) - projector(h - step * d)) / (2 * step)
    stable = np.asarray(jax.jvp(
        lambda t: pj.smeared_projector(hj + t * dj, mu, width, 0.0), (0.0,), (1.0,))[1])
    literal = np.asarray(jax.jvp(
        lambda t: pj.direct_quotient_projector(hj + t * dj, mu, width),
        (0.0,), (1.0,))[1])
    scale = np.linalg.norm(fd)
    assert scale > 1.0, scale
    assert np.linalg.norm(stable - fd) < 1e-7 * scale        # measured 2.6e-9
    assert np.linalg.norm(literal - fd) > 1e3 * np.linalg.norm(stable - fd)
