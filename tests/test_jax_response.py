r"""`elkjax.response`: the density's derivative through a degenerate spectrum.

The rule this module tests is the one thing standing between a Kohn-Sham step
that can be evaluated and one that can be differentiated.  Bulk silicon's
valence top is a three-fold multiplet split only by roundoff, and JAX's own
`eigh` rule divides by that splitting -- so the checks here are built around an
*engineered* degeneracy rather than a generic matrix, and every one of them has
the naive rule as its negative control.

The identity everything is pinned against is

    (1/2) (dX Cbar^dagger + Cbar dX^dagger) = dP,

the tangent of the density matrix, because that is the one object this module's
two factors can be compared against something computed independently -- Phase
0b's `projector.smeared_projector`, whose rule was verified against an analytic
reference.  Without it the factor 2 on the empty block and the Re in the
accumulation are both invisible: each is a constant, and a constant commutes
with differentiation.

No Elk run and no ground state; self-skips without jax.
"""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="jax not installed; pip install -e .[jax]")

WIDTH = 1.0e-3          # Elk's default swidth
MU = 0.2140             # silicon's own Fermi level, 7.3 widths above the triplet
TRIPLET = 0.20682341    # its Gamma_25' valence top


#: Silicon's own splittings inside that triplet, measured on Elk's matrices at
#: Gamma: one pair separated by 3.4e-9 and one by 4e-15.  Both are rounding
#: errors of the assembly rather than physics, and neither is zero -- which is
#: what makes the naive rule return a plausible wrong number instead of a NaN.
SPLITTING = (0.0, 3.4e-9, 3.4e-9 + 4.0e-15)


def _degenerate_hamiltonian(n=18, seed=5, splitting=SPLITTING):
    """A Hermitian matrix whose spectrum carries silicon's own multiplet.

    Built as :math:`U\\,\\mathrm{diag}(\\lambda)\\,U^\\dagger`, so what comes
    back out of `eigh` is the intended spectrum to roundoff.  ``splitting`` is
    the triplet's internal structure: all zeros makes it exactly degenerate,
    which is the case the naive rule fails LOUDLY; the default is silicon's
    own, which is the case it fails quietly.
    """
    import elkjax                       # noqa: F401  (sets jax_enable_x64)
    import jax.numpy as jnp
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    unitary = np.linalg.qr(a)[0]
    # ascending, so that `eigh`'s own ordering puts the triplet at 1:4: one
    # deep band, the three-fold valence top, then everything empty above the
    # Fermi level -- silicon's own arrangement
    values = np.concatenate([[-0.2360], TRIPLET + np.asarray(splitting),
                             np.linspace(0.30, 1.2, n - 4)])
    return jnp.asarray(unitary @ np.diag(values) @ unitary.conj().T), values


def test_the_engineered_multiplet_is_split_the_way_silicons_is():
    """The premise of every test below, asserted rather than assumed.

    Two things have to hold for these tests to be about the hazard rather than
    about a generic matrix: the triplet must be degenerate to within a rounding
    error, and it must not be *bitwise* degenerate -- otherwise the naive rule
    would divide by an exact zero and fail loudly, which is not the failure
    mode that reaches a result.  Asking for three identical eigenvalues gives
    the same thing: `eigh` returns them split by its own arithmetic.
    """
    import jax.numpy as jnp
    for splitting in (SPLITTING, (0.0, 0.0, 0.0)):
        h, _ = _degenerate_hamiltonian(splitting=splitting)
        values = np.asarray(jnp.linalg.eigvalsh(h))
        gaps = np.diff(values[1:4])
        assert (np.abs(gaps) < 1e-8).all(), gaps
        assert (np.abs(gaps) > 0.0).all(), gaps


def test_the_pair_rule_reproduces_the_projector_tangent_at_the_multiplet():
    """Check (i): `o = I`, against Phase 0b's independently verified rule.

    `projector.smeared_projector` computes dP directly with the safe-K kernel;
    this module computes two coefficient factors and never forms P at all, so
    the two share the kernel and nothing else.  Analytic against analytic, with
    the degeneracy in play -- which is what CLAUDE.md asks for in place of a
    finite difference wherever one is.
    """
    import jax
    import jax.numpy as jnp
    from elkjax import projector, response

    h, _ = _degenerate_hamiltonian()
    n = h.shape[0]
    rng = np.random.default_rng(23)
    dh = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    dh = jnp.asarray(0.5 * (dh + dh.conj().T))
    eye = jnp.eye(n, dtype=complex)

    factors, tangents = jax.jvp(
        lambda hh, mm: response.density_factors(hh, eye, mm, n, WIDTH, 1.0,
                                                -1e30, 3),
        (h, MU), (dh, 0.37))
    mine = response.density_matrix_tangent(factors[0], tangents[1])
    reference, dreference = jax.jvp(
        lambda hh, mm: projector.smeared_projector(hh, mm, WIDTH),
        (h, MU), (dh, 0.37))

    primal = factors[1] @ factors[0].conj().T
    assert float(jnp.abs(primal - reference).max()) \
        / float(jnp.abs(reference).max()) < 1e-14
    assert float(jnp.abs(mine - dreference).max()) \
        / float(jnp.abs(dreference).max()) < 1e-12


def test_the_naive_eigenvector_rule_fails_the_same_comparison():
    """The negative control for the test above, on the same matrices.

    `projector.naive_smeared_projector` is the plain transcription JAX
    differentiates through its own `eigh` rule.  It is not slightly worse: the
    two contributions it forms are each ~1e12 and cancel to O(1), so what
    survives is the rounding error of the cancellation.  The size is asserted
    as a floor, since it depends on how the multiplet happens to split -- and
    **finiteness is asserted too**, because that is what makes this dangerous:
    a wrong number no downstream check would flag, not a NaN.
    """
    import jax
    import jax.numpy as jnp
    from elkjax import projector

    rng = np.random.default_rng(23)
    for splitting in (SPLITTING, (0.0, 0.0, 0.0)):
        h, _ = _degenerate_hamiltonian(splitting=splitting)
        n = h.shape[0]
        dh = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        dh = jnp.asarray(0.5 * (dh + dh.conj().T))

        _, safe = jax.jvp(
            lambda hh: projector.smeared_projector(hh, MU, WIDTH), (h,), (dh,))
        _, naive = jax.jvp(
            lambda hh: projector.naive_smeared_projector(hh, MU, WIDTH),
            (h,), (dh,))
        error = float(jnp.abs(naive - safe).max()) / float(jnp.abs(safe).max())
        assert error > 1e-6, (
            f"the naive rule agreed to {error:.2e}: either the multiplet is no "
            f"longer degenerate or the safe rule is no longer being used")
        assert np.isfinite(np.asarray(naive)).all(), (
            "the naive rule returned a NaN here, so this fixture is testing "
            "the loud failure rather than the quiet one")
        assert np.isfinite(np.asarray(safe)).all()


def test_the_generalized_problem_matches_a_central_difference():
    """Check (ii): `dO` is not zero, and nothing here is degenerate.

    The overlap moves with the potential in a real LAPW basis -- Phase 1k
    measured the basis response at 21% of the potential derivative -- so a rule
    verified only at `o = I` is verified on half the problem.  A generic
    spectrum is used deliberately: this test is about the `K^S` term, and a
    degeneracy would hide it behind the term the test above already covers.
    """
    import jax
    import jax.numpy as jnp
    from elkjax import response

    rng = np.random.default_rng(2)
    n, nocc = 24, 9
    make = lambda: rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    hermitian = lambda m: jnp.asarray(0.5 * (m + m.conj().T))
    h, dh, do = hermitian(make()), hermitian(make()), hermitian(make())
    b = make()
    o = jnp.asarray(b @ b.conj().T + n * np.eye(n))
    width, dmu = 0.7, 0.23

    def density_matrix(hh, oo, mm):
        values, c = response._solve(hh, oo)
        f, _ = response.occupation_weights(values, mm, width, 2.0, -1e30, 3,
                                           nocc)
        return (c * f[None, :]) @ c.conj().T

    factors, tangents = jax.jvp(
        lambda hh, oo, mm: response.density_factors(hh, oo, mm, nocc, width,
                                                    2.0, -1e30, 3),
        (h, o, 0.0), (dh, do, dmu))
    mine = response.density_matrix_tangent(factors[0], tangents[1])

    eps = 1e-5
    difference = (density_matrix(h + eps * dh, o + eps * do, eps * dmu)
                  - density_matrix(h - eps * dh, o - eps * do, -eps * dmu))
    difference = difference / (2 * eps)
    assert float(jnp.abs(mine - difference).max()) \
        / float(jnp.abs(difference).max()) < 1e-8


def test_forward_and_reverse_mode_agree():
    """Check (iii): free, and proof on its own.

    For a scalar in and a scalar out the two modes are the same number, so a
    disagreement needs no reference to be a failure.  It is also the only thing
    that exercises the transpose of the rule, which is what the adjoint solve
    of an implicit fixed point runs on every GMRES iteration.
    """
    import jax
    import jax.numpy as jnp
    from elkjax import response

    rng = np.random.default_rng(2)
    n, nocc = 24, 9
    make = lambda: rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    hermitian = lambda m: jnp.asarray(0.5 * (m + m.conj().T))
    h, dh, do = hermitian(make()), hermitian(make()), hermitian(make())
    b = make()
    o = jnp.asarray(b @ b.conj().T + n * np.eye(n))
    probe = jnp.asarray(make()[:, :nocc])

    def loss(t):
        _, x = response.density_factors(h + t * dh, o + t * do, 0.23 * t, nocc,
                                        0.7, 2.0, -1e30, 3)
        return jnp.real(jnp.vdot(probe, x))

    forward = float(jax.jvp(loss, (0.0,), (1.0,))[1])
    reverse = float(jax.grad(loss)(0.0))
    assert abs(forward - reverse) <= 1e-12 * abs(forward)


def test_the_eigenvalue_rule_has_no_denominator_in_it():
    """`response.eigenvalues` against a central difference, at the multiplet.

    Individual eigenvalues of a degenerate group are not differentiable, but
    their SUM is, and so is any symmetric function of them -- which is all this
    port asks of them (the Fermi level, the eigenvalue sum).  The sum over the
    triplet is what this compares, because comparing the three separately would
    be comparing a quantity that does not exist.
    """
    import jax
    import jax.numpy as jnp
    from elkjax import response

    h, _ = _degenerate_hamiltonian()
    n = h.shape[0]
    rng = np.random.default_rng(9)
    dh = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    dh = jnp.asarray(0.5 * (dh + dh.conj().T))
    eye = jnp.eye(n, dtype=complex)

    total = lambda hh: jnp.sum(response.eigenvalues(hh, eye)[1:4])
    analytic = float(jax.jvp(total, (h,), (dh,))[1])
    eps = 1e-6
    difference = float(total(h + eps * dh) - total(h - eps * dh)) / (2 * eps)
    assert abs(analytic - difference) < 1e-8 * max(abs(difference), 1.0)


def test_methfessel_paxton_is_refused_rather_than_approximated():
    """`stype` 0-2 have no closed form for the divided difference.

    Phase 1i measured the `tol` threshold that the literal quotient needs to be
    a cliff -- a pair 67x above it still lost enough of the numerator to hold
    the derivative at 2e-9 instead of 1e-13 -- so the alternative to refusing
    is a number that looks converged and is not.
    """
    import jax.numpy as jnp
    from elkjax import response

    values = jnp.asarray(np.linspace(-0.4, 1.0, 8))
    for stype in (0, 1, 2):
        with pytest.raises(ValueError, match="Fermi-Dirac"):
            response.pair_kernels(values, 0.2, WIDTH, 2.0, -1e30, stype, 6)
