"""Phase 1j: second derivatives of the occupied projector, on Elk's own matrices.

`docs/continue_here.md` §3 item 5, and the last Phase 1 item reachable without Phase 2.
The safe-K rule of §1f is first order BY CONSTRUCTION -- its own JVP body calls
`jnp.linalg.eigh`, so a second derivative falls back on JAX's default eigenvector rule
and the hazard the rule exists to remove comes straight back. `projector.sign_projector`
reaches the same projector as a matrix sign function through Newton-Schulz matrix
products, with no eigendecomposition anywhere, so JAX differentiates it to any order.
Phase 0a' showed that on a toy; this is the first time it has seen an LAPW matrix.

The fixture pair is what makes it a measurement of the RULE rather than of the matrix:
bulk Si at Gamma carries the Gamma_25' triplet inside the occupied window (§1f), and a
generic k on the same ground state carries no degeneracy at all. If the sign route were
merely a different arithmetic path, both would behave the same. They do not -- at
Gamma the eigh-based routes return NaN at second order and the sign route does not,
while at the generic k all three agree.

Skipped without the elk binary, and without jax.
"""

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
GENERIC = (0.1, 0.2, 0.05)
STEPS = 40                      # measured: converged by 20, flat to 60


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


@pytest.fixture(scope="module")
def silicon(tmp_path_factory):
    """One ground state, exported at Gamma and at a generic k."""
    workdir = tmp_path_factory.mktemp("lapw_secondorder") / "Si"
    calculation = Structure(avec=SI_AVEC, species=SI_SPECIES).get_calculation(
        workdir, xc="PW", ngridk=(4, 4, 4), rgkmax=7.0)
    calculation.ensure_ground_state()
    efermi = float((calculation.workdir / "EFERMI.OUT").read_text().split()[0])
    cases = {}
    with calculation.eigenstate_session() as session:
        for label, k in (("gamma", GAMMA), ("generic", GENERIC)):
            export = session.lapw_problem(k)
            h, o = ham.eigenproblem_at(export, np.asarray(export["vkc"]))
            reduced, _ = ham.cholesky_reduce(h, o)
            cases[label] = dict(
                export=export, reduced=np.asarray(reduced),
                tol=ham.projector_tolerance(reduced, o),
                nocc=ham.occupied_band_count(export["evalfv"], efermi),
                evals=np.linalg.eigvalsh(np.asarray(reduced)))
    return cases


def _losses(case, direction, steps=STEPS):
    """The three routes as scalar functions of one displacement parameter."""
    reduced, nocc, tol = case["reduced"], case["nocc"], case["tol"]
    hj = jnp.asarray(reduced)
    dj = jnp.asarray(direction)
    mj = jnp.asarray(np.diag(np.linspace(-1.0, 1.0, reduced.shape[0])).astype(complex))
    trace = lambda p: jnp.real(jnp.trace(p @ mj))
    return (lambda t: trace(pj.sign_projector(hj + t * dj, nocc, steps=steps)),
            lambda t: trace(pj.hard_window_projector(hj + t * dj, nocc, tol)),
            lambda t: trace(pj.naive_hard_window_projector(hj + t * dj, nocc)))


def test_the_newton_schulz_count_is_set_by_the_spectrum_not_the_bandwidth(silicon):
    """The cost measurement, and the reason a toy could not have supplied it.

    The iteration needs ~log(||H~-mu||/gap)/log(3/2) steps. On this reduced LAPW matrix
    the norm is set by the top of the basis (17.9 Ha measured) and not by the 0.35 Ha
    valence manifold, so the ratio is ~190 and the count ~13 rather than ~3. Asserted as
    the separation between the two, plus the observed convergence: 10 steps is not
    converged and 20 is, to the same 1e-14 the safe-K rule itself reaches.
    """
    case = silicon["gamma"]
    reduced, nocc, evals = case["reduced"], case["nocc"], case["evals"]
    gap = evals[nocc] - evals[nocc - 1]
    mu = 0.5 * (evals[nocc - 1] + evals[nocc])
    norm = np.linalg.norm(reduced - mu * np.eye(reduced.shape[0]), 2)
    bandwidth = evals[nocc - 1] - evals[0]
    assert norm / gap > 20 * (bandwidth / gap), (norm, bandwidth, gap)

    safe = np.asarray(pj.hard_window_projector(jnp.asarray(reduced), nocc, case["tol"]))
    coarse = np.asarray(pj.sign_projector(jnp.asarray(reduced), nocc, steps=10))
    converged = np.asarray(pj.sign_projector(jnp.asarray(reduced), nocc, steps=20))
    assert np.linalg.norm(coarse - safe) > 1e-3, "10 steps should NOT be converged"
    assert np.linalg.norm(converged - safe) < 1e-12
    assert np.linalg.norm(converged - ham.elk_occupied_projector(
        case["export"], nocc)) < 1e-12


def test_the_sign_route_reproduces_the_first_derivative(silicon):
    """It has to pass before the second derivative means anything.

    A third, eigensolver-free code path agreeing with the closed form is independent
    confirmation rather than a restatement -- there is no gauge in it and no
    1/(lambda_i - lambda_j) anywhere.
    """
    case = silicon["gamma"]
    n = case["reduced"].shape[0]
    observable = np.diag(np.linspace(-1.0, 1.0, n)).astype(complex)
    occ = ref.hard_occupations(n, case["nocc"])
    for j in range(2):
        direction = ref.random_hermitian_direction(n, 4100 + j)
        sign_loss, _, _ = _losses(case, direction)
        exact = ref.directional_derivative(case["reduced"], direction, observable, occ,
                                           tol=case["tol"])
        assert abs(exact) > 1e-5, "this direction carries no signal"
        assert _rel(float(jax.grad(sign_loss)(0.0)), exact) < 1e-11


def test_only_the_eigensolver_free_route_survives_a_second_derivative(silicon):
    """The Phase 1j result, with its own control in the same ground state.

    At Gamma the Gamma_25' triplet is inside the window, and grad(grad) through either
    eigh-based route is NaN -- the safe-K rule included, exactly as its own docstring
    warns, because its JVP body calls eigh and JAX differentiates THAT with the default
    eigenvector rule. The sign route returns a finite number agreeing with a central
    difference of the (separately validated) first derivative.

    `jax.hessian` is deliberately not used: a custom_vjp cannot be forward-differentiated
    (Phase 0a'), so the composition is grad(grad).
    """
    case = silicon["gamma"]
    n = case["reduced"].shape[0]
    step = 1e-4
    finite = 0
    for j in range(2):
        direction = ref.random_hermitian_direction(n, 4200 + j)
        sign_loss, safe_loss, naive_loss = _losses(case, direction)
        safe_grad = jax.grad(safe_loss)
        reference = float((safe_grad(step) - safe_grad(-step)) / (2 * step))
        second = float(jax.grad(jax.grad(sign_loss))(0.0))
        assert np.isnan(float(jax.grad(jax.grad(safe_loss))(0.0)))
        assert np.isnan(float(jax.grad(jax.grad(naive_loss))(0.0)))
        assert np.isfinite(second)
        if abs(reference) > 1e-4:
            assert _rel(second, reference) < 1e-8, (second, reference)
            finite += 1
    assert finite, "no direction produced a second derivative worth comparing"


def test_the_eigh_routes_are_fine_at_a_generic_k(silicon):
    """The control that makes the NaN above a statement about the MULTIPLET.

    Same ground state, same code, same observable, a k with no degeneracy: all three
    second derivatives agree. So the failure at Gamma is not "second derivatives of a
    projector are hard", it is the enclosed multiplet -- which is what §1f says about
    the first derivative and what this extends to the second.
    """
    case = silicon["generic"]
    n = case["reduced"].shape[0]
    step = 1e-4
    direction = ref.random_hermitian_direction(n, 4300)
    sign_loss, safe_loss, naive_loss = _losses(case, direction)
    safe_grad = jax.grad(safe_loss)
    reference = float((safe_grad(step) - safe_grad(-step)) / (2 * step))
    assert abs(reference) > 1e-4, reference
    for loss in (sign_loss, safe_loss, naive_loss):
        assert _rel(float(jax.grad(jax.grad(loss))(0.0)), reference) < 1e-8


def test_the_second_derivative_in_k_and_that_finite_differences_converge_onto_it(silicon):
    """Through the whole assembly, and with the FD step refined so it can be attributed.

    A single step cannot say which side of a disagreement is wrong. Refining it can: the
    reference marches toward the AD value as O(h^2), which is the finite difference
    converging onto AD rather than the other way round -- the same argument Phase 0a'
    used on the toy, now through `match`, the Gaunt contractions and the Cholesky
    reduction as well.
    """
    case = silicon["gamma"]
    export, nocc, tol = case["export"], case["nocc"], case["tol"]
    kc = np.asarray(export["vkc"])
    n = case["reduced"].shape[0]
    mj = jnp.asarray(np.diag(np.linspace(-1.0, 1.0, n)).astype(complex))
    dk = np.array([0.3, -0.5, 0.2])
    dk /= np.linalg.norm(dk)
    kj, dkj = jnp.asarray(kc), jnp.asarray(dk)

    def reduced_at(t):
        return ham.cholesky_reduce(*ham.eigenproblem_at(export, kj + t * dkj))[0]

    sign_loss = lambda t: jnp.real(jnp.trace(
        pj.sign_projector(reduced_at(t), nocc, steps=STEPS) @ mj))
    safe_loss = lambda t: jnp.real(jnp.trace(
        pj.hard_window_projector(reduced_at(t), nocc, tol) @ mj))

    second = float(jax.grad(jax.grad(sign_loss))(0.0))
    assert np.isfinite(second) and abs(second) > 1e-2, second
    assert _rel(float(jax.grad(sign_loss)(0.0)),
                float(jax.grad(safe_loss)(0.0))) < 1e-10

    safe_grad = jax.grad(safe_loss)
    errors = []
    for h in (1e-3, 3e-4, 1e-4):
        errors.append(abs(float((safe_grad(h) - safe_grad(-h)) / (2 * h)) - second))
    assert errors[0] > errors[1] > errors[2], errors
    assert errors[0] / errors[2] > 30, errors        # O(h^2) over a 10x step range
