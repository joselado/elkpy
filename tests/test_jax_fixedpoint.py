"""Phase 0a of the Elk-to-JAX port: implicit differentiation of an SCF fixed point.

Study §6 item 0a and §8(a) of `docs/jax_port.md`.  The question is not whether implicit
differentiation works in general -- a smooth scalar fixed point with no eigensolve is
already known to work and settles nothing -- but whether it survives having an
eigensolve inside it, since the adjoint matvec is one Kohn-Sham JVP, transposed, and
therefore evaluates `eigh`'s derivative at every multiplet on every GMRES iteration.

The reference is NOT finite differences.  Every comparison is against a dense
implicit-function-theorem solve built from the closed-form projector derivative
(`ScfToy.reference_gradient`): different linear solver, different matrix dimension,
no autodiff anywhere.  FD is checked alongside as a third opinion.
"""

import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import elkjax  # noqa: E402
from elkjax import fixedpoint, memory, phase0a, scftoy  # noqa: E402

memory.limit_address_space(16.0)

SLOW = pytest.mark.skipif(
    not os.environ.get("ELKPY_RUN_SLOW_TESTS"),
    reason="set ELKPY_RUN_SLOW_TESTS=1 to run the full Phase 0a sweep",
)


def _small(**options):
    """A cheap instance: 6 states doubled to 12, every level exactly two-fold.

    ``break_symmetry=True`` makes the parameter couple the degenerate partners, which
    is the case that matters — a displacement or strain lowering a crystal symmetry —
    and the only one in which the naive projector rule actually fails here.
    """
    options.setdefault("break_symmetry", True)
    return scftoy.build(size=6, nocc=2, seed=1, coupling=1.5, degeneracy=2,
                        rotate=True, **options)


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


def test_the_toy_has_the_structure_the_item_asks_for():
    toy = _small()
    result = phase0a.gradient(toy, tol=1e-12)
    evals = np.asarray(jnp.linalg.eigvalsh(toy.hamiltonian(jnp.asarray(result["v"]), 0.0)))
    # every level exactly two-fold, so the eigensolve meets a multiplet at every step
    assert np.min(np.diff(evals)) < 1e-14
    # ... and the window boundary is gapped, so the projector is differentiable at all
    assert evals[2 * toy.nocc] - evals[2 * toy.nocc - 1] > 0.1
    # the self-consistency is doing real work: not a perturbative correction
    assert result["radius"] > 0.3
    assert _rel(result["grad"], result["frozen"]) > 0.05


def test_implicit_gradient_matches_the_dense_reference():
    toy = _small()
    result = phase0a.gradient(toy, tol=1e-12)
    assert result["scf_residual"] < 1e-12
    assert result["adjoint_residual"] < 1e-9      # GMRES really inverted the transpose
    assert _rel(result["grad"], result["reference"]) < 1e-10


def test_the_naive_projector_breaks_the_same_machinery():
    """The control that ties Phase 0a's success to Phase 0b's rule.

    Identical fixed point, identical GMRES, identical loss -- only the projector rule
    differs.  Whether it comes back NaN or merely wrong is the eigensolver's choice
    (Phase 0b), so both count as the failure; measured on this instance, 47% relative.
    """
    toy = _small(rule="naive")
    result = phase0a.gradient(toy, tol=1e-12)
    assert result["scf_residual"] < 1e-12         # the FORWARD solve is unaffected
    assert (not np.isfinite(result["grad"])
            or _rel(result["grad"], result["reference"]) > 1e-2)


def test_a_symmetry_respecting_perturbation_hides_the_bug():
    """Why a badly chosen test would have passed with the naive rule.

    When the perturbation and the observable both commute with the symmetry that
    protects the degeneracy, they have no matrix element between the partners, so the
    erroneous off-diagonal block of the naive rule is suppressed twice and never shows.
    Measured: 1.5e-14 with a symmetric perturbation, 4.7e-1 with a symmetry-breaking one,
    on the same Hamiltonian.  Pinned so the distinction is not lost.
    """
    symmetric = phase0a.gradient(_small(rule="naive", break_symmetry=False), tol=1e-12)
    broken = phase0a.gradient(_small(rule="naive", break_symmetry=True), tol=1e-12)
    assert _rel(symmetric["grad"], symmetric["reference"]) < 1e-10
    assert _rel(broken["grad"], broken["reference"]) > 1e-2


def test_band_energy_is_differentiable_too():
    """Study §8(b)'s "exact and free" quantity, through the fixed point."""
    toy = _small()
    result = phase0a.gradient(toy, quantity="band_energy", tol=1e-12)
    assert _rel(result["grad"], result["reference"]) < 1e-10


def test_gradient_does_not_depend_on_the_mixer():
    """`F` is defined without the mixer, so the gradient must not see it (study §8a).

    Needs no reference value, which makes it the sharpest check available here.
    """
    toy = _small()
    linear = phase0a.gradient(toy, tol=1e-12, mixing=0.4, history=0)
    anderson = phase0a.gradient(toy, tol=1e-12, mixing=0.4, history=5)
    assert np.linalg.norm(linear["v"] - anderson["v"]) < 1e-10
    assert _rel(linear["grad"], anderson["grad"]) < 1e-10


def test_smearing_at_fixed_chemical_potential():
    toy = _small(smearing=(0.0, 0.3))
    result = phase0a.gradient(toy, tol=1e-12)
    assert _rel(result["grad"], result["reference"]) < 1e-10


@SLOW
def test_all_three_spectra_agree_with_the_reference_and_with_fd():
    for quantity in ("loss", "band_energy"):
        for row in phase0a.experiment_degeneracies(quantity=quantity):
            assert _rel(row["grad"], row["reference"]) < 1e-12, row["label"]
            assert _rel(row["fd"], row["reference"]) < 1e-6, row["label"]
            assert row["adjoint_residual"] < 1e-9, row["label"]


@SLOW
def test_implicit_is_independent_of_the_iteration_count_and_unrolling_is_not():
    """Study §8(a): unrolling is not inaccurate -- it is O(iterations) in tape.

    So the assertion is not that unrolling is wrong at convergence, but that it needs
    the iterations to get there while the implicit route never does.
    """
    result = phase0a.experiment_unrolled()
    rows = {r["iterations"]: r["unrolled"] for r in result["rows"]}
    assert _rel(result["implicit"], result["reference"]) < 1e-12
    assert _rel(rows[50], result["reference"]) > 1e-2      # far off after 50 steps
    assert _rel(rows[400], result["reference"]) < 1e-10    # and right in the end
    # and it does not even approach monotonically: measured 1.2e-1, 1.0e+0, 2.0e-2
    assert _rel(rows[50], result["reference"]) > _rel(rows[20], result["reference"])


@SLOW
def test_second_order_is_blocked_by_the_degeneracy_not_by_the_fixed_point():
    """Phase 0a-prime, pinned as it actually stands.

    Reverse-over-reverse through the `custom_vjp` fixed point is fine; what fails is the
    safe-K rule's OWN second derivative, whose JVP body calls `eigh` and meets the
    multiplet with JAX's default rule.  Forward-over-reverse (`jax.hessian`) is refused
    outright by JAX for any `custom_vjp`, which is a library limitation and not a
    statement about the physics -- reporting it as a kill would be wrong.
    """
    rows = {row["label"]: row for row in phase0a.experiment_second_order()}
    plain = rows["no degeneracy"]
    assert np.isfinite(plain["grad(grad)"])
    assert abs(plain["grad(grad)"] - plain["jacrev(jacrev)"]) < 1e-12
    assert _rel(plain["grad(grad)"], plain["central FD"]) < 1e-5
    for label in ("doubled, rotated", "doubled, symmetry-broken"):
        degenerate = rows[label]
        assert not np.isfinite(degenerate["grad(grad)"]), label
        assert np.isfinite(degenerate["central FD"]), label   # the derivative EXISTS
    for row in rows.values():
        assert isinstance(row["jax.hessian"], str) and "TypeError" in row["jax.hessian"]


@SLOW
def test_gradient_error_tracks_the_scf_residual():
    """Implicit gradients are exact only AT a fixed point (study §8a)."""
    rows = phase0a.experiment_tolerance()
    for row in rows:
        assert _rel(row["grad"], row["reference"]) < 1e-12   # exact at the point reached
        assert _rel(row["grad"], row["converged"]) < 100 * row["scf_residual"]
    assert _rel(rows[0]["grad"], rows[0]["converged"]) > 10 * _rel(
        rows[-1]["grad"], rows[-1]["converged"])
