r"""The Newton-Schulz tape as a `lax.scan` (Phase 1 leftover, `elkjax.phase1_scan`).

`sign_projector` is the only projector here that survives a second derivative at a
multiplet, and it gets there by being nothing but a fixed number of matrix products.
Written as a Python loop those products are unrolled into the graph -- exactly the
construct Phase 0e measured compile time as superlinear in.  `lax.scan` emits the
body once.

These tests assert the two things a rewrite like that has to earn: it does not move
the answer, at any of the three orders that matter; and it does actually cost less to
compile, by a margin that grows with the tape.

Nothing here needs the Elk binary -- the matrix is synthetic, shaped like the real
spectrum Phase 1j measured (a narrow occupied manifold under a basis reaching 18 Ha,
so the step count is set by the top of the basis).  Skipped without jax.
"""

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="jax not installed; pip install -e .[jax]")


def _limit():
    from elkjax import memory
    memory.limit_address_space()


def test_the_two_forms_agree_to_roundoff_at_every_order():
    """Value, first and second derivative of tr(PW) along a line.

    A relative tolerance and not exact equality: the two forms call the same
    `_newton_schulz_step` the same number of times, and still differ at 6e-16,
    because XLA fuses a scan body and an unrolled chain into different regions and
    reassociates inside each.  Asserting bitwise equality here would be a test that
    fails on a compiler upgrade and means nothing when it does.
    """
    _limit()
    from elkjax import phase1_scan
    rows = phase1_scan.agreement()
    for order, row in rows.items():
        assert abs(row["scanned"]) > 1e-3, (
            f"order {order}: the observable must actually vary, or this compares "
            "two noise floors -- sum(P*P) is tr(P) = nocc and is constant")
        assert row["difference"] / abs(row["scanned"]) < 1e-13


def test_the_projector_itself_is_not_bitwise_identical():
    """The premise of the tolerance above, asserted rather than assumed.

    If a compiler change ever makes the two forms bitwise equal this fails, which
    is the signal to tighten the test above rather than leave it loose for a reason
    that has stopped applying.
    """
    _limit()
    import numpy as np
    from elkjax import phase1_scan, projector
    h, nocc = phase1_scan.hamiltonian()
    scanned = np.asarray(projector.sign_projector(h, nocc, steps=20))
    unrolled = np.asarray(projector.sign_projector(h, nocc, steps=20, unroll=True))
    difference = np.abs(scanned - unrolled).max()
    assert 0.0 < difference < 1e-14


def test_the_projector_is_a_rank_nocc_projector():
    """P^2 = P and tr P = nocc, which is what makes 20 steps enough here.

    Phase 1j measured 10 steps failing outright (error 1.1, not a projector at all)
    and 20 converging, on a spectrum of this shape.  Without this the agreement test
    above would pass just as well on two identically unconverged iterations.
    """
    _limit()
    import numpy as np
    from elkjax import phase1_scan, projector
    h, nocc = phase1_scan.hamiltonian()
    p = np.asarray(projector.sign_projector(h, nocc, steps=20))
    assert np.abs(p @ p - p).max() < 1e-12
    assert abs(np.trace(p) - nocc) < 1e-10


@pytest.mark.parametrize("order", [1, 2])
def test_scan_compiles_far_cheaper_than_the_unrolled_tape(order):
    """The point of the rewrite, at a tape length where it matters.

    Instruction count is asserted rather than wall time: the count is what Phase 0e
    found compile time to be superlinear in, and it is deterministic, where a
    timing on a shared machine is not.  Measured at 40 steps: 47x the instructions
    at first order and 62x at second.
    """
    _limit()
    from elkjax import phase1_scan
    scan = phase1_scan.compile_cost(steps=40, order=order, unroll=False)
    unrolled = phase1_scan.compile_cost(steps=40, order=order, unroll=True)
    assert unrolled["instructions"] > 20 * scan["instructions"]


def test_the_scanned_instruction_count_is_flat_in_the_tape_length():
    """The structural claim: `scan` emits the body once, so the graph does not grow
    with the number of steps.  Measured over a 32-fold range it goes 197, 197, 199,
    199, 199, 199 -- flat, not exactly constant, the two extra instructions
    appearing somewhere between 10 and 20 steps and never again.  The unrolled tape
    over the same range grows linearly, and that contrast is what makes this a
    measurement of the graph rather than of the machine."""
    _limit()
    from elkjax import phase1_scan
    counts = {steps: phase1_scan.compile_cost(steps=steps, order=1,
                                              unroll=False)["instructions"]
              for steps in (5, 160)}
    assert counts[160] - counts[5] < 10
    unrolled = {steps: phase1_scan.compile_cost(steps=steps, order=1,
                                                unroll=True)["instructions"]
                for steps in (5, 160)}
    assert unrolled[160] > 20 * unrolled[5]
