r"""Phase 1 of the JAX port: the position derivative at frozen potential.

`docs/jax_port.md` states Phase 1's gradient criterion as d(eps)/dR on
displaced h-BN.  Half of what that needs now exists -- the radial integrals are
built rather than imported (§1k) -- and half does not: moving an atom moves the
potential inside its sphere, and where that comes from is Phase 2.

What this checks is the derivative at FROZEN potential, the rigid-muffin-tin
picture.  Positions then enter the eigenproblem in exactly one place, the
structure factor exp(i (G+k) . r_alpha) of the matching coefficients.  It is
NOT a force: Elk's total force also carries the Hellmann-Feynman and core
terms, and the interstitial characteristic function moves with the sphere too.

**The check with teeth is a sum rule, not a finite difference.**  Translating
every atom by the same delta cannot change the spectrum: every basis function
picks up a phase, the whole matrix transforms as U^dag M U with U diagonal, and
the eigenvalues are invariant.  The muffin-tin blocks get that right on their
own through the structure factor; the imported interstitial blocks do not, but
their true response to a rigid shift is the same phase in closed form
(Theta(r) -> Theta(r - delta) gives Theta~(G) -> Theta~(G) exp(-i G . delta)),
which is applied.  A wrong sign or factor in `match`'s position dependence
breaks the invariance and nothing else in it can.

**And the FORWARD form of that null is the sharper one**, which is the
methodological point worth keeping.  With the interstitial response left out,
the finite-shift error is 2.9e-4 Ha on Si and 4.3e-4 Ha on h-BN -- but it
scales as delta^2 on Si (measured ratios 4.01, 4.00 per halving) and only as
delta on h-BN (1.87, 1.79).  So the GRADIENT null is identically satisfied by
the wrong assembly on silicon, and only the finite-shift comparison separates
the two on both fixtures.  A gradient check blind to something a forward check
sees is Phase 0's standing finding, in its third distinct form here.

Skipped without the elk binary, and without jax.
"""

import importlib.util

import numpy as np
import pytest

from elkpy import config

pytestmark = [
    pytest.mark.skipif(not config.default_elk_binary().is_file(),
                       reason="elk binary not built; see docs/design.md #8"),
    pytest.mark.skipif(importlib.util.find_spec("jax") is None,
                       reason="jax not installed; pip install -e .[jax]"),
]

NOCC = 4                     # both fixtures: 8 valence electrons, 4 bands
CASES = ["si_apword1", "hbn"]
DELTA = np.array([0.031, -0.017, 0.023])       # Bohr, a general direction


@pytest.fixture(scope="module")
def nulls(exports):
    from elkjax import phase1_position as pos
    return {case: pos.translation_null(exports[case], NOCC, delta=DELTA)
            for case in CASES}


@pytest.mark.parametrize("case", CASES)
def test_rigid_translation_leaves_the_spectrum_invariant(case, nulls):
    """The sum rule, forward and as a gradient."""
    null = nulls[case]
    assert null["forward_phase-corrected"] < 1e-13, null
    assert np.abs(null["grad_phase-corrected"]).max() < 1e-13, null


@pytest.mark.parametrize("case", CASES)
def test_the_null_is_not_free(case, nulls):
    """Dropping the interstitial's own response to the shift must break it,
    or the test above would be measuring nothing."""
    null = nulls[case]
    assert null["forward_frozen"] > 1e-5, null


def test_the_gradient_null_alone_is_blind_on_silicon(exports, nulls):
    """The methodological point, asserted rather than described.

    Left out, the interstitial's response costs O(delta^2) on silicon and
    O(delta) on h-BN -- so the wrong assembly satisfies the GRADIENT null
    exactly on silicon while failing the finite-shift one.  Both scalings are
    checked directly, since it is the scaling and not the size that decides
    which check can see the omission.
    """
    from elkjax import phase1_position as pos
    assert np.abs(nulls["si_apword1"]["grad_frozen"]).max() < 1e-13
    assert np.abs(nulls["hbn"]["grad_frozen"]).max() > 1e-4
    orders = {}
    for case in CASES:
        export = exports[case]
        atposc = np.asarray(export["atposc"])
        base = np.asarray(pos.spectrum_at_positions(export, atposc, nocc=NOCC))
        errors = []
        for scale in (1.0, 0.5, 0.25):
            moved = atposc + (scale * DELTA)[:, None]
            shifted = np.asarray(
                pos.spectrum_at_positions(export, moved, nocc=NOCC))
            errors.append(np.abs(shifted - base).max())
        orders[case] = [errors[i] / errors[i + 1] for i in range(2)]
    assert all(r > 3.5 for r in orders["si_apword1"]), orders      # delta^2
    assert all(r < 2.5 for r in orders["hbn"]), orders             # delta


@pytest.mark.parametrize("case", CASES)
def test_single_atom_derivative_against_finite_differences(case, exports):
    """One atom along a general direction, AD against central FD of the same
    function.

    No external oracle here -- this is the partial, frozen-potential
    derivative, not a force -- so what is asserted is agreement at the best
    step and a MINIMUM in the middle of the step sweep: truncation dominating
    at the large step and roundoff at the small one, which together say the
    two are converging on each other rather than sitting at a fixed offset.
    """
    from elkjax import phase1_position as pos
    row = pos.single_atom(exports[case], NOCC)
    errors = [abs(fd - row["ad"]) / abs(row["ad"]) for fd in row["fd"]]
    assert abs(row["ad"]) > 1e-4, "this direction carries no signal"
    assert min(errors) < 1e-8, (case, errors)
    assert errors.index(min(errors)) == 1, (case, errors)
