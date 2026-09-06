r"""Phase 1 of the JAX port: differentiate the LAPW spectrum with respect to
the muffin-tin Kohn-Sham potential.

`test_calculation_lapw_radial_functions.py` closes the forward chain
vsmt -> apwfr/lofr -> radial integrals -> H, O -> evalfv.  This differentiates
it, and separates the two things that derivative contains.

  * The FROZEN-BASIS branch holds the radial functions fixed and lets only the
    radial integrals respond.  The overlap does not depend on the potential at
    all then, so first-order perturbation theory for the generalised
    eigenproblem collapses to sum_n c_n^dag dH c_n with c normalised as
    c^dag O c = 1 -- an exact closed form, computed here from Elk's OWN
    `evecfv`, so this branch has an oracle that involves no finite difference
    and no eigensolve of a perturbed matrix.

  * The FULL branch lets the perturbation reach the radial functions, which
    `genapwfr` re-solves in the perturbed potential at the same linearisation
    energies.  The LAPW basis is potential-dependent, so this is a genuinely
    different number, and the difference is the term a frozen-basis argument
    drops.

The two branches turn out to be EXACTLY COMPLEMENTARY, and the reason is how
Elk builds the muffin-tin Hamiltonian rather than anything about the fixtures.

  * The SPHERICAL part of `vsmt` never appears in a radial integral.
    `hmlrad`'s l2 = 0 element is <u|H u>, and `genapwfr` has already applied
    H -- the radial functions ARE that operator's solutions, so the radial
    equation has ELIMINATED the explicit spherical-potential integral in
    favour of the linearisation energy.  The frozen-basis derivative along a
    purely spherical direction is therefore not small: it is EXACTLY ZERO.
  * The NON-SPHERICAL part never reaches the radial equation, which
    `genapwfr` and `genlofr` integrate in the spherical potential alone.  Its
    basis response is exactly zero and the two branches agree to roundoff.

**So "full minus frozen" is NOT the basis relaxation**, and calling it that
would conflate two different things: the Hellmann-Feynman term Elk's assembly
has hidden, and the genuine response of the basis.  `hellmann_feynman` computes
the first explicitly -- the same integrals with the l2 = 0 slice filled by the
potential integral rather than zeroed -- and

    d(sum eps_n)/dt = <psi|dV|psi>  +  basis relaxation

is the decomposition a Pulay / incomplete-basis-set discussion actually wants.

**And the relaxation term depends enormously on the SHAPE of the
perturbation**, which is the measurement this file exists to make.  On bulk Si
it is 29% of the derivative for a white-noise direction that has structure down
to the nuclear cusp, where a basis built at a fixed linearisation energy cannot
follow it -- and 0.3% for a smooth spherical bump in the valence region, which
is roughly what an SCF update to the density does.  Quoting the first number
alone would badly misrepresent the method.

Three further checks:

  1. The derivative is linear in the direction, so the spherical and
     non-spherical halves must add back to the whole -- which a wrong stride
     through the packed potential would break while leaving everything above
     passing.
  2. The closed form pins the frozen-basis branch outright.  Getting it right
     needs one non-obvious fact: the map from the potential to the radial
     integrals is AFFINE, not linear.  Its constant part is that same l2 = 0
     element; carrying it into dH flips the sign of the answer on bulk Si.
  3. AD against central finite differences of the same function.

The linearisation energies are held FIXED.  Letting them float would mean
differentiating `linengy`, which re-solves them by a bisection on the
logarithmic derivative -- a different object, and one Elk's own forces do not
carry either.

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

NOCC = 4          # both fixtures carry 8 valence electrons, i.e. 4 bands
CASES = ["si_apword1", "hbn"]


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def _skip_without_potential(export):
    if "vsmt" not in export:
        pytest.skip("binary predates patch 0015 (no potential exported)")


@pytest.fixture(scope="module")
def gradients(exports):
    """Both branches' AD and the closed form, for all three directions, on
    both fixtures.  Module-scoped because each entry re-solves the radial
    equation for every l."""
    from elkjax import phase1_potential as pot
    out = {}
    for case in CASES:
        export = exports[case]
        _skip_without_potential(export)
        rows = {}
        for name, direction in pot.split_directions(export).items():
            rows[name] = pot.derivatives(export, direction, NOCC)
        out[case] = rows
    return out


@pytest.mark.parametrize("case", CASES)
def test_frozen_basis_derivative_is_first_order_perturbation_theory(
        case, gradients):
    """The oracle with no finite difference in it."""
    for name, row in gradients[case].items():
        if name in ("spherical", "valence"):
            continue        # identically zero -- asserted in its own test
        assert abs(row["closed_form"]) > 1e-8, (name, "no signal")
        assert _rel(row["ad_frozen"], row["closed_form"]) < 1e-11, (name, row)


@pytest.mark.parametrize("case", CASES)
def test_the_spherical_channel_enters_only_through_the_basis(case, gradients):
    """`hmlrad`'s l2 = 0 element is <u|H u> and contains no potential, so the
    spherical channel reaches H only by moving the radial functions.

    The frozen-basis derivative is then exactly zero -- structurally, not
    numerically -- while the Hellmann-Feynman term and the full derivative are
    both O(1).  That gap is not "basis response": it is the physical
    first-order term Elk's bookkeeping has eliminated.
    """
    for name in ("spherical", "valence"):
        row = gradients[case][name]
        assert abs(row["ad_full"]) > 1e-3, (name, "no signal")
        assert row["ad_frozen"] == 0.0, (name, row)
        assert row["closed_form"] == 0.0, (name, row)
        assert abs(row["hellmann_feynman"]) > 1e-3, (name, row)


@pytest.mark.parametrize("case", CASES)
def test_the_non_spherical_channel_enters_only_through_the_integrals(
        case, gradients):
    """The complement: `genapwfr`/`genlofr` integrate in the spherical
    potential alone, so a non-spherical perturbation cannot move the basis.

    All three numbers must coincide to roundoff -- and Elk's own matrix
    element IS the Hellmann-Feynman term in this channel, which is what makes
    the spherical channel's disagreement a statement about the assembly rather
    than about the fixture.
    """
    row = gradients[case]["non-spherical"]
    assert abs(row["ad_full"]) > 1e-8, "the null direction carries no signal"
    assert _rel(row["ad_full"], row["ad_frozen"]) < 1e-11, row
    assert _rel(row["hellmann_feynman"], row["ad_frozen"]) < 1e-11, row


@pytest.mark.parametrize("case", CASES)
def test_the_derivative_is_linear_in_the_direction(case, gradients):
    """Spherical plus non-spherical must return the whole -- a check on the
    direction split itself, which a wrong stride through the packed potential
    would break while leaving every other test here passing."""
    rows = gradients[case]
    for tag in ("ad_frozen", "ad_full", "hellmann_feynman"):
        total = rows["spherical"][tag] + rows["non-spherical"][tag]
        assert _rel(total, rows["random"][tag]) < 1e-11, (tag, rows)


@pytest.mark.parametrize("case", CASES)
def test_the_relaxation_lives_entirely_in_the_spherical_channel(
        case, gradients):
    """It must, since the non-spherical channel has none -- so the random
    direction's relaxation has to equal the spherical direction's exactly,
    which is a sharper statement than either number alone."""
    rows = gradients[case]
    assert _rel(rows["random"]["relaxation"],
                rows["spherical"]["relaxation"]) < 1e-9, rows
    assert abs(rows["non-spherical"]["relaxation"]) < 1e-12 * abs(
        rows["non-spherical"]["ad_full"]), rows


def test_ad_agrees_with_central_differences(exports):
    """AD against central FD of the SAME function, both branches.

    The discrepancy must GROW as the step shrinks -- the 1/h signature of
    roundoff in the difference -- rather than sit at a fixed value, which is
    what a wrong gradient looks like.
    """
    from elkjax import phase1_potential as pot
    export = exports["si_apword1"]
    _skip_without_potential(export)
    steps = (1e-4, 1e-5, 1e-6)
    direction = pot.split_directions(export)["random"]
    row = pot.derivatives(export, direction, NOCC, steps=steps)
    for tag in ("frozen", "full"):
        errors = [_rel(fd, row[f"ad_{tag}"]) for fd in row[f"fd_{tag}"]]
        assert errors[0] < 1e-7, (tag, errors)
        assert errors[2] > 3 * errors[0], (tag, errors)


def test_the_relaxation_depends_on_the_shape_of_the_perturbation(gradients):
    """The measurement this file exists to make.

    A white-noise potential direction has structure down to the nuclear cusp,
    where radial functions built at a FIXED linearisation energy cannot follow
    it, and the relaxation term is 29% of the derivative.  A smooth spherical
    bump in the valence region -- roughly what an SCF update does -- gives
    0.3%.  Both are asserted, and so is the ratio between them, because
    quoting either alone misrepresents the method in opposite directions.
    """
    for case in CASES:
        rows = gradients[case]
        noisy = abs(rows["random"]["relaxation"] / rows["random"]["ad_full"])
        smooth = abs(rows["valence"]["relaxation"]
                     / rows["valence"]["ad_full"])
        assert noisy > 0.1, (case, noisy)
        assert smooth < 0.02, (case, smooth)
        assert noisy > 10 * smooth, (case, noisy, smooth)
