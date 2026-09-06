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

The two branches turn out to be EXACTLY COMPLEMENTARY, which is the finding
this file exists to record, and it follows from how Elk builds the muffin-tin
Hamiltonian rather than from anything about the fixtures.

  * The SPHERICAL part of `vsmt` never appears in a radial integral.
    `hmlrad`'s l2 = 0 element is <u|H u>, which `genapwfr` has already applied
    the radial Hamiltonian to -- the radial functions ARE that operator's
    solutions, so the spherical potential is absorbed into the basis and only
    the non-spherical remainder survives as an explicit matrix element (the
    standard LAPW construction).  The frozen-basis derivative along a purely
    spherical direction is therefore not small: it is EXACTLY ZERO, while the
    full derivative is O(1).
  * The NON-SPHERICAL part never reaches the radial equation, which
    `genapwfr` and `genlofr` integrate in the spherical potential alone.  Its
    basis response is therefore exactly zero and the two branches agree to
    roundoff.

So a Hellmann-Feynman-shaped treatment of the muffin-tin potential does not
lose a small correction in the spherical channel; it loses the whole term.
Measured on bulk Si with a random direction, which mixes the two: 79% of the
derivative is basis response.

Three further checks:

  1. The derivative is linear in the direction, so the two halves must add
     back to the whole -- which a wrong stride through the packed potential
     would break while leaving everything above passing.
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

from test_calculation_lapw_assembly import exports, _module_tmp  # noqa

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
        if name == "spherical":
            continue                       # identically zero; see below
        assert abs(row["closed_form"]) > 1e-8, (name, "no signal")
        assert _rel(row["ad_frozen"], row["closed_form"]) < 1e-11, (name, row)


@pytest.mark.parametrize("case", CASES)
def test_the_spherical_channel_enters_only_through_the_basis(case, gradients):
    """`hmlrad`'s l2 = 0 element is <u|H u> and contains no potential, so the
    spherical channel reaches H only by moving the radial functions.

    The frozen-basis derivative is then exactly zero -- structurally, not
    numerically -- while the full one is O(1).  A frozen-basis argument does
    not approximate this term; it deletes it.
    """
    row = gradients[case]["spherical"]
    assert abs(row["ad_full"]) > 1e-3, "the spherical direction carries no signal"
    assert row["ad_frozen"] == 0.0, row
    assert row["closed_form"] == 0.0, row


@pytest.mark.parametrize("case", CASES)
def test_the_non_spherical_channel_enters_only_through_the_integrals(
        case, gradients):
    """The complement: `genapwfr`/`genlofr` integrate in the spherical
    potential alone, so a non-spherical perturbation cannot move the basis and
    the two branches must agree to ROUNDOFF, not closely."""
    row = gradients[case]["non-spherical"]
    assert abs(row["ad_full"]) > 1e-8, "the null direction carries no signal"
    assert _rel(row["ad_full"], row["ad_frozen"]) < 1e-11, row


@pytest.mark.parametrize("case", CASES)
def test_the_derivative_is_linear_in_the_direction(case, gradients):
    """Spherical plus non-spherical must return the whole -- a check on the
    direction split itself, which a wrong stride through the packed potential
    would break while leaving every other test here passing."""
    rows = gradients[case]
    for tag in ("ad_frozen", "ad_full"):
        total = rows["spherical"][tag] + rows["non-spherical"][tag]
        assert _rel(total, rows["random"][tag]) < 1e-11, (tag, rows)


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


def test_the_basis_response_is_not_a_small_correction(gradients):
    """The measurement this whole file exists to make, on a direction that
    mixes the two channels.

    LAPW's basis depends on the potential, so the frozen-basis
    (Hellmann-Feynman-shaped) term is not the derivative -- the same finding
    `docs/jax_port_phase1.md` §1e records for k, in a channel where it is far
    larger.  Asserted as an order of magnitude, since its exact value depends
    on the direction; measured 79% on bulk Si.
    """
    for case in CASES:
        row = gradients[case]["random"]
        share = abs(row["basis_response"]) / abs(row["ad_full"])
        assert share > 0.1, (case, share, row)
