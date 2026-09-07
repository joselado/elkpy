"""`elkjax.occupations` on synthetic spectra: no Elk run, no ground state.

Everything here is checkable against a closed form or against the constraint
`occupy` solves, so it needs no reference from Elk.  The comparison against
Elk's own `efermi`/`occsv` is `tests/test_calculation_occupations.py`.

Self-skips without jax.
"""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="jax not installed; pip install -e .[jax]")

WIDTH = 0.05
OCCMAX = 2.0
DEEP = -1.0e6


def _spectrum(seed=0, nkpt=4, nstsv=10):
    values = np.sort(np.random.default_rng(seed).normal(size=(nkpt, nstsv)),
                     axis=1)
    return values, np.full(nkpt, 1.0 / nkpt)


@pytest.mark.parametrize("stype", [0, 1, 2, 3])
def test_sdelta_is_the_derivative_of_stheta(stype):
    """The pair Elk keeps in two files, checked as one statement.

    `occupy` uses `stheta` and the derivative rule uses `sdelta`, so a wrong
    pairing would give a Fermi level that is right and a d(mu) that is wrong --
    a failure mode with no forward symptom at all.
    """
    import jax
    import jax.numpy as jnp
    from elkjax import occupations

    x = jnp.linspace(-8.0, 8.0, 65)
    got = jax.vmap(jax.grad(lambda t: occupations.stheta(stype, t)))(x)
    assert np.abs(np.asarray(got)
                  - np.asarray(occupations.sdelta(stype, x))).max() < 1e-12


def test_the_fermi_dirac_cutoffs_are_finite_in_value_and_gradient():
    """Elk's |x| > 50 branches, and the `where`-NaN hazard section 2a names.

    The exponent is clipped before the exponential, so the discarded branch
    cannot overflow; without that the gradient is NaN far outside the window
    even though the value is right.
    """
    import jax
    import jax.numpy as jnp
    from elkjax import occupations

    x = jnp.asarray([-1e4, -60.0, 0.0, 60.0, 1e4])
    assert np.allclose(np.asarray(occupations.stheta(3, x)),
                       [0.0, 0.0, 0.5, 1.0, 1.0])
    assert np.allclose(np.asarray(occupations.sdelta(3, x)),
                       [0.0, 0.0, 0.25, 0.0, 0.0])
    for function in (occupations.stheta, occupations.sdelta):
        slope = jax.vmap(jax.grad(lambda t: function(3, t)))(x)
        assert jnp.isfinite(slope).all()


def test_an_untranscribed_smearing_type_raises():
    """`stype` 4 and 5 are Elk's square-wave and Lorentzian, and are not here.

    They must refuse rather than silently return a Fermi-Dirac occupation,
    which would be a plausible number computed under the wrong statistics.
    """
    from elkjax import occupations
    for stype in (4, 5):
        with pytest.raises(ValueError, match="not transcribed"):
            occupations.stheta(stype, 0.0)
        with pytest.raises(ValueError, match="not transcribed"):
            occupations.sdelta(stype, 0.0)


def test_the_bisection_solves_the_constraint_it_claims_to():
    """The electron count at the returned mu, against the target.

    The bisection stops at a bracket of 1e-12 Ha, so the count it lands on is
    off by that times the response -- which is what is asserted, rather than a
    round number.
    """
    from elkjax import occupations
    evalsv, wkpt = _spectrum()
    target = 6.0
    mu = occupations.fermi_level(evalsv, wkpt, target, WIDTH, OCCMAX, DEEP, 3)
    occsv = occupations.occupations(evalsv, mu, WIDTH, OCCMAX, DEEP, 3)
    assert abs(float(occupations.electron_count(occsv, wkpt)) - target) < 1e-9


def test_the_e0min_gate_changes_the_electron_count():
    """The gate Elk applies below the minimum linearisation energy.

    It does NOT fire on either real fixture -- Si's deepest eigenvalue is
    -0.24 Ha against an `e0min` of -2.0 -- so it is exercised here instead,
    with a state deliberately placed below the cutoff.  Elk zeroes such a
    state rather than filling it: it is a ghost of the linearisation, not an
    electron, and keeping it would move mu.
    """
    from elkjax import occupations
    evalsv, wkpt = _spectrum()
    evalsv = evalsv.copy()
    evalsv[:, 0] = -12.0
    gated = occupations.occupations(evalsv, 0.0, WIDTH, OCCMAX, -10.0, 3)
    ungated = occupations.occupations(evalsv, 0.0, WIDTH, OCCMAX, DEEP, 3)
    assert float(occupations.electron_count(ungated, wkpt)) \
        - float(occupations.electron_count(gated, wkpt)) == pytest.approx(2.0)


def test_dmu_matches_finite_differences_in_every_argument():
    """The three differentiable arguments of `fermi_level`, one at a time.

    d(mu)/d(evalsv) is the one the SCF needs; d(mu)/d(chgval) and
    d(mu)/d(wkpt) come from the same implicit differentiation of the count and
    are checked because getting one term of that expression wrong leaves the
    other two right.
    """
    import jax
    import jax.numpy as jnp
    from elkjax import occupations
    evalsv, wkpt = _spectrum()
    target = 6.0
    rng = np.random.default_rng(7)
    step = 1e-6

    def check(function, point, direction):
        forward = float(jax.jvp(function, (point,), (direction,))[1])
        reverse = float(jnp.sum(jax.grad(function)(point) * direction))
        difference = float((function(point + step * direction)
                            - function(point - step * direction))
                           / (2 * step))
        assert abs(forward) > 1e-3
        assert abs(forward - reverse) < 1e-13 * abs(forward)
        assert abs(forward - difference) < 1e-6 * abs(forward)

    check(lambda e: occupations.fermi_level(e, wkpt, target, WIDTH, OCCMAX,
                                            DEEP, 3),
          jnp.asarray(evalsv), jnp.asarray(rng.normal(size=evalsv.shape)))
    check(lambda w: occupations.fermi_level(evalsv, w, target, WIDTH, OCCMAX,
                                            DEEP, 3),
          jnp.asarray(wkpt), jnp.asarray(rng.normal(size=wkpt.shape)))
    check(lambda n: occupations.fermi_level(evalsv, wkpt, n, WIDTH, OCCMAX,
                                            DEEP, 3),
          jnp.asarray(target), jnp.asarray(1.0))


def test_the_derivative_is_gauge_invariant_inside_a_multiplet():
    """Study 8(b)'s reason d(mu) is safe where an eigenvalue derivative is not.

    Inside a degenerate group the response `sdelta` is constant, so the sum
    over the group is a trace and cannot see the arbitrary basis `eigh` picks
    there.  Concretely: perturbing an exactly degenerate pair in a way that
    preserves its sum must leave mu's derivative unchanged.
    """
    import jax
    import jax.numpy as jnp
    from elkjax import occupations
    evalsv, wkpt = _spectrum()
    evalsv = evalsv.copy()
    evalsv[:, 5] = evalsv[:, 4]

    def mu(values):
        return occupations.fermi_level(values, wkpt, 6.0, WIDTH, OCCMAX,
                                       DEEP, 3)

    trace_preserving = np.zeros_like(evalsv)
    trace_preserving[:, 4], trace_preserving[:, 5] = 1.0, -1.0
    slope = float(jax.jvp(mu, (jnp.asarray(evalsv),),
                          (jnp.asarray(trace_preserving),))[1])
    assert abs(slope) < 1e-15
