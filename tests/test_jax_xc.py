r"""Phase 2 of the JAX port: the exchange-correlation functional.

`src/elkjax/xc.py` transcribes Elk's `xc_pwca.f90` -- Perdew-Wang's
parameterisation of the spin-polarised Ceperley-Alder electron gas, `xctype=3`
and Elk's own default.  These checks need no Elk run; the comparison against
Elk's own v_xc on a real grid is `tests/test_calculation_xc.py`.

Three of them are pinned against closed forms rather than against another
transcription, which matters because AD-versus-hand-coded agreement is an
INTERNAL check: it would pass just as well if the wrong functional had been
transcribed.

  * exchange against Dirac's exact LDA form, and v_x = (4/3) eps_x;
  * exchange spin scaling, rho eps_x(n_up, n_dn) = [2n_up eps_x(2n_up)
    + 2n_dn eps_x(2n_dn)] / 2, which is exact for any density functional of
    the exchange energy and exercises the zeta machinery the unpolarised
    case leaves at zero;
  * the high-density limit d(eps_c)/d(ln r_s) -> (1 - ln 2)/pi^2, the
    Gell-Mann-Brueckner coefficient (Phys. Rev. 106, 364 (1957)) that PW92's
    A_0 is fitted to reproduce -- an analytic anchor on the CORRELATION half,
    which the two exchange checks say nothing about.

Then the study's own Phase 2 gradient criterion: v_xc from `jax.grad` of
rho eps_xc against the hand-coded one, in both spin channels.  Elk
differentiates the parameterisation by hand through a long chain
(d/dr_s, d/dzeta, dr_s/drho, dzeta/drho_sigma) and AD does not.

And the hazard the study names: Elk's `rho < 1e-20` guard, transcribed as one
`jnp.where`, gives the correct VALUE and a NaN GRADIENT.

Skipped without jax.
"""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="jax not installed; pip install -e .[jax]")

DENSITIES = np.array([1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 1e3])


def test_exchange_is_the_dirac_form():
    """eps_x = -(3/4)(3/pi)^(1/3) rho^(1/3) and v_x = (4/3) eps_x, exactly."""
    import jax.numpy as jnp
    from elkjax import xc
    rho = jnp.asarray(DENSITIES)
    ex, _, vxup, vxdn, _, _ = xc.pwca(0.5 * rho, 0.5 * rho)
    exact = np.asarray(xc.dirac_exchange(rho))
    assert np.abs((np.asarray(ex) - exact) / exact).max() < 1e-15
    assert np.abs((np.asarray(vxup) - (4.0 / 3.0) * exact)
                  / exact).max() < 1e-15
    assert np.abs(np.asarray(vxup) - np.asarray(vxdn)).max() == 0.0


def test_exchange_obeys_the_exact_spin_scaling():
    """rho eps_x(n_up, n_dn) = [ (2n_up) eps_x(2n_up)
                               + (2n_dn) eps_x(2n_dn) ] / 2,
    and v_x,sigma(n_up, n_dn) = v_x(2 n_sigma).

    Exact for exchange at any polarisation, and the only check here that
    exercises the (1 +/- zeta)^(4/3) machinery -- which the unpolarised case
    leaves sitting at zeta = 0.
    """
    import jax.numpy as jnp
    from elkjax import xc
    rho = jnp.asarray(np.repeat(DENSITIES, 3))
    zeta = jnp.asarray(np.tile([0.0, 0.3, 0.85], len(DENSITIES)))
    up, dn = 0.5 * rho * (1 + zeta), 0.5 * rho * (1 - zeta)
    ex, _, vxup, vxdn, _, _ = xc.pwca(up, dn)
    scaled = 0.5 * (2 * up * xc.dirac_exchange(2 * up)
                    + 2 * dn * xc.dirac_exchange(2 * dn))
    left, right = np.asarray(rho * ex), np.asarray(scaled)
    assert np.abs((left - right) / right).max() < 1e-14
    for v, n in ((vxup, up), (vxdn, dn)):
        exact = (4.0 / 3.0) * np.asarray(xc.dirac_exchange(2 * n))
        assert np.abs((np.asarray(v) - exact) / exact).max() < 1e-14


def test_correlation_has_the_gell_mann_brueckner_high_density_limit():
    """d(eps_c)/d(ln r_s) -> (1 - ln 2)/pi^2 as r_s -> 0.

    The one analytic anchor available on the correlation half.  PW92's A_0 is
    fitted to this exactly, so a transcription that mangled the fitting
    function's structure (rather than its parameters) shows up here.
    """
    from elkjax import xc
    exact = (1.0 - np.log(2.0)) / np.pi ** 2
    assert abs(xc._A[0] - exact) < 1e-6

    def eps_c(rs):
        rho = (3.0 / (4.0 * np.pi)) / rs ** 3
        return float(xc.exc_density(rho) - xc.dirac_exchange(rho))

    errors = []
    for rs in (1e-3, 1e-4, 1e-5):
        h = 1e-4
        slope = (eps_c(rs * np.exp(h)) - eps_c(rs * np.exp(-h))) / (2 * h)
        errors.append(abs(slope - exact))
    assert errors[-1] < 1e-5, errors
    assert errors[0] > errors[-1], errors        # converging, not accidental


def test_autodiff_reproduces_the_hand_coded_potential():
    """The study's Phase 2 gradient criterion, in both spin channels.

    v_sigma = d(rho eps_xc)/d(rho_sigma).  Its stated tolerance is 1e-10; the
    measured agreement is machine precision.
    """
    import jax
    import jax.numpy as jnp
    from elkjax import xc
    rho = jnp.asarray(np.repeat(DENSITIES, 3))
    zeta = jnp.asarray(np.tile([0.0, 0.3, 0.85], len(DENSITIES)))
    up, dn = 0.5 * rho * (1 + zeta), 0.5 * rho * (1 - zeta)
    _, _, vxup, vxdn, vcup, vcdn = xc.pwca(up, dn)

    def energy(u, d):
        return jnp.sum((u + d) * xc.exc_from_spin_densities(u, d))

    gup, gdn = jax.grad(energy, argnums=(0, 1))(up, dn)
    for got, hand in ((gup, np.asarray(vxup) + np.asarray(vcup)),
                      (gdn, np.asarray(vxdn) + np.asarray(vcdn))):
        assert np.abs((np.asarray(got) - hand) / hand).max() < 1e-12


def test_the_naive_guard_gives_the_right_value_and_a_nan_gradient():
    """Elk's `if (rho < 1e-20) cycle`, transcribed as one `jnp.where`.

    `jnp.where` evaluates both branches, and the live one contains
    r_s ~ rho^(-1/3) whose derivative at rho = 0 is infinite; the VJP then
    multiplies that infinity by a zero cotangent.  A zero-initialised or
    zero-padded density array is exactly this case, which is why the guard
    has to substitute a safe argument BEFORE evaluating rather than select
    afterwards.

    The values must agree everywhere -- otherwise this would be a test of two
    different functions rather than of two ways to write one.
    """
    import jax
    import jax.numpy as jnp
    from elkjax import xc
    rho = jnp.asarray([0.0, 1e-30, 1e-22, 1e-19, 1e-6])

    def energy(fn):
        def inner(r):
            ex, ec = fn(0.5 * r, 0.5 * r)[:2]
            return jnp.sum(r * (ex + ec))
        return inner

    safe_val = np.asarray(energy(xc.pwca)(rho))
    naive_val = np.asarray(energy(xc.naive_pwca)(rho))
    assert np.array_equal(safe_val, naive_val)
    safe = np.asarray(jax.grad(energy(xc.pwca))(rho))
    naive = np.asarray(jax.grad(energy(xc.naive_pwca))(rho))
    assert np.isfinite(safe).all(), safe
    assert not np.isfinite(naive).all(), naive
    assert np.isnan(naive[0]) and safe[0] == 0.0
    assert np.allclose(safe[1:], naive[1:], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# PBE (xctype = 20)
# ---------------------------------------------------------------------------
#
# Only the ENERGY densities are transcribed; the potential is what `jax.grad`
# is supposed to supply.  These checks are the closed forms available without
# an Elk run; the exact comparison against Elk's own `exir`/`ecir`, and the
# functional derivative against Elk's hand-coded v_xc, are in
# tests/test_calculation_xc.py.


def test_pbe_reduces_to_lda_at_zero_gradient():
    """F_x(0) = 1 and H(t = 0) = 0, so PBE must return exactly the LDA.

    This is the one check that ties the GGA transcription to the LDA one,
    which is independently pinned against Dirac and Gell-Mann-Brueckner.
    """
    import jax.numpy as jnp
    from elkjax import xc
    rho = jnp.asarray(DENSITIES)
    zero = jnp.zeros_like(rho)
    ex, ec = xc.pbe(0.5 * rho, 0.5 * rho, zero, zero, zero)
    ex_lda, ec_lda = xc.pwca(0.5 * rho, 0.5 * rho)[:2]
    assert np.abs((np.asarray(ex) - np.asarray(ex_lda))
                  / np.asarray(ex_lda)).max() < 1e-14
    assert np.abs((np.asarray(ec) - np.asarray(ec_lda))
                  / np.asarray(ec_lda)).max() < 1e-13


def test_the_pbe_exchange_enhancement_obeys_its_two_defining_limits():
    """F_x -> 1 + mu s^2 as s -> 0 (the gradient expansion), and
    F_x -> 1 + kappa as s -> infinity (the Lieb-Oxford bound PBE is
    constructed to respect).

    Both are properties of the FORM, so they catch a mistyped kappa or mu --
    which the LDA limit above cannot, since it sends both to zero.
    """
    import jax.numpy as jnp
    from elkjax import xc
    rho = jnp.asarray(np.full(4, 0.1))
    # s is |grad rho_sigma| / (2 k_F rho_sigma) with k_F from 2 rho_sigma
    half = 0.5 * rho
    kf = (2.0 * half * 3.0 * np.pi ** 2) ** (1.0 / 3.0)
    for s_target, expected in ((1e-3, None), (1e4, 1.0 + xc.KAPPA_PBE)):
        grad = jnp.asarray(s_target) * 2.0 * kf * half
        ex, _ = xc.pbe(half, half, grad, grad, 2.0 * grad)
        lda = xc.dirac_exchange(rho)
        enhancement = np.asarray(ex) / np.asarray(lda)
        if expected is None:
            slope = (enhancement - 1.0) / s_target ** 2
            assert np.abs(slope - xc.MU_PBE).max() < 1e-6, slope
        else:
            assert np.abs(enhancement - expected).max() < 1e-6, enhancement


def test_pbe_is_safe_at_zero_density():
    """The same `rho -> 0` guard as the LDA, with Elk's own 1e-12 cutoff."""
    import jax
    import jax.numpy as jnp
    from elkjax import xc
    rho = jnp.asarray([0.0, 1e-20, 1e-13, 1e-6, 1.0])
    grad = 0.3 * rho

    def energy(r):
        ex, ec = xc.pbe(0.5 * r, 0.5 * r, 0.15 * r, 0.15 * r, 0.3 * r)
        return jnp.sum(r * (ex + ec))

    del grad
    value = np.asarray(energy(rho))
    assert np.isfinite(value).all()
    assert np.isfinite(np.asarray(jax.grad(energy)(rho))).all()
