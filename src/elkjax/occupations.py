r"""``occupy.f90``: the zone-summed Fermi level and the occupation numbers.

Phase 1i built :math:`\mu` at a **single** k-point, where the k-point weights
cancel and the electron-count constraint is one sum over one spectrum.  A
self-consistent field needs it over the whole zone,

.. math::

    \sum_{\bf k} w_{\bf k}\sum_i n_{\max}\,
    \tilde\Theta\!\Big(\frac{\mu-\varepsilon_{i\bf k}}{\sigma}\Big)
    \;=\; N_{\rm val},

with :math:`w_{\bf k}` Elk's own ``wkpt`` (which already carry the symmetry
reduction), :math:`n_{\max}` its ``occmax`` -- 2 without spin polarisation, 1
with -- and :math:`\tilde\Theta` the smeared step selected by ``stype``.  That
is the one piece of Fortran standing between the two half-steps
`docs/jax_port_phase2.md` §2i and §2j already close: everything else in the
loop exists.

**The derivative is the reason this is not just a bisection.**  Differentiating
the constraint at fixed :math:`N` gives

.. math::

    d\mu \;=\;
    \frac{\sum_{i\bf k} w_{\bf k}\,\tilde\delta_{i\bf k}\,d\varepsilon_{i\bf k}}
         {\sum_{i\bf k} w_{\bf k}\,\tilde\delta_{i\bf k}} ,
    \qquad \tilde\delta=\tilde\Theta' ,

so :math:`n_{\max}` and :math:`\sigma` cancel and the weights do **not** -- which
is exactly the half of study §8(b)'s rule that a single k-point cannot test.  It
is installed as a ``custom_jvp`` over a bisection that is never differentiated,
the Phase 0a lesson about unrolled solvers in miniature.

**The denominator vanishing is physics.**  In a gap at small ``swidth`` no state
responds, the electron count stops determining :math:`\mu`, and :math:`d\mu` is
genuinely :math:`0/0` -- :func:`check_fermi_level_determined` refuses rather
than returning the ratio of two rounding errors.  This is the zone-summed
sibling of ``projector.check_fermi_level_determined``.

Elk's own ``occupy`` is transcribed rather than replaced, cutoffs included: the
bisection brackets :math:`[\min\varepsilon,\max\varepsilon]`, stops when the
bracket is below :math:`10^{-12}` Ha, and returns the **last midpoint tested**
rather than the midpoint of the final bracket -- which is what ``occsv`` is
consistent with, so the two must be produced together.
"""

import functools
import math

import numpy as np

import jax
import jax.numpy as jnp

__all__ = ["stheta", "sdelta", "occupations", "electron_count", "fermi_level",
           "response", "check_fermi_level_determined", "inputs", "occupy"]

SQRT_PI = 1.7724538509055160273
BRACKET_TOL = 1.0e-12
MAXIT = 1000


# --------------------------------------------------------- the smearing functions


def stheta_fd(x):
    r"""``stheta_fd``: :math:`\tilde\Theta(x)=1/(1+e^{-x})`.

    Elk's hard cutoffs at :math:`|x|>50` are kept, and the exponent is clipped
    *before* the exponential so the discarded branch cannot overflow -- the
    ``jnp.where`` hazard §2a already turned into an assertion.  Outside the
    cutoff the function is constant in Elk too, so the zero gradient there is
    a transcription and not an approximation.
    """
    xs = jnp.clip(x, -50.0, 50.0)
    value = 1.0 / (1.0 + jnp.exp(-xs))
    return jnp.where(x > 50.0, 1.0, jnp.where(x < -50.0, 0.0, value))


def sdelta_fd(x):
    r"""``sdelta_fd``: :math:`\tilde\delta(x)=e^{-x}/(1+e^{-x})^2`."""
    xs = jnp.clip(x, -50.0, 50.0)
    t = jnp.exp(-xs)
    return jnp.where(jnp.abs(x) > 50.0, 0.0, t / (1.0 + t) ** 2)


def hermite(n, x):
    """``hermite``: the physicists' :math:`H_n(x)`, by the same recursion."""
    if n == 0:
        return jnp.ones_like(x)
    h2, h1 = jnp.ones_like(x), 2.0 * x
    for i in range(2, n + 1):
        h1, h2 = 2.0 * (x * h1 - (i - 1) * h2), h1
    return h1


def stheta_mp(n, x):
    r"""``stheta_mp``: the Methfessel-Paxton step of order ``n``.

    :math:`n=0` is plain Gaussian smearing, :math:`\tfrac12(1+\mathrm{erf}\,x)`.
    Elk's :math:`|x|>12` cutoffs are kept for the same reason as
    :func:`stheta_fd`'s.
    """
    total = 0.5 * (1.0 + jax.scipy.special.erf(x))
    if n:
        t0 = -jnp.exp(-jnp.clip(x, -12.0, 12.0) ** 2) / SQRT_PI
        for i in range(1, n + 1):
            t1 = t0 / (float(math.factorial(i)) * float(4 ** i))
            total = total + (-t1 if i % 2 else t1) * hermite(2 * i - 1, x)
        total = jnp.where(x < -12.0, 0.0, jnp.where(x > 12.0, 1.0, total))
    return total


def sdelta_mp(n, x):
    r"""``sdelta_mp``: the Methfessel-Paxton delta of order ``n``."""
    if n == 0:
        return jnp.exp(-x ** 2) / SQRT_PI
    t0 = jnp.exp(-jnp.clip(x, -12.0, 12.0) ** 2) / SQRT_PI
    total = t0
    for i in range(1, n + 1):
        t1 = t0 / (float(math.factorial(i)) * float(4 ** i))
        total = total + (-t1 if i % 2 else t1) * hermite(2 * i, x)
    return jnp.where(jnp.abs(x) > 12.0, 0.0, total)


def stheta(stype, x):
    """``stheta``: the smeared step Elk's ``stype`` selects.

    Types 0-2 are Methfessel-Paxton (0 being Gaussian) and 3 is Fermi-Dirac,
    which is Elk's default and the only one with an entropy term.  Types 4 and
    5 (square-wave and Lorentzian) are not transcribed; asking for one raises
    rather than silently returning a Fermi-Dirac occupation.
    """
    if stype == 3:
        return stheta_fd(x)
    if stype in (0, 1, 2):
        return stheta_mp(stype, x)
    raise ValueError(f"stype={stype} is not transcribed; use 0, 1, 2 or 3")


def sdelta(stype, x):
    """``sdelta``: the smeared delta matching :func:`stheta`."""
    if stype == 3:
        return sdelta_fd(x)
    if stype in (0, 1, 2):
        return sdelta_mp(stype, x)
    raise ValueError(f"stype={stype} is not transcribed; use 0, 1, 2 or 3")


# ------------------------------------------------------------ occupancies and count


def occupations(evalsv, mu, swidth, occmax, e0min, stype=3):
    r""":math:`n_{i\bf k}=n_{\max}\tilde\Theta((\mu-\varepsilon_{i\bf k})/\sigma)`.

    ``evalsv`` is ``(nkpt, nstsv)``, as the ``DENSITYK`` export holds it, and
    the result has the same shape.

    **The** ``e0min`` **gate is not decoration.**  Elk zeroes the occupancy of
    any state below the minimum linearisation energy (the deepest ``apwe0`` or
    ``lorbe0``, minus 2 Ha) rather than filling it: such a state is a ghost of
    the linearisation, not an electron.  Dropping the gate changes the electron
    count and therefore :math:`\mu`.
    """
    x = (mu - jnp.asarray(evalsv)) / swidth
    return jnp.where(jnp.asarray(evalsv) < e0min, 0.0,
                     occmax * stheta(stype, x))


def electron_count(occsv, wkpt):
    r""":math:`\sum_{\bf k}w_{\bf k}\sum_i n_{i\bf k}`."""
    return jnp.sum(jnp.asarray(wkpt)[:, None] * jnp.asarray(occsv))


def response(evalsv, wkpt, mu, swidth, e0min, stype=3):
    r""":math:`\sum_{i\bf k}w_{\bf k}\tilde\delta((\mu-\varepsilon_{i\bf k})/\sigma)`.

    The denominator of :math:`d\mu`, and -- times :math:`n_{\max}/\sigma` --
    Elk's own ``fermidos``, the density of states at the Fermi surface.  It is
    the quantity that decides whether :math:`\mu` is differentiable at all.
    """
    evalsv = jnp.asarray(evalsv)
    x = (mu - evalsv) / swidth
    return jnp.sum(jnp.asarray(wkpt)[:, None]
                   * jnp.where(evalsv < e0min, 0.0, sdelta(stype, x)))


# ------------------------------------------------------------- the Fermi level


@functools.partial(jax.custom_jvp, nondiff_argnums=(3, 4, 5, 6))
def fermi_level(evalsv, wkpt, chgval, swidth, occmax, e0min, stype=3):
    r""":math:`\mu` from the zone-summed electron count, by Elk's own bisection.

    ``swidth``, ``occmax``, ``e0min`` and ``stype`` are static (they come off
    the export as Python scalars); ``evalsv``, ``wkpt`` and ``chgval`` are the
    differentiable arguments.

    The primal is a bisection and is **never** differentiated -- that is what
    the ``custom_jvp`` is for.  The bracket, the stopping rule and the returned
    value are Elk's: the last midpoint *tested*, which is the one ``occsv`` was
    filled from, not the midpoint of the final bracket.
    """
    evalsv = jnp.asarray(evalsv)
    lo, hi = jnp.min(evalsv), jnp.max(evalsv)

    def condition(state):
        it, lo, hi, _ = state
        return (it == 0) | ((hi - lo >= BRACKET_TOL) & (it < MAXIT))

    def body(state):
        it, lo, hi, _ = state
        mid = 0.5 * (lo + hi)
        count = electron_count(
            occupations(evalsv, mid, swidth, occmax, e0min, stype), wkpt)
        low = count < chgval
        return (it + 1, jnp.where(low, mid, lo), jnp.where(low, hi, mid), mid)

    _, _, _, mu = jax.lax.while_loop(condition, body, (0, lo, hi, 0.5 * (lo + hi)))
    return mu


@fermi_level.defjvp
def _fermi_level_jvp(swidth, occmax, e0min, stype, primals, tangents):
    r"""Study §8(b)'s rule, with the k-point weights that a single k cannot test.

    Differentiating :math:`\sum w_{\bf k}n_{\max}\tilde\Theta((\mu-\varepsilon)/\sigma)=N`,

    .. math::

        d\mu = \frac{\sum_{i\bf k}w_{\bf k}\tilde\delta_{i\bf k}\,
                     d\varepsilon_{i\bf k}
                   + \frac{\sigma}{n_{\max}}\big(dN
                   - n_{\max}\sum_{i\bf k}\tilde\Theta_{i\bf k}\,dw_{\bf k}\big)}
               {\sum_{i\bf k}w_{\bf k}\tilde\delta_{i\bf k}} ,

    gauge-invariant at a degeneracy for the reason study §8(b) gives -- inside a
    multiplet :math:`\tilde\delta` is constant, so the numerator's sum over the
    group is a trace.
    """
    evalsv, wkpt, chgval = primals
    devalsv, dwkpt, dchgval = tangents
    mu = fermi_level(evalsv, wkpt, chgval, swidth, occmax, e0min, stype)

    evalsv = jnp.asarray(evalsv)
    gate = evalsv >= e0min
    delta = jnp.where(gate, sdelta(stype, (mu - evalsv) / swidth), 0.0)
    theta = jnp.where(gate, stheta(stype, (mu - evalsv) / swidth), 0.0)
    weights = jnp.asarray(wkpt)[:, None]

    numerator = jnp.sum(weights * delta * devalsv)
    numerator = numerator + (swidth / occmax) * (
        dchgval - occmax * jnp.sum(theta * jnp.asarray(dwkpt)[:, None]))
    return mu, numerator / jnp.sum(weights * delta)


def check_fermi_level_determined(evalsv, wkpt, chgval, swidth, occmax, e0min,
                                 stype=3, tol=1e-8):
    r"""Refuse a Fermi level the electron count does not determine.

    Returns :math:`(\mu,\ \sum w_{\bf k}\tilde\delta)`; raises ``ValueError``
    when the sum is below ``tol``.  That is the gapped case at small ``swidth``:
    no state lies within a few smearing widths of :math:`\mu`, every occupancy
    is exactly 0 or :math:`n_{\max}`, and :math:`\mu` may be placed anywhere in
    the gap -- so its derivative is not a number.  The forward loop is
    unaffected; only the gradient is refused.
    """
    mu = float(fermi_level(evalsv, wkpt, chgval, swidth, occmax, e0min, stype))
    total = float(response(evalsv, wkpt, mu, swidth, e0min, stype))
    if total < tol:
        raise ValueError(
            f"sum w_k delta = {total:.3e} is below {tol:.3e}: no state lies "
            f"within a few smearing widths of mu = {mu:.6f}, so the electron "
            f"count does not determine the Fermi level and dmu is 0/0. This "
            f"is a gapped system at swidth={swidth:g}; the occupations are "
            f"exact integers and a hard window is the differentiable route.")
    return mu, total


# ------------------------------------------------------------------- the export


def inputs(densityk):
    """``occupy``'s scalars off the ``DENSITYK`` export (patch 0023), as a dict.

    ``swidth``, ``occmax``, ``e0min`` and ``stype`` are returned as Python
    scalars because :func:`fermi_level` takes them as static arguments.
    """
    return dict(wkpt=np.asarray(densityk["wkpt"]),
                chgval=float(densityk["chgval"]),
                swidth=float(densityk["swidth"]),
                occmax=float(densityk["occmax"]),
                e0min=float(densityk["e0min"]),
                stype=int(densityk["stype"]))


def occupy(evalsv, densityk):
    """All of ``occupy.f90`` this port needs: ``(mu, occsv)`` from eigenvalues.

    ``evalsv`` is ``(nkpt, nstsv)``.  The two are returned together because
    Elk's own ``occsv`` belongs to the last :math:`\\mu` its bisection tested,
    so recomputing one without the other would not reproduce Elk.
    """
    got = inputs(densityk)
    mu = fermi_level(evalsv, got["wkpt"], got["chgval"], got["swidth"],
                     got["occmax"], got["e0min"], got["stype"])
    return mu, occupations(evalsv, mu, got["swidth"], got["occmax"],
                           got["e0min"], got["stype"])
