r"""Phase 2 of the JAX port: the exchange-correlation functional.

Elk's `xc_pwca.f90` -- the Perdew-Wang parameterisation of the spin-polarised
Ceperley-Alder electron gas (J. P. Perdew and Y. Wang, *Phys. Rev. B* **45**,
13244 (1992); D. M. Ceperley and B. J. Alder, *Phys. Rev. Lett.* **45**, 566
(1980)), which is `xctype = 3`, elkpy's ``xc="PW"`` and Elk's own default.

It is transcribed here in two forms, and the difference between them is the
point.  `pwca` returns the energy densities and the hand-coded potentials
exactly as Elk does; `exc_density` returns the energy density alone, so that
:math:`v_{xc}=\partial(\rho\varepsilon_{xc})/\partial\rho` can be obtained by
`jax.grad` and compared against the hand-coded one.  That comparison is
`docs/jax_port.md`'s stated Phase 2 gradient criterion, and it is a real check:
Elk differentiates the parameterisation by hand through a long chain
(:math:`d\varepsilon/dr_s`, :math:`d\varepsilon/d\zeta`, :math:`dr_s/d\rho`,
:math:`d\zeta/d\rho_\sigma`) and AD does not.

**The** :math:`\rho\to0` **guard is the hazard, and it is not hypothetical.**
`xc_pwca` returns zero for :math:`\rho<10^{-20}`, which the obvious
transcription writes as ``jnp.where(rho < 1e-20, 0.0, f(rho))``.  That gives
the correct VALUE and a `NaN` GRADIENT: `jnp.where` evaluates both branches,
and :math:`f` contains :math:`r_s\propto\rho^{-1/3}`, whose derivative at
:math:`\rho=0` is infinite -- and `where`'s vector-Jacobian product multiplies
that infinity by a zero cotangent.  `naive_pwca` keeps that form deliberately,
so the failure can be asserted rather than described; `pwca` uses the
double-`where` (substitute a safe argument *before* evaluating, then select).
Every interstitial point of a slab calculation with vacuum sits in that branch.

**One edge is left as Elk leaves it.**  At full polarisation
(:math:`\rho_\downarrow=0`, :math:`\zeta=1`) the exchange term forms
:math:`(1-\zeta)^{4/3}/(1-\zeta)`, which is :math:`0/0`.  Elk's own code has
the same expression and the same behaviour, and no Elk calculation reaches it
(a Kohn-Sham density is nowhere exactly spin-pure), so it is documented rather
than patched -- patching it would make this transcription disagree with the
reference it is checked against.
"""

import numpy as np

import jax.numpy as jnp

# xc_pwca.f90's own parameters -- Table I of Perdew & Wang 1992, in Hartree
_A = (0.0310907, 0.01554535, 0.0168869)
_A1 = (0.21370, 0.20548, 0.11125)
_B1 = (7.5957, 14.1189, 10.357)
_B2 = (3.5876, 6.1977, 3.6231)
_B3 = (1.6382, 3.3662, 0.88026)
_B4 = (0.49294, 0.62517, 0.49671)
_D2F0 = 1.709921            # f''(0), the spin-stiffness normalisation
_THRD = 1.0 / 3.0
_THRD4 = 4.0 / 3.0
RHO_MIN = 1.0e-20           # xc_pwca's own cutoff


def _g(index, rs, rs12, rs32, rs2, rs12i):
    """The PW92 fitting function and its :math:`r_s` derivative.

    .. math::
       G(r_s) = -2A(1+\\alpha_1 r_s)\\,
                \\ln\\!\\left[1 + \\frac{1}{2A\\bigl(\\beta_1r_s^{1/2}
                + \\beta_2r_s + \\beta_3r_s^{3/2}
                + \\beta_4r_s^{2}\\bigr)}\\right]

    used three times: for :math:`\\varepsilon_c(r_s,0)`,
    :math:`\\varepsilon_c(r_s,1)` and (with the sign flipped) the spin
    stiffness :math:`-\\alpha_c(r_s)`.
    """
    a2 = 2.0 * _A[index]
    t1 = a2 * (_B1[index] * rs12 + _B2[index] * rs + _B3[index] * rs32
               + _B4[index] * rs2)
    dt1 = a2 * (0.5 * _B1[index] * rs12i + _B2[index]
                + 1.5 * _B3[index] * rs12 + 2.0 * _B4[index] * rs)
    t3 = 1.0 / t1
    t2 = 1.0 + t3
    dt2 = -dt1 * t3 ** 2
    t3 = 1.0 / t2
    t4 = 1.0 + _A1[index] * rs
    t5 = jnp.log(t2)
    return -a2 * t4 * t5, -a2 * (_A1[index] * t5 + t4 * t3 * dt2)


def _pwca_core(rup, rdn, r):
    """The body of `xc_pwca`'s loop, with `r` the density it may safely use.

    Split out so the guard can substitute a safe `r` before this is evaluated
    rather than selecting afterwards -- the double-`where` that keeps the
    gradient finite.
    """
    p1 = (3.0 / (4.0 * np.pi)) ** _THRD
    p2 = (3.0 / (4.0 * np.pi)) * (9.0 * np.pi / 4.0) ** _THRD
    p3 = 1.0 / (2.0 ** _THRD4 - 2.0)
    ri = 1.0 / r
    ri2 = ri ** 2
    rs = p1 * ri ** _THRD
    rs2 = rs ** 2
    rs12 = jnp.sqrt(rs)
    rs32 = rs12 * rs
    rsi = 1.0 / rs
    rs12i = 1.0 / rs12
    mz = rup - rdn
    z = mz * ri
    z3 = z ** 3
    z4 = z3 * z
    drs = -_THRD * rs * ri
    t1 = mz * ri2
    dzu = ri - t1
    dzd = -ri - t1
    # ---- exchange
    t1 = -p2 * rsi / 2.0
    t2 = 1.0 + z
    t3 = 1.0 - z
    t4 = t2 ** _THRD4
    t5 = t3 ** _THRD4
    t6 = t4 + t5
    ex = t1 * t6
    ders = -ex * rsi
    fz = p3 * (t6 - 2.0)
    t4 = t4 / t2
    t5 = t5 / t3
    t6 = t4 - t5
    t7 = _THRD4 * t6
    dez = t1 * t7
    dfz = p3 * t7
    t1 = ders * drs
    vxup = ex + r * (t1 + dez * dzu)
    vxdn = ex + r * (t1 + dez * dzd)
    # ---- correlation
    ec0, dec0 = _g(0, rs, rs12, rs32, rs2, rs12i)
    ec1, dec1 = _g(1, rs, rs12, rs32, rs2, rs12i)
    ac, dac = _g(2, rs, rs12, rs32, rs2, rs12i)
    ac, dac = -ac, -dac
    t1 = 1.0 - z4
    t2 = (fz / _D2F0) * t1
    t3 = ec1 - ec0
    t4 = fz * z4
    ec = ec0 + ac * t2 + t3 * t4
    t5 = dec1 - dec0
    ders = dec0 + dac * t2 + t5 * t4
    t6 = 4.0 * fz * z3
    dez = (ac / _D2F0) * (dfz * t1 - t6) + t3 * (dfz * z4 + t6)
    t1 = ders * drs
    vcup = ec + r * (t1 + dez * dzu)
    vcdn = ec + r * (t1 + dez * dzd)
    return ex, ec, vxup, vxdn, vcup, vcdn


def pwca(rhoup, rhodn):
    """`xc_pwca`: returns `(ex, ec, vxup, vxdn, vcup, vcdn)`.

    Safe at :math:`\\rho\\to0` in value AND in gradient -- see the module
    docstring for why that needs the double-`where` rather than one.
    """
    rup, rdn = jnp.asarray(rhoup), jnp.asarray(rhodn)
    r = rup + rdn
    live = (rup >= 0.0) & (rdn >= 0.0) & (r >= RHO_MIN)
    safe = jnp.where(live, r, 1.0)
    # the spin split has to be made safe too, or z = (rup - rdn)/r is 0/0
    half = 0.5 * safe
    rup_s = jnp.where(live, rup, half)
    rdn_s = jnp.where(live, rdn, half)
    out = _pwca_core(rup_s, rdn_s, safe)
    return tuple(jnp.where(live, value, 0.0) for value in out)


def naive_pwca(rhoup, rhodn):
    """The transcription that gets the VALUE right and the GRADIENT wrong.

    A single `jnp.where` around the evaluated expression, which is how Elk's
    `if (r < 1e-20) cycle` reads if translated literally.  Kept so the failure
    can be asserted; never use it.
    """
    rup, rdn = jnp.asarray(rhoup), jnp.asarray(rhodn)
    r = rup + rdn
    live = (rup >= 0.0) & (rdn >= 0.0) & (r >= RHO_MIN)
    out = _pwca_core(rup, rdn, r)
    return tuple(jnp.where(live, value, 0.0) for value in out)


def exc_density(rho, zeta=0.0):
    """:math:`\\varepsilon_{xc}` alone, as a function of the TOTAL density.

    This is what `jax.grad` differentiates: :math:`v_{xc}` is
    :math:`\\partial(\\rho\\varepsilon_{xc})/\\partial\\rho` at fixed
    polarisation only when :math:`\\zeta=0`; for a spin-polarised point the
    two potentials are the partial derivatives with respect to
    :math:`\\rho_\\uparrow` and :math:`\\rho_\\downarrow` separately, which is
    what `exc_from_spin_densities` exposes.
    """
    rho = jnp.asarray(rho)
    up = 0.5 * rho * (1.0 + zeta)
    dn = 0.5 * rho * (1.0 - zeta)
    ex, ec = pwca(up, dn)[:2]
    return ex + ec


def exc_from_spin_densities(rhoup, rhodn):
    """:math:`\\varepsilon_{xc}(\\rho_\\uparrow,\\rho_\\downarrow)`."""
    ex, ec = pwca(rhoup, rhodn)[:2]
    return ex + ec


def dirac_exchange(rho):
    """The exact LDA exchange energy density,
    :math:`\\varepsilon_x = -\\tfrac34(3/\\pi)^{1/3}\\rho^{1/3}`
    (P. A. M. Dirac, *Proc. Camb. Phil. Soc.* **26**, 376 (1930)).

    Independent of the parameterisation, so it pins `pwca`'s exchange half
    against a closed form rather than against another transcription."""
    rho = jnp.asarray(rho)
    return -0.75 * (3.0 / np.pi) ** _THRD * rho ** _THRD
