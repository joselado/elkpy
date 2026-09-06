r"""The LAPW matching coefficients in JAX — Phase 0c of ``docs/jax_port.md``.

Study §6 item 0c: differentiate the APW matching coefficients with respect to an
atomic position and check the result against the analytic identity Elk already has in
``dmatch.f90``.  Inside muffin-tin :math:`\alpha`,

.. math::

    \phi^{\alpha}_{\bf G+p}({\bf r})=\sum_{lm}\sum_{j=1}^{M^{\alpha}_l}
    A^{\alpha}_{jlm}({\bf G+p})\,u^{\alpha}_{jl}(r)\,Y_{lm}(\hat{\bf r}),

and continuity of the plane wave :math:`e^{i({\bf G+p})\cdot{\bf r}}/\sqrt\Omega` and its
first :math:`M^{\alpha}_l-1` derivatives across :math:`r=R_\alpha` fixes
:math:`A^\alpha_{jlm}` through the small linear system :math:`\sum_j D_{ij}A_j=b_i` with

.. math::

    D_{ij}=\frac{d^{\,i-1}u^{\alpha}_{jl}}{dr^{\,i-1}}\bigg|_{R_\alpha},
    \qquad
    b_i=\frac{4\pi i^l}{\sqrt\Omega}\,|{\bf G+p}|^{i-1}
        j^{(i-1)}_l(|{\bf G+p}|R_\alpha)\,
        e^{i({\bf G+p})\cdot{\bf r}_\alpha}\,Y^*_{lm}(\widehat{{\bf G+p}}).

**The whole atomic-position dependence sits in one factor**, the structure factor
:math:`e^{i({\bf G+p})\cdot{\bf r}_\alpha}`; :math:`D` and the Bessel functions do not
move with the atom.  So ``dmatch.f90`` is a one-liner,
:math:`\partial A/\partial r_{\alpha,p} = i({\bf G+p})_p A`, and that exactness is what
makes it a sharp AD test rather than a plausibility check.

**On fidelity.**  ``spherical_harmonics`` and ``spherical_bessel`` are *re-derivations*
validated against SciPy, not transcriptions of Elk's own recursions — Elk's
``genylmv``/``sbessel`` are written for a Fortran loop nest with in-place rescaling, and
copying that shape into JAX would buy bit-identity at the cost of readability and
differentiability.  What *is* transcribed exactly is every **convention**: the packed
index :math:`lm=l(l+1)+m+1` (one-based; zero-based here), the Condon-Shortley phase, and
``genylmv``'s ``t4pil`` prefactor :math:`4\pi(-i)^l` — which is why ``match`` below
carries no explicit :math:`4\pi i^l` even though the formula above does, exactly as
Elk's own ``match.f90`` does not.

The derivative matrix :math:`D` is an **input** here.  Building it from ``apwfr``
requires the radial Schrödinger solutions and ``polynm``'s divided-difference fit, which
is Phase 1 work; item 0c is about the :math:`({\bf G+p})`-dependent half.

**Differentiability in** :math:`{\bf G+k}` **needed one more change, made in Phase 1f.**
Written the way :math:`b_i` is written above — a harmonic of a *direction* times a
Bessel function of a *length* — ``match`` has no derivative wherever
:math:`{\bf G+k}` lies on the :math:`z`-axis, from two independent causes:
:math:`Y_{lm}(\hat v)` is singular where :math:`\hat v` is undefined, and
:math:`\lvert{\bf G+k}\rvert` is :math:`\sqrt\cdot` at zero.  Since
:math:`{\bf G}=0` is in every basis, that is every reciprocal-lattice point.  The
*product* is smooth, so ``match`` now regroups it as a **regular solid harmonic**
:math:`r^lY_{lm}` (:func:`solid_harmonics`, a polynomial in the Cartesian components)
times :math:`j_l^{(i_o)}(x)x^{i_o-l}` (:func:`spherical_bessel_scaled`, an even function
of :math:`x` and therefore a function of
:math:`x^2=R^2({\bf G+k})\cdot({\bf G+k})`), and forms neither :math:`\hat v` nor
:math:`\lvert v\rvert` at all.  :func:`spherical_harmonics` and
:func:`spherical_bessel` are unchanged and still what item 0c checks.
"""

import functools
import math

import numpy as np

import jax
import jax.numpy as jnp

__all__ = [
    "lm_index",
    "spherical_harmonics",
    "spherical_bessel",
    "spherical_bessel_derivative",
    "gkvectors",
    "structure_factor",
    "match",
    "dmatch",
]

FOURPI = 4.0 * np.pi


def lm_index(l, m):
    """Elk's packed index, zero-based: ``lm = l(l+1)+m``."""
    return l * (l + 1) + m


def _safe_direction(v, eps=1e-14):
    r"""``(cos(theta), sin(theta), e^{i phi})`` for a Cartesian vector, guarded at the poles.

    Mirrors ``genylmv``'s own guards: on the z-axis the azimuth is undefined and Elk
    picks :math:`\phi=0`; at the origin it picks the +z direction.  The guards are
    written with the safe-denominator idiom so the gradient is finite there too.
    """
    r2 = jnp.sum(v * v)
    r = jnp.sqrt(jnp.where(r2 > eps ** 2, r2, 1.0))
    ct = jnp.where(r2 > eps ** 2, v[2] / r, 1.0)
    ct = jnp.clip(ct, -1.0, 1.0)
    st = jnp.sqrt(jnp.maximum(1.0 - ct ** 2, 0.0))
    rho2 = v[0] ** 2 + v[1] ** 2
    rho = jnp.sqrt(jnp.where(rho2 > eps ** 2, rho2, 1.0))
    on_axis = rho2 <= eps ** 2
    cp = jnp.where(on_axis, 1.0, v[0] / rho)
    sp = jnp.where(on_axis, 0.0, v[1] / rho)
    return ct, st, cp + 1j * sp


def spherical_harmonics(lmax, v, t4pil=True):
    r"""Elk's ``genylmv``: :math:`Y_{lm}(\hat v)` packed as ``lm = l(l+1)+m``.

    With ``t4pil`` (Elk's own default at every ``match`` call site) each :math:`l` block
    is multiplied by :math:`4\pi(-i)^l`, the prefactor of the Rayleigh expansion
    :math:`e^{-i{\bf G}\cdot{\bf r}}=4\pi\sum_l(-i)^lj_l(Gr)\sum_mY_{lm}(\hat{\bf G})
    Y^*_{lm}(\hat{\bf r})`.  Forgetting it is the trap study §6 names by
    name: the result stays smooth, correctly shaped and wrong by an :math:`l`-dependent
    complex factor, which cancels out of many ratios and out of every modulus.

    **Differentiable in** :math:`v` **away from the z-axis, and loudly not on it.**
    For :math:`m\neq0` the azimuthal factor :math:`e^{im\phi}` has no derivative where
    :math:`\phi` is undefined, and :math:`\sin\theta=\sqrt{1-\cos^2\theta}` has an
    infinite one at the pole.  Measured: ``jax.grad`` of :math:`\mathrm{Re}\,Y_{11}` at
    :math:`v=(0,0,1)` returns ``NaN`` — the *good* outcome, the failure being audible
    rather than a plausible finite number, unlike the padding-block case of §0b.  It
    does not affect item 0c, whose derivative is with respect to the atomic position and
    never touches :math:`\hat v`; it would affect Phase 4's stress, which moves
    :math:`{\bf G+p}` itself, and a G-vector lying exactly along z is not exotic.
    Pinned by a test rather than left to be discovered.
    """
    ct, st, phase = _safe_direction(v)
    size = (lmax + 1) ** 2
    values = [None] * size

    # normalised associated Legendre functions, Condon-Shortley phase included:
    # Y_lm = Pbar_l^m(cos theta) e^{i m phi},  Pbar_0^0 = 1/sqrt(4 pi)
    pmm = 1.0 / np.sqrt(FOURPI)
    powers = [jnp.ones((), dtype=jnp.complex128)]
    for m in range(1, lmax + 1):
        powers.append(powers[-1] * phase)
    for m in range(lmax + 1):
        if m > 0:
            pmm = -np.sqrt((2 * m + 1) / (2 * m)) * st * previous_pmm
        previous_pmm = pmm
        column = {m: pmm}
        if m + 1 <= lmax:
            column[m + 1] = np.sqrt(2 * m + 3) * ct * pmm
        for l in range(m + 2, lmax + 1):
            alpha = np.sqrt((4 * l ** 2 - 1) / (l ** 2 - m ** 2))
            beta = np.sqrt(((l - 1) ** 2 - m ** 2) / (4 * (l - 1) ** 2 - 1))
            column[l] = alpha * (ct * column[l - 1] - beta * column[l - 2])
        for l, plm in column.items():
            values[lm_index(l, m)] = plm * powers[m]
            if m > 0:
                values[lm_index(l, -m)] = ((-1) ** m) * jnp.conj(plm * powers[m])

    ylm = jnp.stack([jnp.asarray(value, dtype=jnp.complex128) for value in values])
    if not t4pil:
        return ylm
    prefactor = np.concatenate([
        np.full(2 * l + 1, FOURPI * (-1j) ** l) for l in range(lmax + 1)])
    return ylm * jnp.asarray(prefactor)


def spherical_bessel(lmax, x, extra=14):
    r"""Elk's ``sbessel``: :math:`j_l(x)` for :math:`l=0\ldots l_{\max}`.

    Three branches, the same three ``sbessel.f90`` has, and the split is not cosmetic:
    the upward recurrence :math:`j_l=(2l-1)x^{-1}j_{l-1}-j_{l-2}` is unstable for
    :math:`x<l` and Miller's downward recurrence loses accuracy for :math:`x\gg l`.
    Measured against SciPy at ``lmax=8``, downward alone is good to 2e-15 at
    :math:`x\le3` and wrong by **7.6e-2** at :math:`x=20`; upward alone fails at the
    other end.  Elk switches at :math:`x=l_{\max}` and so does this.

    Both branches are evaluated under ``jnp.where``, so every denominator is guarded —
    a discarded branch's ``NaN`` would still poison the gradient.  The small-:math:`x`
    series :math:`j_l\simeq x^l/(2l+1)!!` takes over below :math:`10^{-8}`, as in Elk.
    """
    lstart = lmax + lmax // 8 + extra
    small = jnp.abs(x) < 1e-8
    safe_x = jnp.where(small, 1.0, x)
    inverse = 1.0 / safe_x

    # Miller downward recurrence, normalised against j_0 = sin(x)/x
    j1, j0 = jnp.ones_like(safe_x), jnp.zeros_like(safe_x)
    for l in range(lstart, lmax, -1):
        j1, j0 = (2 * l + 1) * j1 * inverse - j0, j1
    unnormalised = [None] * (lmax + 1)
    for l in range(lmax, -1, -1):
        j1, j0 = (2 * l + 1) * j1 * inverse - j0, j1
        unnormalised[l] = j0
    scale = jnp.sin(safe_x) * inverse / unnormalised[0]
    downward = jnp.stack([scale * value for value in unnormalised])

    # upward recurrence from the closed forms, stable for x >= lmax
    upward = [jnp.sin(safe_x) * inverse]
    if lmax >= 1:
        upward.append((upward[0] - jnp.cos(safe_x)) * inverse)
    for l in range(2, lmax + 1):
        upward.append((2 * l - 1) * upward[l - 1] * inverse - upward[l - 2])
    upward = jnp.stack(upward)

    series, term = [], jnp.ones_like(safe_x)
    for l in range(lmax + 1):
        if l:
            term = term * x / (2 * l + 1)
        series.append(term)

    recurrence = jnp.where(safe_x < max(lmax, 1), downward, upward)
    return jnp.where(small, jnp.stack(series), recurrence)


def spherical_bessel_derivative(lmax, x, order):
    r""":math:`d^{\,n}j_l/dx^{\,n}`, Elk's ``sbesseldm``, by differentiating the recurrence.

    Elk builds these from a hand-derived polynomial-in-:math:`1/x` expansion; here they
    come from ``jax.jacfwd`` applied to :func:`spherical_bessel`, which is both shorter
    and a live test that the recurrence is differentiable.  ``order`` is small in
    practice — it runs to ``apwordmax-1``, i.e. 1 or 2.
    """
    function = lambda t: spherical_bessel(lmax, t)
    for _ in range(order):
        function = jax.jacfwd(function)
    return function(x)


@functools.lru_cache(maxsize=None)
def _solid_harmonic_series(lmax, terms=12):
    r"""Coefficients of :math:`P_{\ell,i_o}` as a polynomial in :math:`x^2`.

    From :math:`j_\ell(x)=\sum_n c_n(\ell)\,x^{\ell+2n}` with
    :math:`c_n=(-1)^n/(2^nn!\,(2\ell+2n+1)!!)`, differentiating :math:`i_o` times and
    multiplying by :math:`x^{i_o-\ell}` gives
    :math:`P_{\ell,i_o}(x)=\sum_n c_n(\ell)\,(\ell+2n)_{i_o}\,x^{2n}` with
    :math:`(a)_k` the falling factorial -- **even in** :math:`x`, hence a polynomial in
    :math:`x^2`, hence a smooth function of :math:`{\bf G+k}` with no square root in it.
    Returned as ``(order, terms, lmax+1)`` for orders 0..lmax so a caller can index,
    and cached: the loops below are pure Python and would otherwise run on every
    assembly, including inside a traced one.  The array is never mutated.
    """
    coefficients = np.zeros((lmax + 1, terms, lmax + 1))
    for l in range(lmax + 1):
        for n in range(terms):
            double_factorial = 1.0
            for k in range(2 * l + 2 * n + 1, 0, -2):
                double_factorial *= k
            base = (-1.0) ** n / (2.0 ** n * math.factorial(n) * double_factorial)
            for order in range(lmax + 1):
                falling = 1.0
                for i in range(order):
                    falling *= (l + 2 * n - i)
                coefficients[order, n, l] = base * falling
    return coefficients


def spherical_bessel_scaled(lmax, x2, order, threshold=0.1, terms=12):
    r""":math:`P_{\ell,i_o}(x)=j_\ell^{(i_o)}(x)\,x^{i_o-\ell}` as a function of :math:`x^2`.

    This is the piece that lets ``match`` avoid :math:`\lvert{\bf G+k}\rvert` entirely.
    The matching coefficient needs
    :math:`Y_{\ell m}(\hat g)\,g^{i_o}j_\ell^{(i_o)}(gR)`, and splitting off
    :math:`g^\ell` from the harmonic leaves exactly this — an **even, analytic**
    function, so it can be evaluated from :math:`x^2=R^2\,({\bf G+k})\cdot({\bf G+k})`,
    which is a polynomial in the Cartesian components and has no derivative
    singularity at the origin.  :math:`\sqrt{\cdot}` at zero is the *second* of the two
    poles that made ``jax.jvp`` of the assembly ``NaN`` at :math:`\Gamma`
    (`docs/jax_port_phase1.md` §1f), and it is invisible until the first is fixed.

    Two branches under ``jnp.where``, with the same guarded-denominator discipline as
    :func:`spherical_bessel`: below ``threshold`` the series above, and elsewhere
    :func:`spherical_bessel_derivative` divided by :math:`x^{\ell-i_o}`.  At
    :math:`x=0.1` the first neglected term is :math:`O(10^{-19})` relative, and the
    division branch never sees an :math:`x` smaller than ``threshold``, so neither
    branch is ever near its own edge.
    """
    small = x2 < threshold ** 2
    safe_x2 = jnp.where(small, threshold ** 2, x2)
    x = jnp.sqrt(safe_x2)
    if order == 0:
        radial = spherical_bessel(lmax, x)
    else:
        radial = spherical_bessel_derivative(lmax, x, order)
    exponents = np.arange(lmax + 1) - order            # l - i_o
    big = radial * x ** (-exponents)

    coefficients = _solid_harmonic_series(lmax, terms)[order]      # (terms, lmax+1)
    series = jnp.zeros_like(big)
    power = jnp.ones_like(x2)
    for n in range(terms):
        series = series + power * jnp.asarray(coefficients[n])
        power = power * x2
    return jnp.where(small, series, big)


def solid_harmonics(lmax, v, t4pil=True):
    r"""The **regular solid harmonics** :math:`r^\ell Y_{\ell m}({\bf \hat v})`, packed
    as ``lm = l(l+1)+m`` and carrying :func:`spherical_harmonics`' own
    :math:`4\pi(-i)^\ell` when ``t4pil``.

    Same recursion, three substitutions.  Writing
    :math:`Y_{\ell m}=\bar P_\ell^m(\cos\theta)e^{im\phi}` and multiplying through by
    :math:`r^\ell`, the seeds and steps of :func:`spherical_harmonics` become

    .. math::

        T_m = -\sqrt{\tfrac{2m+1}{2m}}\,(v_x + iv_y)\,T_{m-1},\quad
        S_{m+1,m} = \sqrt{2m+3}\,v_z\,T_m,\quad
        S_{\ell m} = \alpha\big(v_z S_{\ell-1,m} - \beta\,r^2 S_{\ell-2,m}\big),

    i.e. :math:`\cos\theta\to v_z`, :math:`\sin\theta\,e^{i\phi}\to v_x+iv_y` and
    :math:`\beta\to\beta r^2`, with the same :math:`\alpha,\beta` and the same
    :math:`S_{\ell,-m}=(-1)^m\overline{S_{\ell m}}`.  Every entry is then a
    **polynomial** in the Cartesian components: analytic at the origin and on the
    :math:`z`-axis, where :math:`Y_{\ell m}` itself has no derivative at all.

    Kept as a separate function rather than a refactor of
    :func:`spherical_harmonics`, deliberately.  That one is the transcription of
    ``genylmv`` that item 0c and the element-wise ``apwalm`` comparison check, and
    reordering its real/complex products would move its last bits for no gain.  The two
    are tied instead by a test asserting
    ``solid_harmonics(v) == spherical_harmonics(v) * |v|**l`` off-axis.
    """
    xy = v[0] + 1j * v[1]
    z = jnp.asarray(v[2], dtype=jnp.complex128)
    r2 = jnp.asarray(jnp.sum(v * v), dtype=jnp.complex128)
    size = (lmax + 1) ** 2
    values = [None] * size

    tmm = jnp.asarray(1.0 / np.sqrt(FOURPI), dtype=jnp.complex128)
    for m in range(lmax + 1):
        if m > 0:
            tmm = -np.sqrt((2 * m + 1) / (2 * m)) * xy * previous_tmm
        previous_tmm = tmm
        column = {m: tmm}
        if m + 1 <= lmax:
            column[m + 1] = np.sqrt(2 * m + 3) * z * tmm
        for l in range(m + 2, lmax + 1):
            alpha = np.sqrt((4 * l ** 2 - 1) / (l ** 2 - m ** 2))
            beta = np.sqrt(((l - 1) ** 2 - m ** 2) / (4 * (l - 1) ** 2 - 1))
            column[l] = alpha * (z * column[l - 1] - beta * r2 * column[l - 2])
        for l, slm in column.items():
            values[lm_index(l, m)] = slm
            if m > 0:
                values[lm_index(l, -m)] = ((-1) ** m) * jnp.conj(slm)

    slm = jnp.stack([jnp.asarray(value, dtype=jnp.complex128) for value in values])
    if not t4pil:
        return slm
    prefactor = np.concatenate([
        np.full(2 * l + 1, FOURPI * (-1j) ** l) for l in range(lmax + 1)])
    return slm * jnp.asarray(prefactor)


def gkvectors(vgc, vkc, gkmax):
    r"""Elk's ``gengkvec``: the :math:`{\bf G+k}` with :math:`|{\bf G+k}|<{\tt gkmax}`.

    Returned as ``(vgkc, gkc, mask)`` at fixed length rather than compacted, because a
    traced program cannot have a data-dependent ``ngk``.  §5's ``gk_mask`` is exactly
    this decision, and it is why the port pads rather than compacts.
    """
    vgkc = vgc + vkc[None, :]
    gkc = jnp.linalg.norm(vgkc, axis=1)
    return vgkc, gkc, gkc < gkmax


def structure_factor(vgkc, atposc):
    r"""Elk's ``gensfacgp``: :math:`S_\alpha({\bf G+k})=e^{i({\bf G+k})\cdot{\bf r}_\alpha}`.

    The only place an atomic position enters ``match`` at all.
    """
    return jnp.exp(1j * (vgkc @ atposc))


def match(lmax, vgkc, atposc, derivative_matrices, rmt, omega, t4pil=True):
    r"""Elk's ``match``, for one atom: the APW matching coefficients.

    ``derivative_matrices`` is a list indexed by :math:`l`, entry :math:`l` being the
    :math:`M_l\times M_l` matrix :math:`D_{ij}` of radial-function derivatives at
    :math:`R_\alpha` — Phase 1 supplies it from ``apwfr``; item 0c takes it as given.
    Returns ``apwalm`` with shape ``(ngk, apwordmax, (lmax+1)**2)``, zero-padded in the
    order axis exactly as Elk's array is.

    Follows ``match.f90`` including its two branches: the :math:`M_l=1` shortcut is a
    plain division rather than a 1x1 solve.  ``t4pil=False`` drops ``genylmv``'s
    :math:`4\pi(-i)^l` and is there so the trap can be *measured*, never as an option to
    use.

    **Written so that it is differentiable in** :math:`{\bf G+k}` **everywhere**,
    which the direct transcription is not.  Elk forms
    :math:`Y_{\ell m}(\widehat{\bf G+k})` and :math:`\lvert{\bf G+k}\rvert`
    separately, and both are singular at a basis function on the :math:`z`-axis —
    the harmonic because its argument is a *direction*, the modulus because
    :math:`\sqrt{\cdot}` has no derivative at zero.  That is not exotic:
    :math:`{\bf G}=0` is in every basis, so it is every reciprocal-lattice point, and
    in a slab cell :math:`{\bf G}=(0,0,\pm2\pi/c)` is in it too, so it is the whole
    :math:`k_z=0` plane.  Measured before the fix, ``jax.jvp`` of the assembled
    Hamiltonian in :math:`k` returned ``NaN`` at :math:`\Gamma` while its value there
    was exact (`docs/jax_port_phase1.md` §1f).

    The product that appears here,
    :math:`Y_{\ell m}(\hat g)\,g^{i_o}j_\ell^{(i_o)}(gR)`, has no such singularity —
    only its two factors do.  Splitting :math:`g^\ell` off the harmonic and onto the
    Bessel factor gives

    .. math::

        \overline{Y_{\ell m}(\hat g)}\;g^{i_o}j_\ell^{(i_o)}(gR)
        = \overline{S_{\ell m}({\bf G+k})}\;R^{\,\ell-i_o}\,P_{\ell,i_o}(x),
        \qquad x^2 = R^2\,({\bf G+k})\!\cdot\!({\bf G+k}),

    with :math:`S_{\ell m}=r^\ell Y_{\ell m}` the regular solid harmonic — a
    polynomial in the Cartesian components (:func:`solid_harmonics`) — and
    :math:`P_{\ell,i_o}(x)=j_\ell^{(i_o)}(x)x^{i_o-\ell}` an even analytic function of
    :math:`x`, hence a function of :math:`x^2` alone
    (:func:`spherical_bessel_scaled`).  **Neither** :math:`\hat g` **nor**
    :math:`\lvert g\rvert` **is ever formed**, which is why ``gkc`` is no longer an
    argument: passing it would reintroduce the second pole at the call site.

    The identity is exact, so the forward values are unchanged to rounding and the
    element-wise comparison against Elk's own ``apwalm`` is unaffected.
    """
    ngk = vgkc.shape[0]
    orders = [d.shape[0] for d in derivative_matrices]
    ordmax = max(orders)
    scale = 1.0 / jnp.sqrt(omega)

    slm = jax.vmap(lambda v: solid_harmonics(lmax, v, t4pil))(vgkc)   # (ngk, lmmax)
    sfac = structure_factor(vgkc, atposc)                             # (ngk,)
    x2 = jnp.sum(vgkc * vgkc, axis=1) * rmt ** 2                      # (ngk,)

    # radial[io, igp, l] = P_{l,io}(x), Elk's t2 accumulation with g^l divided out
    radial = jnp.stack([
        jax.vmap(lambda y, io=io: spherical_bessel_scaled(lmax, y, io))(x2)
        for io in range(ordmax)])                                     # (ordmax, ngk, l)

    columns = []
    for l in range(lmax + 1):
        order = orders[l]
        block = slice(l * l, (l + 1) ** 2)
        # R^{l - io}, the factor left over from moving g^l across
        powers = jnp.asarray(float(rmt) ** (l - np.arange(order)))
        # b[i, igp, m] = t0 S_alpha conj(S_lm) R^{l-i} P_{l,i}(x)
        target = (scale * sfac)[None, :, None] * jnp.conj(slm[:, block])[None, :, :]
        target = target * (radial[:order, :, l] * powers[:, None])[:, :, None]
        if order == 1:
            solved = target / derivative_matrices[l][0, 0]
        else:
            flat = target.reshape(order, -1)
            solved = jnp.linalg.solve(derivative_matrices[l], flat)
            solved = solved.reshape(order, ngk, 2 * l + 1)
        padded = jnp.zeros((ordmax, ngk, 2 * l + 1), dtype=jnp.complex128)
        columns.append(padded.at[:order].set(solved))
    return jnp.concatenate(columns, axis=2).transpose(1, 0, 2)


def dmatch(vgkc, apwalm, direction):
    r"""Elk's ``dmatch``: :math:`\partial A/\partial r_{\alpha,p}=i({\bf G+k})_p A`.

    Exact, and the reference item 0c is measured against.  ``direction`` is the Cartesian
    component :math:`p`.
    """
    return 1j * vgkc[:, direction][:, None, None] * apwalm
