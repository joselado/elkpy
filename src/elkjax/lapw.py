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
"""

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


def match(lmax, vgkc, gkc, atposc, derivative_matrices, rmt, omega,
          t4pil=True):
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
    """
    ngk = gkc.shape[0]
    orders = [d.shape[0] for d in derivative_matrices]
    ordmax = max(orders)
    scale = 1.0 / jnp.sqrt(omega)

    ylm = jax.vmap(lambda v: spherical_harmonics(lmax, v, t4pil))(vgkc)  # (ngk, lmmax)
    sfac = structure_factor(vgkc, atposc)                          # (ngk,)
    argument = gkc * rmt

    # djl[io, l, igp] = |G+k|^{io} j_l^{(io)}(|G+k| R), Elk's own t2 accumulation
    djl = [jax.vmap(lambda t: spherical_bessel(lmax, t))(argument)]
    for io in range(1, ordmax):
        higher = jax.vmap(lambda t: spherical_bessel_derivative(lmax, t, io))(argument)
        djl.append(higher * (gkc ** io)[:, None])
    djl = jnp.stack(djl)                                           # (ordmax, ngk, lmax+1)

    columns = []
    for l in range(lmax + 1):
        order = orders[l]
        block = slice(l * l, (l + 1) ** 2)
        # b[i, igp, m] = t0 S_alpha conj(Y_lm) |G+k|^{i} j_l^{(i)}(|G+k| R)
        target = (scale * sfac)[None, :, None] * jnp.conj(ylm[:, block])[None, :, :]
        target = target * djl[:order, :, l][:, :, None]
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
