r"""Phase 2 item 2c: Elk's Weinert Poisson solve, transcribed.

The Coulomb potential is the one ingredient of the Kohn-Sham potential that no
export supplies as a *function of the density* -- :mod:`elkjax.xc` covers the
exchange-correlation half pointwise, but ``vclmt``/``vclir`` come out of
``potcoul`` and nothing else.  Without it the SCF loop cannot close.

**The method.**  A muffin-tin density is not a small perturbation of anything, so
the interstitial Poisson equation cannot be solved by Fourier transforming the true
density: its Fourier series converges hopelessly slowly near a nucleus.  Weinert's
construction (*J. Math. Phys.* **22**, 2433 (1981)) replaces the charge inside each
sphere by a smooth **pseudocharge** with the same multipole moments
:math:`q_{lm}`, whose Fourier series converges fast; outside the spheres it
produces the identical potential, since a multipole expansion only knows the
moments.  The intra-sphere problem is then solved exactly on the radial mesh and
matched to the interstitial solution at :math:`R_{\rm MT}` by adding the harmonic
function :math:`r^lY_{lm}` that fixes the boundary value.

Elk does it in four steps (``potcoul.f90``), and this module is one function per
step:

1. :func:`real_to_complex` -- the real-harmonic density to complex harmonics
   (``rtozfmt``), which is the basis the multipole algebra is written in.
2. :func:`intra_sphere` -- ``zpotclmt``, the exact radial solution

   .. math::

       V_{lm}(r)=\frac{4\pi}{2l+1}\Big[r^{-l-1}\!\!\int_0^{r}\!\!\rho_{lm}r'^{l+2}dr'
       + r^{l}\!\!\int_r^{R}\!\!\rho_{lm}r'^{1-l}dr'\Big],

   with both integrals done by ``wsplint``'s cumulative spline weights.
3. the nuclear potential added to the :math:`l=0` channel
   (:func:`add_nuclear`) -- note this happens BEFORE the multipoles are read, so
   :math:`q_{00}` carries :math:`-Z` and a transcription that adds the nucleus
   later gets every moment wrong.
4. :func:`pseudocharge_solve` -- ``zpotcoul``: read :math:`q_{lm}` off the
   sphere-boundary value, subtract what the interstitial density already
   contributes there, add the pseudocharge in :math:`G`-space, divide by
   :math:`G^2`, and add the matching harmonic inside each sphere.

:func:`coulomb_potential` composes them and returns Elk's own ``vclmt``/``vclir``.

**What is exported and what is rebuilt.**  Patch 0017 adds only ``wprmt`` (the
spline weights, which have no closed form worth retyping), ``vcln`` (the nuclear
potential, from Elk's finite-nucleus model), ``npsd``/``lnpsd`` and ``atposc``.
Everything else is rebuilt here from what the ``GROUNDSTATE`` query already
carries: :math:`r^l` and :math:`R^l` from the mesh, :math:`4\pi/G^2` from ``gc``,
and :math:`Y_{lm}(\hat G)`, :math:`e^{i{\bf G}\cdot{\bf r}_\alpha}` and
:math:`j_l(GR)` from :mod:`elkjax.lapw`, whose transcriptions Phase 0c already
pinned against Elk element-wise.  Exporting ``ylmg`` alone would be ~38 MB of
text to avoid reusing code that is already checked.

**Not differentiated, deliberately.**  Poisson is *linear* in the density, so its
linearisation is itself and an AD-versus-FD check is very nearly vacuous.  The
content here is forward exactness against Elk, and the one check that owes Elk
nothing is :func:`intra_sphere_residual` -- the radial Poisson equation itself,
:math:`\nabla^2V_{lm}=-4\pi\rho_{lm}`, evaluated on the solution.
"""

import numpy as np

import jax
import jax.numpy as jnp

from . import lapw

__all__ = [
    "real_to_complex", "complex_to_real", "cumulative_spline",
    "intra_sphere", "add_nuclear", "multipoles", "pseudocharge_solve",
    "coulomb_potential", "intra_sphere_residual", "dense", "pack",
]

Y00 = 0.28209479177387814347
FOURPI = 4.0 * np.pi
EPSLAT = 1.0e-6


# ------------------------------------------------------------------ packing


def dense(packed, groundstate, ias):
    """Elk's packed muffin-tin layout -> a dense ``(nr, lmmaxo)`` array.

    The inner region's missing harmonics are zero, which is what makes the
    ``l <= lmaxi`` branches below fall out of the arithmetic instead of needing
    a guard -- except in :func:`intra_sphere`, where they genuinely cannot,
    because the radial integral for :math:`l>l_{\\max}^{\\rm i}` starts at
    :math:`r_{\\rm iro}` rather than at the origin.
    """
    isp = int(groundstate["idxis"][ias]) - 1
    nr, nri = int(groundstate["nrmt"][isp]), int(groundstate["nrmti"][isp])
    lmmaxi, lmmaxo = int(groundstate["lmmaxi"]), int(groundstate["lmmaxo"])
    packed = jnp.asarray(packed)
    inner = packed[:lmmaxi * nri].reshape(nri, lmmaxi)
    outer = packed[lmmaxi * nri:lmmaxi * nri + lmmaxo * (nr - nri)]
    top = jnp.zeros((nri, lmmaxo), dtype=packed.dtype).at[:, :lmmaxi].set(inner)
    return jnp.concatenate([top, outer.reshape(nr - nri, lmmaxo)], axis=0)


def pack(values, groundstate, ias):
    """The inverse of :func:`dense`."""
    isp = int(groundstate["idxis"][ias]) - 1
    nri = int(groundstate["nrmti"][isp])
    lmmaxi = int(groundstate["lmmaxi"])
    return jnp.concatenate([values[:nri, :lmmaxi].reshape(-1),
                            values[nri:].reshape(-1)])


# ------------------------------------------------- real <-> complex harmonics


def _rtoz_indices(lmax):
    """``rtozflm``'s coefficients, as flat arrays over ``lm``.

    For each ``lm`` the complex coefficient is
    ``a * rflm[lm] + b * rflm[partner]``, where the partner is ``(l,-m)``.  Elk
    writes this as four sign cases; expressing it as two complex weights keeps
    the transcription checkable against ``rtozflm.f90`` line by line and makes
    the inverse a matrix transpose rather than a second case analysis.
    """
    c1 = 0.7071067811865475244
    n = (lmax + 1) ** 2
    a = np.zeros(n, dtype=complex)
    b = np.zeros(n, dtype=complex)
    partner = np.zeros(n, dtype=int)
    for l in range(lmax + 1):
        base = l * l
        for m in range(-l, l + 1):
            lm = base + l + m
            partner[lm] = base + l - m
            if m < 0:
                sign = -1.0 if m % 2 else 1.0
                a[lm] = -1j * c1
                b[lm] = sign * c1
            elif m == 0:
                a[lm] = 1.0
                b[lm] = 0.0
            else:
                sign = -1j if m % 2 else 1j
                a[lm] = c1
                b[lm] = sign * c1
    return a, b, partner


def real_to_complex(values, lmax):
    """``rtozfmt``: real spherical-harmonic coefficients -> complex ones.

    ``values`` is ``(nr, lmmax)`` real; the transform is per radial point and
    mixes only :math:`(l,m)` with :math:`(l,-m)`.
    """
    a, b, partner = _rtoz_indices(lmax)
    values = jnp.asarray(values)
    n = (lmax + 1) ** 2
    return (jnp.asarray(a)[None, :] * values[:, :n]
            + jnp.asarray(b)[None, :] * values[:, partner])


def complex_to_real(values, lmax):
    """``ztorfmt``, as the inverse of :func:`real_to_complex`.

    Built by inverting the same 2x2 blocks rather than transcribing
    ``ztorflm.f90`` separately: two independent transcriptions of one linear
    map is exactly where a sign convention drifts, and a round-trip test on the
    pair would then pass while both were wrong in the same way.
    """
    a, b, partner = _rtoz_indices(lmax)
    n = (lmax + 1) ** 2
    matrix = np.zeros((n, n), dtype=complex)
    matrix[np.arange(n), np.arange(n)] += a
    matrix[np.arange(n), partner] += b
    inverse = np.linalg.inv(matrix)
    return jnp.real(jnp.asarray(values)[:, :n] @ jnp.asarray(inverse).T)


# --------------------------------------------------------- radial quadrature


def cumulative_spline(weights, f):
    r"""``zpotclmt``'s inner ``splintwp``: :math:`g(r_i)=\int_{r_1}^{r_i}f`.

    ``weights`` is ``wprmt[:, :]`` of shape ``(n, 4)``, four per radial point,
    and each contributes a four-point stencil.  The accumulation is a plain
    prefix sum of terms that depend only on ``f``, so it is a ``cumsum`` rather
    than a loop -- the sequential-looking Fortran carries no state the terms do
    not already contain.
    """
    f = jnp.asarray(f)
    w = jnp.asarray(weights)
    n = f.shape[0]
    first = jnp.dot(w[1], f[:4])
    index = jnp.arange(2, n - 1)
    windows = f[index[:, None] + jnp.arange(-2, 2)[None, :]]
    partial = first + jnp.cumsum(jnp.sum(w[2:n - 1] * windows, axis=1))
    last = partial[-1] + jnp.dot(w[n - 1], f[n - 4:])
    return jnp.concatenate([jnp.zeros(1, dtype=partial.dtype),
                            jnp.array([first], dtype=partial.dtype),
                            partial, jnp.array([last], dtype=partial.dtype)])


# ---------------------------------------------------------- the intra-sphere


def _solve_channel(r, weights, rho, l):
    """One :math:`(l,m)` channel of ``zpotclmt``, on whatever mesh is given."""
    t0 = FOURPI / (2 * l + 1)
    inner = cumulative_spline(weights, r ** (l + 2) * rho)
    outer = cumulative_spline(weights, r ** (1 - l) * rho)
    return t0 * (r ** (-l - 1) * inner + r ** l * (outer[-1] - outer))


def intra_sphere(rho, groundstate, ias):
    r"""``zpotclmt``: the exact radial Poisson solution inside one sphere.

    ``rho`` is dense complex ``(nr, lmmaxo)``.  The two regions are handled
    separately and that is **not** an optimisation: for :math:`l>l_{\max}^{\rm
    i}` the inner region does not store the harmonic at all, so Elk integrates
    from :math:`r_{\rm iro}` outward using the sub-mesh's own spline weights.
    Zero-padding and integrating from the origin would use the wrong weights at
    the lower boundary and give a plausible, slightly wrong potential.
    """
    isp = int(groundstate["idxis"][ias]) - 1
    nr, nri = int(groundstate["nrmt"][isp]), int(groundstate["nrmti"][isp])
    lmaxi, lmaxo = int(groundstate["lmaxi"]), int(groundstate["lmaxo"])
    r = jnp.asarray(groundstate["rlmt"][isp][:nr])
    weights = jnp.asarray(groundstate["wprmt"][isp].T[:nr])

    columns = []
    for l in range(lmaxo + 1):
        for _ in range(2 * l + 1):
            lm = len(columns)
            if l <= lmaxi:
                columns.append(_solve_channel(r, weights, rho[:, lm], l))
            else:
                tail = _solve_channel(r[nri:], weights[nri:], rho[nri:, lm], l)
                columns.append(jnp.concatenate(
                    [jnp.zeros(nri, dtype=tail.dtype), tail]))
    return jnp.stack(columns, axis=1)


def intra_sphere_residual(potential, rho, groundstate, ias, l, lm,
                          margin=6):
    r""":math:`\nabla^2V_{lm}+4\pi\rho_{lm}` on the radial mesh -- a forward
    check owing Elk nothing.

    For a single spherical-harmonic channel the Laplacian is
    :math:`V''+\tfrac2rV'-\tfrac{l(l+1)}{r^2}V`, so the defining equation can be
    tested directly on the solution rather than against a reference.  The
    derivatives are taken by fitting a local polynomial through Elk's own
    (logarithmic, non-uniform) mesh, so ``margin`` points at each end are
    dropped.

    Returned relative to :math:`4\pi|\rho_{lm}|`, so it is a residual and not a
    scale.
    """
    isp = int(groundstate["idxis"][ias]) - 1
    nr = int(groundstate["nrmt"][isp])
    r = np.asarray(groundstate["rlmt"][isp][:nr])
    v = np.asarray(potential[:, lm])
    rhs = -FOURPI * np.asarray(rho[:, lm])

    residual = np.zeros(nr - 2 * margin, dtype=complex)
    for i, ir in enumerate(range(margin, nr - margin)):
        window = slice(ir - 2, ir + 3)
        fit = np.polyfit(r[window] - r[ir], v[window], 4)
        second, first = 2.0 * fit[-3], fit[-2]
        lap = second + 2.0 / r[ir] * first - l * (l + 1) / r[ir] ** 2 * v[ir]
        residual[i] = lap - rhs[ir]
    scale = np.abs(rhs[margin:nr - margin])
    return residual, scale


def add_nuclear(potential, groundstate, ias):
    r"""``potcoul``'s nuclear term: ``vcln`` into the :math:`l=0` slot.

    ``vcln`` is the :math:`(0,0)` **coefficient**, i.e. :math:`\sqrt{4\pi}` times
    the potential itself -- measured, its outermost value is
    :math:`-Z/(R\,y_{00})`.  This runs BEFORE the multipoles are read, which is
    why :math:`q_{00}` comes out proportional to :math:`N_{\rm MT}-Z` rather
    than to the electron count alone.
    """
    isp = int(groundstate["idxis"][ias]) - 1
    nr = int(groundstate["nrmt"][isp])
    vcln = jnp.asarray(groundstate["vcln"][isp][:nr])
    return potential.at[:, 0].add(vcln)


# ---------------------------------------------------------- the interstitial


def _double_factorial(n):
    out = 1.0
    while n > 1:
        out *= n
        n -= 2
    return out


def _reciprocal(groundstate):
    """``ylmg``, ``sfacg``, ``jlgrmt`` and ``gclg``, rebuilt rather than exported."""
    lmaxo = int(groundstate["lmaxo"])
    lnpsd = int(groundstate["lnpsd"])
    vgc = jnp.asarray(groundstate["vgc"]).T
    gc = jnp.asarray(groundstate["gc"])
    atposc = jnp.asarray(groundstate["atposc"])
    ylmg = jax.vmap(lambda v: lapw.spherical_harmonics(lmaxo, v))(vgc)
    sfacg = lapw.structure_factor(vgc, atposc)
    nspecies = int(groundstate["nspecies"])
    bessel = jax.vmap(lambda x: lapw.spherical_bessel(lnpsd, x))
    jl = []
    for isp in range(nspecies):
        nr = int(groundstate["nrmt"][isp])
        rmt = float(groundstate["rlmt"][isp][nr - 1])
        jl.append(bessel(gc * rmt))
    gclg = jnp.where(gc > EPSLAT, FOURPI / jnp.where(gc > EPSLAT, gc, 1.0) ** 2,
                     0.0)
    return ylmg, sfacg, jl, gclg


def multipoles(potentials, groundstate):
    r""":math:`q_{lm}=\frac{2l+1}{4\pi}R^{l+1}V_{lm}(R)`, from ``zpotcoul``.

    Read off the sphere-BOUNDARY value of the intra-sphere potential rather
    than by integrating :math:`\rho r^l` -- which matters, because that
    potential already carries the nucleus, so these are the moments of the
    total charge.
    """
    lmaxo = int(groundstate["lmaxo"])
    out = []
    for ias, potential in enumerate(potentials):
        isp = int(groundstate["idxis"][ias]) - 1
        nr = int(groundstate["nrmt"][isp])
        rmt = float(groundstate["rlmt"][isp][nr - 1])
        factor = jnp.concatenate([
            jnp.full(2 * l + 1, (2 * l + 1) / FOURPI * rmt ** (l + 1))
            for l in range(lmaxo + 1)])
        out.append(factor * potential[nr - 1])
    return jnp.stack(out)


def _forward_fft(values, ngridg):
    return jnp.fft.fftn(jnp.asarray(values).reshape(ngridg, order="F")
                        ).reshape(-1, order="F") / np.prod(ngridg)


def _inverse_fft(values, ngridg):
    return jnp.fft.ifftn(jnp.asarray(values).reshape(ngridg, order="F")
                         ).reshape(-1, order="F") * np.prod(ngridg)


def pseudocharge_solve(potentials, rhoir, groundstate):
    r"""``zpotcoul``: the interstitial solve and the boundary matching.

    Returns ``(potentials, vclir)``, both complex.  The pseudocharge is the
    smooth density with the same multipoles,
    :math:`\propto(1-r^2/R^2)^{N_{\rm psd}}r^lY_{lm}`, whose Fourier transform
    is :math:`j_{l+N+1}(GR)/(GR)^{N+1}` up to normalisation -- which is where
    ``lnpsd`` and the double factorials come from.
    """
    natmtot = int(groundstate["natmtot"])
    lmaxi, lmaxo = int(groundstate["lmaxi"]), int(groundstate["lmaxo"])
    lmmaxo = int(groundstate["lmmaxo"])
    ngvec = int(groundstate["ngvec"])
    lnpsd = int(groundstate["lnpsd"])
    omega = float(groundstate["omega"])
    ngridg = tuple(int(n) for n in groundstate["ngridg"])
    igfft = np.asarray(groundstate["igfft"])[:ngvec] - 1
    gc = jnp.asarray(groundstate["gc"])
    ylmg, sfacg, jl, gclg = _reciprocal(groundstate)
    big = gc > EPSLAT

    zvclir = _forward_fft(rhoir, ngridg)
    qlm = multipoles(potentials, groundstate)

    lslot = np.concatenate([np.full(2 * l + 1, l) for l in range(lmaxo + 1)])
    lslot = jnp.asarray(lslot)

    # what the interstitial density already contributes at the sphere boundary
    for ias in range(natmtot):
        isp = int(groundstate["idxis"][ias]) - 1
        nr = int(groundstate["nrmt"][isp])
        rmt = float(groundstate["rlmt"][isp][nr - 1])
        z1 = jnp.where(big, zvclir[igfft] * sfacg[:, ias]
                       / jnp.where(big, gc, 1.0), 0.0)
        weight = jl[isp][:, lslot + 1] * rmt ** (lslot + 2)
        # Elk writes the l=0 slot as `jlgprmt(1,ig,is)*R^2*fourpi*y00*z1`, with
        # no conj(Y_lm) factor -- but `genylmv(.true.,...)` carries a 4*pi*(-i)^l
        # prefactor, so ylmg[:,0] IS 4*pi*y00, a real constant, and the uniform
        # expression below reproduces that slot exactly.  Same in both blocks
        # further down.  Special-casing it would be a second transcription of
        # one line.
        zlm = jnp.sum(weight * z1[:, None] * jnp.conj(ylmg), axis=0)
        head = (FOURPI / 3.0) * rmt ** 3 * Y00
        zlm = zlm.at[0].add(jnp.sum(jnp.where(big, 0.0, 1.0)
                                    * head * zvclir[igfft]))
        qlm = qlm.at[ias].add(-zlm)

    # the pseudocharge, added in G-space
    t1 = _double_factorial(2 * lnpsd + 1) / omega
    denominator = jnp.asarray(
        [_double_factorial(2 * int(l) + 1) for l in np.asarray(lslot)])
    for ias in range(natmtot):
        isp = int(groundstate["idxis"][ias]) - 1
        nr = int(groundstate["nrmt"][isp])
        rmt = float(groundstate["rlmt"][isp][nr - 1])
        zlm = t1 / (denominator * rmt ** lslot) * qlm[ias]
        t2 = gc * rmt
        safe = jnp.where(big, t2, 1.0)
        power = safe[:, None] ** (lslot[None, :] - lnpsd)
        z1 = jnp.sum(power * (zlm * ylmg), axis=1)
        z2 = jl[isp][:, lnpsd] * jnp.conj(sfacg[:, ias])
        head = FOURPI * Y00 / _double_factorial(2 * lnpsd + 1) * zlm[0]
        contribution = jnp.where(big, z1 * z2, head)
        zvclir = zvclir.at[igfft].add(contribution)

    # Only the first `ngvec` slots are divided by G^2; Elk leaves the rest of the
    # FFT array holding the raw density transform, and this reproduces that
    # rather than zeroing them, which would be a correction and not a
    # transcription.
    zvclir = zvclir.at[igfft].multiply(gclg)

    # the harmonic function that matches the boundary value
    out = []
    for ias in range(natmtot):
        isp = int(groundstate["idxis"][ias]) - 1
        nr, nri = int(groundstate["nrmt"][isp]), int(groundstate["nrmti"][isp])
        rmt = float(groundstate["rlmt"][isp][nr - 1])
        r = jnp.asarray(groundstate["rlmt"][isp][:nr])
        z1 = zvclir[igfft] * sfacg[:, ias]
        weight = jl[isp][:, lslot]
        zlm = jnp.sum(weight * z1[:, None] * jnp.conj(ylmg), axis=0)
        amplitude = (zlm - potentials[ias][nr - 1]) / rmt ** lslot
        harmonic = amplitude[None, :] * r[:, None] ** lslot[None, :]
        mask = jnp.asarray(
            (np.arange(nr)[:, None] >= nri)
            | (np.arange(lmmaxo)[None, :] < (lmaxi + 1) ** 2))
        out.append(potentials[ias] + jnp.where(mask, harmonic, 0.0))
    return jnp.stack(out), _inverse_fft(zvclir, ngridg)


def coulomb_potential(groundstate):
    """``potcoul``: Elk's own ``vclmt`` and ``vclir``, from the density alone."""
    lmaxo = int(groundstate["lmaxo"])
    natmtot = int(groundstate["natmtot"])
    potentials = []
    for ias in range(natmtot):
        rho = real_to_complex(dense(groundstate["rhomt"][ias], groundstate, ias),
                              lmaxo)
        potentials.append(
            add_nuclear(intra_sphere(rho, groundstate, ias), groundstate, ias))
    potentials, vclir = pseudocharge_solve(
        jnp.stack(potentials), groundstate["rhoir"], groundstate)
    vclmt = jnp.stack([complex_to_real(potentials[ias], lmaxo)
                       for ias in range(natmtot)])
    return vclmt, jnp.real(vclir)
