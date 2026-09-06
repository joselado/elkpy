r"""Phase 1 of the JAX port: the APW and local-orbital radial functions.

`radial.py` builds the muffin-tin radial integrals from the radial functions;
this builds the radial functions themselves from the potential, which is the
remaining link in

    vsmt  ->  apwfr / lofr  ->  radial integrals  ->  H, O  ->  evalfv

Elk's routines are `rschrodint.f90` (the scalar-relativistic radial
Schrodinger equation), `genapwfr.f90` and `genlofr.f90`.

**The equation.**  Following Koelling and Harmon, *J. Phys. C* **10**, 3107
(1977), the scalar-relativistic radial problem is written as a first-order
pair in :math:`P_\ell=rg_\ell` and :math:`Q_\ell=(r/2M)\,dg_\ell/dr`, with
:math:`g_\ell` the major component of the Dirac equation and
:math:`M=1+(E-V)/2c^2` the relativistic mass enhancement:

.. math::

   \frac{dP_\ell}{dr} = 2MQ_\ell + \frac{P_\ell}{r}, \qquad
   \frac{dQ_\ell}{dr} = -\frac{Q_\ell}{r}
     + \left[\frac{\ell(\ell+1)}{2Mr^2} + V - E\right] P_\ell .

Spin-orbit coupling is absent by construction -- it is the term dropped in
going from Dirac to *scalar*-relativistic -- which is why `soc_scale` cannot
move the first-variational spectrum at all (`docs/jax_port_phase1.md` §1f).

**The integration is not a generic ODE solve and is not replaced by one.**
Elk marches outward on its own logarithmic mesh with a predictor-corrector:
a 3-point extrapolation of the derivative, then eight fixed-point corrections
in which the function is recovered by integrating the cubic through the last
four derivative samples.  The radial functions are compared against Elk's
element-wise, so the scheme is transcribed exactly rather than improved --
`jax.experimental.ode` or a Runge-Kutta step would converge to the same
continuum solution and disagree with Elk at the mesh's own truncation error,
which is far above the roundoff this is checked at.

Two pieces of Elk's routine are deliberately dropped, both non-differentiable
branches: the node count `nn` (an integer diagnostic `linengy` uses, and this
port holds the linearisation energies fixed) and the overflow freeze at
:math:`|P|>10^{100}` (which exists for deep core states in `rdirac`; at the
valence linearisation energies the APW solutions do not approach it).

**The radial functions reach `apwalm` through `D`.**  `match` solves
:math:`DA=b` with :math:`D` the matrix of radial derivatives at
:math:`R_{\rm MT}`, so rebuilding `apwfr` without rebuilding :math:`D` freezes
the basis at the sphere boundary while changing it inside -- and every APW
block of both :math:`H` and :math:`O` then misses the matching response.
`derivative_matrices` closes that, and it is checked against the exported
`dmat` rather than against a finite difference, which would differentiate the
same truncated function twice and see nothing.

**What `apwfr`'s second component is.**  `genapwfr` stores :math:`u` and
:math:`Eu`, then Gram-Schmidt-orthonormalises the set at each :math:`\ell`
carrying BOTH through the same combination -- so the second component is a
linear combination of :math:`E_{i}u_{i}` and is NOT :math:`E` times the first,
except at APW order 1.  `apwdfr` (the surface derivative
:math:`(dg/dr)R_{\rm MT}^2/2`) rides through the same combination.  Nothing
below order 2 can see either, which is why the `apword=2` fixture exists.
"""

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax


# ---------------------------------------------------------------------------
# The three polynomial helpers, transcribed from rschrodint.f90 and polynm.f90
# ---------------------------------------------------------------------------

def poly3(xa, ya, x):
    """Value at `x` of the quadratic through three points -- `rschrodint`'s
    own contained function, transcribed operation for operation."""
    x0 = xa[0]
    x1, x2 = xa[1] - x0, xa[2] - x0
    y0 = ya[0]
    y1, y2 = ya[1] - y0, ya[2] - y0
    t0 = 1.0 / (x1 * x2 * (x2 - x1))
    t1, t2 = x1 * y2, x2 * y1
    c1 = x2 * t2 - x1 * t1
    c2 = t1 - t2
    t1 = x - x0
    return y0 + t0 * t1 * (c1 + c2 * t1)


def poly4i(xa, ya, x):
    """Integral from `xa[0]` to `x` of the cubic through four points."""
    x0 = xa[0]
    x1, x2, x3 = xa[1] - x0, xa[2] - x0, xa[3] - x0
    y0 = ya[0]
    y1, y2, y3 = ya[1] - y0, ya[2] - y0, ya[3] - y0
    t4, t5, t6 = x1 - x2, x1 - x3, x2 - x3
    t1, t2, t3 = x1 * x2 * y3, x2 * x3 * y1, x1 * x3
    t0 = 1.0 / (x2 * t3 * t4 * t5 * t6)
    t3 = t3 * y2
    c3 = t1 * t4 + t2 * t6 - t3 * t5
    t4, t5, t6 = x1 ** 2, x2 ** 2, x3 ** 2
    c2 = t1 * (t5 - t4) + t2 * (t6 - t5) + t3 * (t4 - t6)
    c1 = (t1 * (x2 * t4 - x1 * t5) + t2 * (x3 * t5 - x2 * t6)
          + t3 * (x1 * t6 - x3 * t4))
    t1 = x - x0
    return t1 * (y0 + t0 * t1 * (0.5 * c1 + t1 * (c2 / 3.0 + 0.25 * c3 * t1)))


def polynm(m, xa, ya, x):
    """`polynm.f90`: the `m`-th derivative at `x` of the polynomial through
    all of `xa`/`ya`.  Only `m > 0` is used here (`genlofr`'s matching
    conditions), and `m == 0` is included because it is the cheapest check
    that the divided-difference bookkeeping is right."""
    np_ = len(xa)
    if np_ < 1 or m >= np_:
        return jnp.asarray(0.0)
    c = [ya[i] for i in range(np_)]
    for i in range(2, np_ + 1):
        for j in range(np_, i - 1, -1):
            c[j - 1] = (c[j - 1] - c[j - 2]) / (xa[j - 1] - xa[j - i])
    if m == 0:
        out = c[0]
        t1 = jnp.asarray(1.0)
        for i in range(2, np_ + 1):
            t1 = t1 * (x - xa[i - 2])
            out = out + c[i - 1] * t1
        return out
    x0 = xa[0]
    for j in range(1, np_):
        for i in range(1, np_ - j + 1):
            k = np_ - i
            c[k - 1] = c[k - 1] + (x0 - xa[k - j]) * c[k]
    for j in range(1, m + 1):
        for i in range(m + 1, np_ + 1):
            c[i - 1] = c[i - 1] * float(i - j)
    out = c[np_ - 1]
    t1 = x - x0
    for i in range(np_ - 1, m, -1):
        out = out * t1 + c[i - 1]
    return out


# ---------------------------------------------------------------------------
# rschrodint
# ---------------------------------------------------------------------------

_NCORR = 8          # rschrodint's fixed corrector count, `do i=1,8`


def rschrodint(sol, l, e, r, vr):
    r"""Integrate the scalar-relativistic radial Schrodinger equation outward.

    Returns ``(p0, p1, q0, q1)`` -- :math:`P_\ell`, :math:`dP_\ell/dr`,
    :math:`Q_\ell` and :math:`dQ_\ell/dr` on the mesh `r`, with `vr` the
    spherical part of the potential there.

    `l` must be a Python int (it enters as the centrifugal coefficient and as
    nothing else); `e`, `r` and `vr` are traced.  The first three mesh points
    are special in Elk -- their stencil windows overlap the
    :math:`r\to0` boundary values and the initial derivative extrapolation --
    so they are unrolled here and the `lax.scan` starts at the fourth, which
    is the first point whose window is entirely behind it.
    """
    nr = r.shape[0]
    t1 = 1.0 / sol ** 2
    t2 = float(l * (l + 1))
    ri = 1.0 / r
    t3 = 2.0 + t1 * (e - vr)
    t4 = (t2 * ri ** 2) / t3 + vr - e
    # r -> 0 boundary values
    q0_0 = jnp.asarray(1.0, dtype=r.dtype)
    p0_0 = ri[0] / t4[0]
    p1_0 = t3[0] + p0_0 * ri[0]
    # Elk extrapolates p1 flat over points 2..4 and zeroes q1 over 1..4; the
    # first two steps below read those slots, so they are literal here.
    p1a = [p1_0, p1_0, p1_0, p1_0]
    q1a = [jnp.zeros((), r.dtype)] * 4
    p0a, q0a = [p0_0], [q0_0]

    def _correct(j, ir0, p1win, q1win, p0base, q0base):
        """Eight corrections at mesh point `j`, with `ir0` the window start."""
        rw4 = r[ir0:ir0 + 4]
        p1j = poly3(r[ir0:ir0 + 3], p1win[:3], r[j])
        q1j = poly3(r[ir0:ir0 + 3], q1win[:3], r[j])
        slot = j - ir0                       # which window slot point j is
        for _ in range(_NCORR):
            pw = list(p1win)
            qw = list(q1win)
            pw[slot], qw[slot] = p1j, q1j
            p0j = poly4i(rw4, pw, r[j]) + p0base
            q0j = poly4i(rw4, qw, r[j]) + q0base
            p1j = t3[j] * q0j + p0j * ri[j]
            q1j = t4[j] * p0j - q0j * ri[j]
        return p0j, p1j, q0j, q1j

    for j in (1, 2):
        p0j, p1j, q0j, q1j = _correct(j, 0, p1a, q1a, p0a[0], q0a[0])
        p1a[j], q1a[j] = p1j, q1j
        p0a.append(p0j)
        q0a.append(q0j)

    def body(carry, j):
        rw, p1w, q1w, p0w, q0w = carry
        rw4 = jnp.concatenate([rw, r[j][None]])
        p1j = poly3(rw, p1w, r[j])
        q1j = poly3(rw, q1w, r[j])
        for _ in range(_NCORR):
            p0j = poly4i(rw4, jnp.concatenate([p1w, p1j[None]]), r[j]) + p0w[0]
            q0j = poly4i(rw4, jnp.concatenate([q1w, q1j[None]]), r[j]) + q0w[0]
            p1j = t3[j] * q0j + p0j * ri[j]
            q1j = t4[j] * p0j - q0j * ri[j]
        shift = lambda w, new: jnp.concatenate([w[1:], new[None]])
        carry = (shift(rw, r[j]), shift(p1w, p1j), shift(q1w, q1j),
                 shift(p0w, p0j), shift(q0w, q0j))
        return carry, jnp.stack([p0j, p1j, q0j, q1j])

    init = (r[0:3],
            jnp.stack(p1a[0:3]), jnp.stack(q1a[0:3]),
            jnp.stack(p0a[0:3]), jnp.stack(q0a[0:3]))
    _, tail = lax.scan(body, init, jnp.arange(3, nr))
    head = jnp.stack([jnp.stack(p0a), jnp.stack(p1a[:3]),
                      jnp.stack(q0a), jnp.stack(q1a[:3])], axis=1)
    full = jnp.concatenate([head, tail], axis=0)
    return full[:, 0], full[:, 1], full[:, 2], full[:, 3]


# ---------------------------------------------------------------------------
# genapwfr / genlofr
# ---------------------------------------------------------------------------

def spherical_potential(export, potential=None):
    """The spherical part of the muffin-tin potential, per atom, as the
    radial function :math:`V(r)` the ODE takes -- i.e. the :math:`\\ell=0`
    coefficient times :math:`Y_{00}`."""
    from . import radial as _radial
    potential = (_radial.potential_arrays(export) if potential is None
                 else potential)
    y00 = float(export["y00"])
    return [jnp.asarray(v)[:, 0] * y00 for v in potential]


def _atom_mesh(export, ias):
    idxis = np.asarray(export["idxis"]) - 1
    is_ = int(idxis[ias])
    nr = int(np.asarray(export["nrmt"])[is_])
    r = jnp.asarray(np.asarray(export["rlmt"])[is_, :nr])
    w = jnp.asarray(np.asarray(export["wr2mt"])[is_, :nr])
    return is_, nr, r, w


def apw_radial_functions(export, potential=None, apwe=None):
    """`genapwfr`: returns `(apwfr, apwdfr)` in Elk's own index order,
    `apwfr[ir, {u, Hu}, io, l, ias]` and `apwdfr[io, l, ias]`.

    `potential` (the list of dense muffin-tin arrays) and `apwe` (the
    linearisation energies) are the differentiable inputs.
    """
    vr_all = spherical_potential(export, potential)
    apwe = export["apwe"] if apwe is None else apwe
    apword = np.asarray(export["apword"])
    apwdm = np.asarray(export["apwdm"])
    rmt = np.asarray(export["rmt"])
    deapw = float(export["deapw"])
    sol = float(export["solsc"])
    natmtot = int(export["natmtot"])
    nl = int(export["lmaxapw"]) + 1
    apwordmax = int(export["apwordmax"])
    nrmtmax = int(export["nrmtmax"])
    fr = jnp.zeros((nrmtmax, 2, apwordmax, nl, natmtot))
    dfr = jnp.zeros((apwordmax, nl, natmtot))
    for ias in range(natmtot):
        is_, nr, r, w = _atom_mesh(export, ias)
        vr = vr_all[ias][:nr]
        for l in range(nl):
            p0s, ep0s, p1ss = [], [], []
            for io in range(int(apword[l, is_])):
                e = apwe[io, l, ias] + apwdm[io, l, is_] * deapw
                p0, p1, _, _ = rschrodint(sol, l, e, r, vr)
                p0 = p0 / r                     # P = r g, and u is g
                ep0 = e * p0
                t1 = 1.0 / jnp.sqrt(jnp.abs(jnp.sum(w * p0 ** 2)))
                p0, ep0, p1s = t1 * p0, t1 * ep0, t1 * p1[nr - 1]
                for jo in range(io):            # Gram-Schmidt, in place
                    t1 = -jnp.sum(w * p0 * p0s[jo])
                    p0 = p0 + t1 * p0s[jo]
                    p1s = p1s + t1 * p1ss[jo]
                    ep0 = ep0 + t1 * ep0s[jo]
                if io > 0:
                    t1 = 1.0 / jnp.sqrt(jnp.sum(w * p0 ** 2))
                    p0, ep0, p1s = t1 * p0, t1 * ep0, t1 * p1s
                p0s.append(p0)
                ep0s.append(ep0)
                p1ss.append(p1s)
                fr = fr.at[:nr, 0, io, l, ias].set(p0)
                fr = fr.at[:nr, 1, io, l, ias].set(ep0)
                dfr = dfr.at[io, l, ias].set(
                    (p1s - p0[nr - 1]) * rmt[is_] / 2.0)
    return fr, dfr


def lo_radial_functions(export, potential=None, lorbe=None):
    """`genlofr`: returns `lofr[ir, {u, Hu}, ilo, ias]`.

    A local orbital is the combination of `lorbord` solutions at different
    energies whose value and first `lorbord - 1` radial derivatives vanish at
    :math:`R_{\\rm MT}` -- the linear system `genlofr` sets up with `polynm`
    and solves.  It is then orthonormalised against the previously processed
    local orbitals *of the same* :math:`\\ell` only, in Elk's own ascending
    energy order (`idxelo`), which is why that ordering is exported.
    """
    vr_all = spherical_potential(export, potential)
    lorbe = export["lorbe"] if lorbe is None else lorbe
    nlorb = np.asarray(export["nlorb"])
    lorbl = export["lorbl"]
    lorbord = np.asarray(export["lorbord"])
    lorbdm = np.asarray(export["lorbdm"])
    idxelo = np.asarray(export["idxelo"])
    rmt = np.asarray(export["rmt"])
    delorb = float(export["delorb"])
    sol = float(export["solsc"])
    nplorb = int(export["nplorb"])
    natmtot = int(export["natmtot"])
    nlomax = int(export["nlomax"])
    nrmtmax = int(export["nrmtmax"])
    out = jnp.zeros((nrmtmax, 2, nlomax, natmtot))
    for ias in range(natmtot):
        is_, nr, r, w = _atom_mesh(export, ias)
        vr = vr_all[ias][:nr]
        done = {}                                # ilo -> (u, Hu), for the
        for i in range(int(nlorb[is_])):         # Gram-Schmidt below
            ilo = int(idxelo[i, is_]) - 1
            l = int(lorbl[is_][ilo])
            ord_ = int(lorbord[ilo, is_])
            p0s, ep0s, rows = [], [], []
            for jo in range(ord_):
                e = lorbe[jo, ilo, ias] + lorbdm[jo, ilo, is_] * delorb
                p0, _, _, _ = rschrodint(sol, l, e, r, vr)
                p0 = p0 / r
                p0s.append(p0)
                ep0s.append(e * p0)
                tail = slice(nr - nplorb, nr)
                col = [p0[nr - 1]]
                for io in range(1, ord_):
                    col.append(polynm(io, r[tail], p0[tail], rmt[is_]))
                rows.append(jnp.stack(col))
            amat = jnp.stack(rows, axis=1)       # a[io, jo]
            b = jnp.zeros(ord_).at[ord_ - 1].set(1.0)
            coeff = jnp.linalg.solve(amat, b)
            u = sum(coeff[io] * p0s[io] for io in range(ord_))
            hu = sum(coeff[io] * ep0s[io] for io in range(ord_))
            t1 = 1.0 / jnp.sqrt(jnp.abs(jnp.sum(w * u ** 2)))
            u, hu = t1 * u, t1 * hu
            for j in range(i):
                jlo = int(idxelo[j, is_]) - 1
                if int(lorbl[is_][jlo]) == l:
                    uj, huj = done[jlo]
                    t1 = -jnp.sum(w * u * uj)
                    u, hu = u + t1 * uj, hu + t1 * huj
            if i > 0:
                t1 = 1.0 / jnp.sqrt(jnp.sum(w * u ** 2))
                u, hu = t1 * u, t1 * hu
            done[ilo] = (u, hu)
            out = out.at[:nr, 0, ilo, ias].set(u)
            out = out.at[:nr, 1, ilo, ias].set(hu)
    return out


def derivative_matrices(export, apwfr=None):
    r"""The matrices $D$ that `match` inverts, rebuilt from the radial
    functions.

    $D^{\alpha\ell}_{ij}$ is the $(i-1)$-th radial derivative of the $j$-th APW
    radial function at $R_{\rm MT}$, with the zeroth row the value itself:

    .. math::
       D_{1j} = u_{j\ell}(R_{\rm MT}), \qquad
       D_{ij} = \left.\frac{d^{\,i-1}u_{j\ell}}{dr^{\,i-1}}\right|_{R_{\rm MT}},
       \quad i>1,

    the derivatives taken by fitting a polynomial through the last `npapw`
    mesh points (`polynm`).  `match` solves $DA=b$ for the matching
    coefficients, so **$D$ is how the radial functions reach `apwalm`**: a
    caller that rebuilds `apwfr` from a perturbed potential and leaves $D$
    alone silently freezes the basis's shape at the sphere boundary while
    changing it inside, and every APW block of both $H$ and $O$ then misses
    the matching response.  That is not visible in an AD-versus-FD comparison,
    which differentiates the same truncated function twice; the check with
    teeth is this array against the exported `dmat`.

    Returns the same nested list shape the export uses: over atoms, then over
    $\ell$, an `(ord, ord)` array with `ord = apword[l, is]`.
    """
    apwfr = export["apwfr_full"] if apwfr is None else apwfr
    apword = np.asarray(export["apword"])
    rmt = np.asarray(export["rmt"])
    npapw = int(export["npapw"])
    nl = int(export["lmaxapw"]) + 1
    out = []
    for ias in range(int(export["natmtot"])):
        is_, nr, r, _ = _atom_mesh(export, ias)
        tail = slice(nr - npapw, nr)
        rtail = r[tail]
        per_l = []
        for l in range(nl):
            ord_ = int(apword[l, is_])
            cols = []
            for jo in range(ord_):
                u = apwfr[tail, 0, jo, l, ias]
                col = [u[-1]] + [polynm(io, rtail, u, rmt[is_])
                                 for io in range(1, ord_)]
                cols.append(jnp.stack(col))
            per_l.append(jnp.stack(cols, axis=1))       # d[io, jo]
        out.append(per_l)
    return out


def radial_functions_from_export(export, potential=None, apwe=None,
                                 lorbe=None):
    """A copy of `export` with `apwfr_full`, `apwdfr` and `lofr` REPLACED by
    the ones built here, so `radial.py` and `hamiltonian.py` consume them
    unchanged.  Composed with `radial.integrals_from_export`, this makes the
    whole first-variational spectrum a function of `vsmt`."""
    from . import hamiltonian as _ham

    fr, dfr = apw_radial_functions(export, potential=potential, apwe=apwe)
    lofr = lo_radial_functions(export, potential=potential, lorbe=lorbe)
    dmat = derivative_matrices(export, apwfr=fr)
    out = dict(export)
    out.update(apwfr_full=fr, apwdfr=dfr, lofr=lofr, dmat=dmat)
    # apwalm depends on the radial functions THROUGH dmat, so it has to be
    # rebuilt too -- see `derivative_matrices`.  At the export's own G+k, so
    # the result is directly comparable with Elk's own matrices.
    ngp = int(export["ngp"])
    vgkc = jnp.asarray(np.asarray(export["vgpc"])[:, :ngp].T)
    out["apwalm"] = _ham.matching_coefficients(out, vgkc)
    return out
