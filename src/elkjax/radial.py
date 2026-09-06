"""Phase 1 of the JAX port: the muffin-tin radial integrals.

`hamiltonian.py` assembles the first-variational $H$ and $O$ from the radial
integrals `haa`, `hloa`, `hlolo`, `oalo`, `ololo`, which it imports from the
LAPW export.  This module builds them instead, from the muffin-tin Kohn-Sham
potential and the APW/local-orbital radial functions -- Elk's `hmlrad.f90` and
`olprad.f90`.  Together with `radial_functions.py` (which builds the radial
functions themselves) it closes the chain

    vsmt  ->  apwfr / lofr  ->  radial integrals  ->  H, O  ->  evalfv

so the spectrum becomes a differentiable function of the potential rather than
of a set of imported numbers.

Everything here is pure quadrature: one weighted sum over the radial mesh per
integral.  Three details of it are load-bearing and none is visible in the
formulae as Elk's manual states them.

**The packing.**  A muffin-tin function is stored with `lmmaxi` spherical
harmonics per radial point over the inner region (points 1..`nrmti`) and
`lmmaxo` over the outer one, so every consumer in `vendor/elk/src/` addresses
the potential with a stride.  `hmlrad` then splits each integral into an inner
and an outer sum, and treats $\\ell_2>\\ell_{\\max}^{\\rm i}$ by dropping the
inner one.  Unpacking to a dense `(nr, lmmaxo)` array with the missing inner
harmonics set to ZERO (`parsers.eigenstates.unpack_muffin_tin`) makes both the
split and the guard automatic: the single full-range sum is then correct for
every $\\ell_2$.

**The $\\ell_2=0$ element is not a potential integral at all.**  It is
$\\int u_{q\\ell}\\,\\hat H u_{q'\\ell}\\,r^2dr$, and Elk stores $\\hat H u$ as
the second component of `apwfr`/`lofr`.  For the APW-APW block it is
symmetrised over the two orderings AND carries a kinetic surface term
(`apwdfr`, the derivative at $R_{\\rm MT}$); the two local-orbital blocks carry
neither, because a local orbital vanishes at the sphere boundary.

**`hlolo`'s $\\ell_2=0$ element is asymmetric.**  It is built as
$\\int u^{\\rm lo}_i(\\hat H u^{\\rm lo}_j)r^2dr$ with no averaging, so the two
orderings genuinely differ (measured 1.3e-2 Ha on h-BN's nitrogen) and only the
half Elk evaluates may be used -- see `docs/jax_port_phase1.md`.  It is
transcribed here exactly as written, asymmetry included.
"""

import numpy as np

import jax.numpy as jnp



def potential_arrays(export, packed=None):
    """The muffin-tin Kohn-Sham potential, unpacked, one dense array per atom.

    Returns a list of `(nr, lmmaxo)` arrays indexed `[ir, lm]`, with the
    harmonics the inner region does not carry set to zero.

    `packed` defaults to the exported `vsmt` and is the differentiable input:
    everything downstream -- the radial functions, the radial integrals, $H$
    and the spectrum -- is a function of it.  The unpacking is done with
    `jnp` rather than `parsers.eigenstates.unpack_muffin_tin` so a traced
    array survives it; that function is the NumPy reference for the same
    layout and is what `tests/test_calculation_lapw_radial.py` checks this
    against.
    """
    idxis = np.asarray(export["idxis"]) - 1
    nrmt = np.asarray(export["nrmt"])
    nrmti = np.asarray(export["nrmti"])
    lmmaxi, lmmaxo = int(export["lmmaxi"]), int(export["lmmaxo"])
    packed = export["vsmt"] if packed is None else packed
    out = []
    for ias in range(int(export["natmtot"])):
        is_ = int(idxis[ias])
        nr, nri = int(nrmt[is_]), int(nrmti[is_])
        row = packed[ias]
        inner = jnp.reshape(row[:lmmaxi * nri], (nri, lmmaxi))
        outer = jnp.reshape(
            row[lmmaxi * nri:lmmaxi * nri + lmmaxo * (nr - nri)],
            (nr - nri, lmmaxo))
        dense = jnp.zeros((nr, lmmaxo))
        out.append(dense.at[:nri, :lmmaxi].set(inner).at[nri:, :].set(outer))
    return out


def _atom_shapes(export, ias):
    idxis = np.asarray(export["idxis"]) - 1
    is_ = int(idxis[ias])
    return is_, int(np.asarray(export["nrmt"])[is_])


def overlap_integrals(export, apwfr=None, lofr=None):
    """`olprad`: the APW--local-orbital and local-orbital--local-orbital radial
    overlaps.

    Returns `(oalo, ololo)` in Elk's own index order, `oalo[io, ilo, ias]` and
    `ololo[ilo, jlo, ias]`.  Entries Elk never assigns -- an APW order beyond
    that species' `apword`, or a local-orbital pair with different $\\ell$ --
    are zero here; Elk leaves them uninitialised, so a comparison must mask
    them (`assigned_masks`).
    """
    apwfr = export["apwfr_full"] if apwfr is None else apwfr
    lofr = export["lofr"] if lofr is None else lofr
    apword = np.asarray(export["apword"])
    nlorb = np.asarray(export["nlorb"])
    lorbl = export["lorbl"]
    wr2mt = np.asarray(export["wr2mt"])
    natmtot = int(export["natmtot"])
    nlomax, apwordmax = int(export["nlomax"]), int(export["apwordmax"])
    oalo = jnp.zeros((apwordmax, nlomax, natmtot))
    ololo = jnp.zeros((nlomax, nlomax, natmtot))
    for ias in range(natmtot):
        is_, nr = _atom_shapes(export, ias)
        w = jnp.asarray(wr2mt[is_, :nr])
        u = apwfr[:nr, 0, :, :, ias]                       # (nr, io, l)
        v = lofr[:nr, 0, :, ias]                           # (nr, ilo)
        for ilo in range(int(nlorb[is_])):
            l = int(lorbl[is_][ilo])
            wv = w * v[:, ilo]
            for io in range(int(apword[l, is_])):
                oalo = oalo.at[io, ilo, ias].set(jnp.sum(wv * u[:, io, l]))
            for jlo in range(int(nlorb[is_])):
                if int(lorbl[is_][jlo]) == l:
                    ololo = ololo.at[ilo, jlo, ias].set(
                        jnp.sum(wv * v[:, jlo]))
    return oalo, ololo


def hamiltonian_integrals(export, potential=None, apwfr=None, lofr=None,
                          apwdfr=None):
    """`hmlrad`: the APW--APW, local-orbital--APW and
    local-orbital--local-orbital radial Hamiltonian integrals.

    Returns `(haa, hloa, hlolo)` in Elk's own index order:
    `haa[lm2, jo, l3, io, l1, ias]`, `hloa[lm2, io, l3, ilo, ias]`,
    `hlolo[lm2, jlo, ilo, ias]`.

    `potential` is the list of dense `(nr, lmmaxo)` arrays
    `potential_arrays` returns; passing it explicitly is what makes the
    result differentiable with respect to the potential.
    """
    potential = potential_arrays(export) if potential is None else potential
    apwfr = export["apwfr_full"] if apwfr is None else apwfr
    lofr = export["lofr"] if lofr is None else lofr
    apwdfr = export["apwdfr"] if apwdfr is None else apwdfr
    apword = np.asarray(export["apword"])
    nlorb = np.asarray(export["nlorb"])
    lorbl = export["lorbl"]
    wr2mt = np.asarray(export["wr2mt"])
    natmtot = int(export["natmtot"])
    nl = int(export["lmaxapw"]) + 1
    nlomax, apwordmax = int(export["nlomax"]), int(export["apwordmax"])
    lmmaxo = int(export["lmmaxo"])
    y00i = 1.0 / float(export["y00"])
    haa = jnp.zeros((lmmaxo, apwordmax, nl, apwordmax, nl, natmtot))
    hloa = jnp.zeros((lmmaxo, apwordmax, nl, nlomax, natmtot))
    hlolo = jnp.zeros((lmmaxo, nlomax, nlomax, natmtot))
    for ias in range(natmtot):
        is_, nr = _atom_shapes(export, ias)
        w = jnp.asarray(wr2mt[is_, :nr])
        vlm = jnp.asarray(potential[ias])[:nr]             # (nr, lmmaxo)
        u = apwfr[:nr, 0, :, :, ias]                       # (nr, io, l)
        hu = apwfr[:nr, 1, :, :, ias]
        ulo = lofr[:nr, 0, :, ias]                         # (nr, ilo)
        hulo = lofr[:nr, 1, :, ias]
        dfr = apwdfr[:, :, ias]                            # (io, l)
        # --- the l2 > 0 potential integrals, every pair at once.  The
        # expression is symmetric under (io, l1) <-> (jo, l3), which is why
        # Elk computes only the l1 >= l3 half and assigns both.
        uw = u * w[:, None, None]
        haa_pot = jnp.einsum("rjb,ria,rm->mjbia", u, uw, vlm)
        hloa_pot = jnp.einsum("rib,rp,rm->mibp", u, ulo * w[:, None], vlm)
        hlolo_pot = jnp.einsum("rq,rp,rm->mqp", ulo, ulo * w[:, None], vlm)
        # --- the l2 = 0 element: <u | H u>, not a potential integral
        kinetic = jnp.einsum("ria,rja->ija", uw, hu)
        kinetic = kinetic + jnp.swapaxes(kinetic, 0, 1)
        surface = (u[nr - 1][:, None, :] * dfr[None, :, :]
                   + dfr[:, None, :] * u[nr - 1][None, :, :])
        diag = (kinetic + surface) * (y00i / 2.0)          # (io, jo, l)
        lo_apw = jnp.einsum("rp,ria->pia", ulo * w[:, None], hu) * y00i
        lo_lo = jnp.einsum("rp,rq->pq", ulo * w[:, None], hulo) * y00i
        haa = haa.at[1:, :, :, :, :, ias].set(haa_pot[1:])
        hloa = hloa.at[1:, :, :, :, ias].set(hloa_pot[1:])
        hlolo = hlolo.at[1:, :, :, ias].set(hlolo_pot[1:])
        for l in range(nl):
            haa = haa.at[0, :, l, :, l, ias].set(
                jnp.transpose(diag[:, :, l]))
        for ilo in range(int(nlorb[is_])):
            l = int(lorbl[is_][ilo])
            hloa = hloa.at[0, :, l, ilo, ias].set(lo_apw[ilo, :, l])
            for jlo in range(int(nlorb[is_])):
                if int(lorbl[is_][jlo]) == l:
                    hlolo = hlolo.at[0, jlo, ilo, ias].set(lo_lo[ilo, jlo])
    return haa, hloa, hlolo


def assigned_masks(export):
    """Boolean masks marking the entries Elk actually assigns.

    Elk allocates `haa`, `hloa`, `hlolo`, `oalo` and `ololo` at the maxima
    over species (`apwordmax`, `nlomax`) and writes only the entries a given
    atom has, so the rest of the exported array is uninitialised memory and
    must be excluded from any element-wise comparison.  Returns a dict of
    masks with the same shapes as the arrays above.
    """
    idxis = np.asarray(export["idxis"]) - 1
    apword = np.asarray(export["apword"])
    nlorb = np.asarray(export["nlorb"])
    lorbl = export["lorbl"]
    natmtot = int(export["natmtot"])
    nl = int(export["lmaxapw"]) + 1
    nlomax, apwordmax = int(export["nlomax"]), int(export["apwordmax"])
    lmmaxo = int(export["lmmaxo"])
    haa = np.zeros((lmmaxo, apwordmax, nl, apwordmax, nl, natmtot), bool)
    hloa = np.zeros((lmmaxo, apwordmax, nl, nlomax, natmtot), bool)
    hlolo = np.zeros((lmmaxo, nlomax, nlomax, natmtot), bool)
    oalo = np.zeros((apwordmax, nlomax, natmtot), bool)
    ololo = np.zeros((nlomax, nlomax, natmtot), bool)
    for ias in range(natmtot):
        is_ = int(idxis[ias])
        for l1 in range(nl):
            for l3 in range(nl):
                haa[:, :int(apword[l3, is_]), l3,
                    :int(apword[l1, is_]), l1, ias] = True
        for ilo in range(int(nlorb[is_])):
            l = int(lorbl[is_][ilo])
            for l3 in range(nl):
                hloa[:, :int(apword[l3, is_]), l3, ilo, ias] = True
            oalo[:int(apword[l, is_]), ilo, ias] = True
            for jlo in range(int(nlorb[is_])):
                hlolo[:, jlo, ilo, ias] = True
                if int(lorbl[is_][jlo]) == l:
                    ololo[ilo, jlo, ias] = True
    return {"haa": haa, "hloa": hloa, "hlolo": hlolo,
            "oalo": oalo, "ololo": ololo}


def integrals_from_export(export, potential=None, apwfr=None, lofr=None,
                          apwdfr=None):
    """A copy of `export` with the five radial integrals REPLACED by the ones
    built here, so `hamiltonian.py`'s assembly consumes them unchanged."""
    haa, hloa, hlolo = hamiltonian_integrals(
        export, potential=potential, apwfr=apwfr, lofr=lofr, apwdfr=apwdfr)
    oalo, ololo = overlap_integrals(export, apwfr=apwfr, lofr=lofr)
    out = dict(export)
    out.update(haa=haa, hloa=hloa, hlolo=hlolo, oalo=oalo, ololo=ololo)
    return out
