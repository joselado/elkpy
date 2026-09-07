r"""Phase 2 item 2f: the total energy, from the density and the JAX chain.

This is the first thing in Phase 2 that uses more than one of its own pieces at
once, and that is the point of doing it.  §2a/§2b give the exchange-correlation
functional, §2c the cell integral, §2e the Poisson solve; each was checked
against Elk on its own, and none of those checks says the four are *mutually*
consistent -- that the potential built here is the one integrated there, against
the density they all share.  ``energy.f90``'s own decomposition says so, term by
term, and patch 0017 exports Elk's thirteen scalars at full precision so the
comparison is not limited to ``INFO.OUT``'s print width.

**What is reproduced and what is imported.**  For a non-magnetic ground state
``energy.f90`` assembles

.. math::

    E_{\rm tot}=E_{\rm kin}+\tfrac12E_{v_{cl}}+E_{\rm Mad}+E_x+E_c+E_{TS},
    \qquad E_{\rm kin}=\Sigma_\varepsilon-E_{v_{cl}}-E_{v_{xc}},

so every term but :math:`\Sigma_\varepsilon` (the occupation-weighted eigenvalue
sum) and :math:`E_{TS}` (the smearing entropy) is a functional of the density and
the potentials built from it.  Those two need the second-variational step and a
zone sum, which is Phase 3, so they are **imported** from the same export -- as
is :math:`E_{nn}`, the nucleus-nucleus Madelung constant from ``energynn``, which
is a property of the lattice and not of the density at all.  Nothing here claims
to transcribe ``energy.f90``; what it claims is the density-functional half.

**§2d predicts one disagreement, and it is asserted rather than tolerated.**
:math:`E_{v_{xc}}=\int\rho\,v_{xc}` uses Elk's *symmetrised* muffin-tin
:math:`v_{xc}` (``potxc.f90:55-58``), while the pointwise :math:`v_{xc}[\rho]`
built here is not symmetrised.  On a symmetric cell the two must differ, by
roughly the muffin-tin fraction of :math:`E_{v_{xc}}` times §2d's 1.2e-4 -- and
on a ``symtype=0`` ground state they must agree.  That is a prediction §2d made
before this module existed, and :mod:`tests.test_calculation_energy` checks both
directions.
"""

import numpy as np

import jax.numpy as jnp

from . import grid, integrate, poisson, symmetry, xc

__all__ = ["madelung", "coulomb_terms", "exchange_correlation_terms",
           "kohn_sham_potentials", "eigenvalue_sum", "entropy_term", "terms",
           "report"]

Y00 = 0.28209479177387814347


def madelung(vclmt, groundstate):
    r""":math:`E_{\rm Mad}=\tfrac12\sum_\alpha Z_\alpha[V_{cl}(0)-V_{\rm nuc}(0)]y_{00}`.

    ``energy.f90`` reads ``vclmt(1,ias)`` -- the :math:`l=0` coefficient at the
    FIRST radial point, not at the boundary -- and subtracts ``vcln(1,is)``, the
    nucleus's own divergent contribution, leaving the potential every OTHER
    charge produces at the site.  ``spzn`` is negative in Elk's convention, so
    no sign is inserted here.

    ``vclmt`` is a stack of dense ``(nr, lmmaxo)`` arrays, as
    :func:`elkjax.poisson.coulomb_potential` returns.
    """
    total = 0.0
    for ias in range(int(groundstate["natmtot"])):
        isp = int(groundstate["idxis"][ias]) - 1
        nuclear = float(groundstate["vcln"][isp][0])
        total = total + (float(groundstate["spzn"][isp])
                         * (vclmt[ias][0, 0] - nuclear) * Y00 / 2.0)
    return total


def coulomb_terms(vclmt, vclir, groundstate):
    """``energy.f90``'s Coulomb block, given a Coulomb potential.

    The split into ``engyen``/``engyhar`` is Elk's own and is not two
    independent integrals: only :math:`E_{v_{cl}}` and :math:`E_{\\rm Mad}` are
    computed, and the electron-nucleus and Hartree pieces are algebra on them
    plus the imported :math:`E_{nn}`.  Reproducing that algebra rather than
    building the two integrals separately is deliberate -- it is what makes a
    term-by-term comparison a check of the CONVENTION and not of a second
    quadrature.
    """
    packed = jnp.stack([poisson.pack(vclmt[ias], groundstate, ias)
                        for ias in range(int(groundstate["natmtot"]))])
    engyvcl = integrate.cell_inner_product(
        groundstate["rhomt"], groundstate["rhoir"], packed, vclir, groundstate)
    engymad = madelung(vclmt, groundstate)
    engynn = float(groundstate["engynn"])
    engyen = 2.0 * (engymad - engynn)
    engyhar = (engyvcl - engyen) / 2.0
    return dict(engyvcl=engyvcl, engymad=engymad, engyen=engyen,
                engyhar=engyhar, engycl=engynn + engyen + engyhar)


def kohn_sham_potentials(groundstate, symmetrise=False):
    r"""The pointwise :math:`v_{xc}`, :math:`\varepsilon_x`, :math:`\varepsilon_c`.

    Muffin tin via the angular grid (§2d) and interstitial pointwise (§2a), both
    from Elk's own density.  ``vxcir`` gets ``potks``'s ``trimrfg`` low-pass and
    the energy densities do not, which is Elk's own asymmetry and not a choice
    made here.

    ``symmetrise=True`` applies §2g's operator to the muffin-tin
    :math:`v_{xc}`, which is what makes it Elk's ``vxcmt`` rather than
    something $1.2\\times10^{-4}$ away from it (§2d) -- necessary for an SCF
    iteration, which compares potentials pointwise, and *irrelevant* to
    everything in this module, since §2f showed the difference is orthogonal to
    :math:`\\rho`.  It defaults to off so that the energy terms are computed
    from the raw functional and the two facts stay separable.

    The energy densities are never symmetrised: Elk does not symmetrise them
    either (``potxc.f90:55-58``), and doing so here would break the exact
    agreement §2d measured.
    """
    natmtot = int(groundstate["natmtot"])
    out = {}
    for key in ("vxcmt", "exmt", "ecmt"):
        out[key] = []
    for ias in range(natmtot):
        rho = jnp.asarray(grid.to_angular(groundstate["rhomt"][ias],
                                          groundstate, ias))
        ex, ec, vx, _, vc, _ = xc.pwca(0.5 * rho, 0.5 * rho)
        for key, values in (("vxcmt", vx + vc), ("exmt", ex), ("ecmt", ec)):
            out[key].append(grid.from_angular(jnp.asarray(values),
                                              groundstate, ias))
    for key in ("vxcmt", "exmt", "ecmt"):
        out[key] = jnp.stack(out[key])

    if symmetrise:
        dense = jnp.stack([poisson.dense(out["vxcmt"][ias], groundstate, ias)
                           for ias in range(natmtot)])
        moved = symmetry.symmetrise(dense, groundstate)
        out["vxcmt"] = jnp.stack([poisson.pack(moved[ias], groundstate, ias)
                                  for ias in range(natmtot)])

    rho = jnp.asarray(groundstate["rhoir"])
    ex, ec, vx, _, vc, _ = xc.pwca(0.5 * rho, 0.5 * rho)
    out["vxcir"] = grid.trim(vx + vc, groundstate)
    out["exir"], out["ecir"] = ex, ec
    return out


def exchange_correlation_terms(potentials, groundstate):
    r""":math:`E_{v_{xc}}`, :math:`E_x` and :math:`E_c` as :math:`\int\rho\,f`."""
    rhomt, rhoir = groundstate["rhomt"], groundstate["rhoir"]
    out = {}
    for name, mt, ir in (("engyvxc", "vxcmt", "vxcir"),
                         ("engyx", "exmt", "exir"),
                         ("engyc", "ecmt", "ecir")):
        out[name] = integrate.cell_inner_product(
            rhomt, rhoir, potentials[mt], potentials[ir], groundstate)
    return out


def eigenvalue_sum(evalsv, occsv, wkpt, evalsumcr):
    r"""``energy.f90``'s :math:`\Sigma_\varepsilon`, the occupied eigenvalue sum.

    .. math::

        \Sigma_\varepsilon = \sum_{\alpha,\,{\rm core}}
        n_c\,\varepsilon_c
        \;+\;
        \sum_{\bf k}w_{\bf k}\sum_i n_{i\bf k}\,\varepsilon_{i\bf k}.

    The **core** half is imported (patch 0023's ``evalsumcr``) for the same
    reason patch 0021's ``rhocr`` is: the core states are a functional of the
    potential, so at fixed potential they are an input and not a step declined.
    The valence half is the zone sum, and it is what
    :mod:`elkjax.occupations` makes computable -- so this closes one of the two
    scalars §2f had to import.
    """
    return (jnp.sum(jnp.asarray(wkpt)[:, None] * jnp.asarray(occsv)
                    * jnp.asarray(evalsv)) + evalsumcr)


def entropy_term(occsv, wkpt, swidth, occmax, stype=3):
    r"""``energy.f90``'s :math:`E_{TS}`, the smearing's free-energy term.

    .. math::

        E_{TS} = \sigma\,n_{\max}\sum_{\bf k}w_{\bf k}\sum_i
        \big[f\ln f + (1-f)\ln(1-f)\big],
        \qquad f = n_{i\bf k}/n_{\max},

    which is Elk's :math:`-\sigma S/k_B` written without the two factors of
    :math:`k_B` that cancel.  It is **non-zero only for Fermi-Dirac smearing**:
    ``energy.f90:242`` sets it to zero for every other ``stype``, which the
    study flags as the source of a force error one would otherwise chase for a
    week, so the branch is transcribed rather than assumed away.

    Fully occupied and empty states contribute nothing and their logarithms do
    not exist, so the argument is made safe *before* the log -- the discarded
    branch of a ``jnp.where`` still propagates a ``NaN`` through the gradient.
    """
    if stype != 3:
        return jnp.asarray(0.0)
    f = jnp.asarray(occsv) / occmax
    inside = (f > 0.0) & (f < 1.0)
    safe = jnp.where(inside, f, 0.5)
    terms = jnp.where(inside,
                      safe * jnp.log(safe) + (1.0 - safe) * jnp.log(1.0 - safe),
                      0.0)
    return swidth * occmax * jnp.sum(jnp.asarray(wkpt)[:, None] * terms)


def terms(groundstate, poisson_potential=True, symmetrise=False,
          evalsum=None, engyts=None):
    """Every term of ``energy.f90`` this phase can build, plus the total.

    ``poisson_potential=False`` substitutes Elk's own ``vclmt``/``vclir`` for
    the ones §2e builds, which isolates the Poisson solve from the quadrature:
    if the two differ, the Coulomb terms are the reason.

    ``symmetrise`` must change nothing, and that is a result rather than a
    default: §2f measured the symmetrisation leak to be orthogonal to
    :math:`\\rho`, so both settings give the same energy to roundoff.
    """
    if poisson_potential:
        vclmt, vclir = poisson.coulomb_potential(groundstate)
    else:
        vclmt = jnp.stack([poisson.dense(groundstate["vclmt"][ias],
                                         groundstate, ias)
                           for ias in range(int(groundstate["natmtot"]))])
        vclir = jnp.asarray(groundstate["vclir"])
    out = dict(coulomb_terms(vclmt, vclir, groundstate))
    out.update(exchange_correlation_terms(
        kohn_sham_potentials(groundstate, symmetrise=symmetrise), groundstate))

    # `evalsum` and `engyts` default to the export's own converged scalars --
    # §2f had no zone sum and no occupations to build them from.  Phase 3 does:
    # `elkjax.scf` passes both, computed by `eigenvalue_sum` and
    # `entropy_term`, and then the only scalar still imported is `engynn`,
    # which is a property of the lattice rather than of the density.
    if evalsum is None:
        evalsum = float(groundstate["evalsum"])
    if engyts is None:
        engyts = float(groundstate["engyts"])
    out["evalsum"], out["engyts"] = evalsum, engyts
    out["engykn"] = evalsum - out["engyvcl"] - out["engyvxc"]
    out["engytot"] = (out["engykn"] + out["engyvcl"] / 2.0 + out["engymad"]
                      + out["engyx"] + out["engyc"] + engyts)
    return out


def report(groundstate, poisson_potential=True):
    got = terms(groundstate, poisson_potential=poisson_potential)
    print("  %-9s %20s %20s %12s" % ("term", "elkjax", "elk", "relative"))
    for name in ("engyvcl", "engymad", "engyen", "engyhar", "engycl",
                 "engyvxc", "engyx", "engyc", "engykn", "engytot"):
        mine, reference = float(got[name]), float(groundstate[name])
        rel = abs(mine - reference) / max(abs(reference), 1e-300)
        print("  %-9s %+20.12f %+20.12f %12.3e" % (name, mine, reference, rel))
    return got
