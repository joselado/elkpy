r"""Phase 3: the self-consistent field loop, closed and iterated.

Phase 2 left two half-steps that each reproduce Elk exactly and had never been
composed into a cycle:

* §2i, density :math:`\to` potential -- the Weinert solve plus the
  exchange-correlation functional plus the symmetrisation, pointwise against
  Elk's own ``vsir``;
* §2j, potential :math:`\to` :math:`H,O` at every :math:`\bf k` :math:`\to`
  eigenvectors :math:`\to` density, with **nothing in the path reading an
  eigenvector**.

`elkjax.occupations` supplies what joined them -- the zone-summed Fermi level --
and this module closes the cycle:

.. math::

    v \;\longrightarrow\; \{u_\ell,\,u^{\rm lo}_\ell\} \;\longrightarrow\;
    H_{\bf k},O_{\bf k} \;\longrightarrow\;
    \varepsilon_{i\bf k},c_{i\bf k} \;\longrightarrow\;
    \mu,\,n_{i\bf k} \;\longrightarrow\; \rho \;\longrightarrow\;
    v_{\rm cl}[\rho]+\hat Sv_{xc}[\rho] \;=\; F(v) .

**What is held fixed, and why that still makes Elk's own answer a fixed
point.**  Four things are inputs here that ``gndstate`` recomputes every
iteration: the core density ``rhocr`` and its eigenvalue sum ``evalsumcr``
(``gencore``), the target charge ``chgtot``, and the linearisation energies
(``linengy``).  Each is a functional of :math:`v`, so freezing them changes
:math:`F` -- but every one of them is evaluated *at Elk's converged potential*,
so Elk's own :math:`v^*` still satisfies :math:`v^*=F(v^*)` and the forward
check is valid.  What is **not** valid is calling this "the SCF": the
trajectory is not Elk's, and a loop started far away converges to a slightly
different point.  That is the Phase 3 boundary, not a tolerance.

**Scalar only.**  ``eveqnsv`` is not transcribed, so a spin-polarised or
spin-orbit ground state is refused rather than silently treated as
first-variational -- :func:`check_scalar` is the refusal.

**Forward only, so far.**  The loop runs on concrete arrays: the density
accumulation skips states below ``epsocc`` with a Python ``continue``, exactly
as ``rhomagk`` does, and that comparison needs a value.  Making the chain
traceable (the skip as a zeroed weight) is what "Gradient A" of the study's
Phase 3 needs and is not done here.
"""

import numpy as np

import jax.numpy as jnp

from . import density, energy, fixedpoint, occupations, poisson, radial
from . import radial_functions

__all__ = ["check_scalar", "interstitial_potential_transform",
           "potential_from_density", "at_potential", "pack", "unpack",
           "step", "run", "total_energy"]


def check_scalar(densityk):
    """Refuse anything the first-variational problem alone cannot describe.

    ``eveqnsv`` adds the spin-orbit and exchange terms and mixes the
    first-variational states; without it ``evalsv`` is ``evalfv`` only when
    ``nstsv == nstfv``, which is Elk's own marker for a scalar calculation.
    """
    nstfv, nstsv = int(densityk["nstfv"]), int(densityk["nstsv"])
    if nstsv != nstfv:
        raise ValueError(
            f"nstsv = {nstsv} but nstfv = {nstfv}: this ground state is "
            f"spin-polarised or spin-orbit coupled, and the second-variational "
            f"step (eveqnsv) is not transcribed. Only a scalar calculation "
            f"can be iterated here.")
    if int(densityk["nspnfv"]) != 1:
        raise ValueError("spin-spiral ground states are not supported here")


# ------------------------------------------------------- density -> potential


def interstitial_potential_transform(vsir, groundstate, ngvc=None):
    r"""``genvsig``: what ``hmlistl`` actually looks up.

    .. math::

        \tilde v_s({\bf G}) = \frac1{N}\sum_{\bf r}
        v_s({\bf r})\,\Theta({\bf r})\,e^{-i{\bf G}\cdot{\bf r}} ,

    the transform of the potential **times the characteristic function**, not
    of the potential.  That factor is `rfirftoc.f90`'s first line
    (``zfft = rfir*cfunir``) and it is easy to lose: ``rfirctof``, the map back
    to the fine grid that the density uses, has no such factor, so
    :func:`elkjax.density.coarsen` -- which is `rfirctof`'s exact inverse --
    is **not** `rfirftoc`.  Measured, using it here is wrong by a factor of
    order 10 and drives the spectrum below ``e0min``, where every occupancy is
    gated to zero.

    ``potks`` routes this through the coarse grid (``rfirftoc`` then
    ``genvsig``), but that round trip preserves exactly the ``ngvc``
    components ``vsig`` keeps, so it is done on the fine grid in one step.

    **Against Elk's own** ``vsig`` **this lands at 2.6e-8, and that is Elk's
    mixing step rather than an error here.**  ``init0.f90:633-641`` makes the
    mixer's target array ``vsbs`` = [``vsmt``, ``vsirc``] -- the COARSE
    interstitial potential -- and ``vsir`` is a separate array that ``potks``
    fills from ``vclir+vxcir`` and nothing mixes.  So ``gndstate`` exports a
    ``vsir`` one un-mixed step ahead of the ``vsig`` that ``genvsig`` built
    from the mixed ``vsirc``.  Measured: 2.6e-8 at Elk's default
    ``epspot=1e-6`` and 2.0e-9 at ``epspot=1e-8``.  It also means Elk's own
    SCF variable is :math:`(v^{\rm MT}, \tilde v^{\rm I}_{\rm coarse})`
    while this module iterates :math:`(v^{\rm MT}, v^{\rm I}_{\rm fine})`;
    the extra high-:math:`|G|` tail of the fine array is inert -- nothing in
    :func:`step` reads it -- so the two have the same fixed point.
    """
    ngvc = int(np.asarray(groundstate["vsig"]).size if ngvc is None else ngvc)
    fine_grid = tuple(int(n) for n in groundstate["ngridg"])
    igfft = np.asarray(groundstate["igfft"])[:ngvc] - 1
    spectrum = density._forward_fft(
        jnp.asarray(vsir) * jnp.asarray(groundstate["cfunir"]), fine_grid)
    return spectrum[jnp.asarray(igfft)]


def potential_from_density(rhomt, rhoir, groundstate):
    r"""``potks``: :math:`v_s=v_{\rm cl}[\rho]+\hat Sv_{xc}[\rho]`.

    ``rhomt`` is a stack of Elk-**packed** muffin-tin arrays and ``rhoir`` is
    the interstitial on the fine grid, i.e. what ``density.converged_density``
    produces once each atom is passed through ``pack_fine``.  Returns
    ``(vsmt packed, vsir on the fine grid)`` in the same packing
    ``lapw["vsmt"]`` arrives in, which is what ``radial.potential_arrays``
    unpacks.

    Two asymmetries of ``potks`` are Elk's own and are kept: the
    exchange-correlation potential is symmetrised and the energy densities are
    not (``potxc.f90:55-58``), and ``trimrfg``'s low-pass is applied to
    ``vxcir`` and **not** to ``vclir``.  §2i measured that the second is
    invisible against every reference but ``vsir`` itself.
    """
    local = dict(groundstate)
    local["rhomt"], local["rhoir"] = rhomt, rhoir
    coulomb_mt, coulomb_ir = poisson.coulomb_potential(local)
    exchange = energy.kohn_sham_potentials(local, symmetrise=True)
    natmtot = int(groundstate["natmtot"])
    vsmt = jnp.stack([poisson.pack(coulomb_mt[ias], groundstate, ias)
                      + exchange["vxcmt"][ias] for ias in range(natmtot)])
    return vsmt, jnp.asarray(coulomb_ir) + jnp.asarray(exchange["vxcir"])


# ------------------------------------------------------- potential -> density


def at_potential(lapw, groundstate, densityk, vsmt, vsir):
    """Copies of ``lapw`` and ``groundstate`` carrying a new potential.

    The potential reaches the eigenproblem by **two** routes and both are
    rebuilt here: through the muffin-tin radial functions and the radial
    integrals built from them (§1k), and through ``vsig`` in the interstitial.

    Rebuilding the radial functions rebuilds ``dmat`` and hence ``apwalm``,
    which §1k measured to be worth 21% of the potential derivative and which no
    gradient check could see -- so the SAME refreshed ``lapw`` must be handed to
    the eigensolve and to the density accumulation, or the eigenvectors come
    from one basis and the wavefunctions are matched with another.
    """
    potential = radial.potential_arrays(lapw, packed=vsmt)
    refreshed = radial_functions.radial_functions_from_export(
        lapw, potential=potential)
    refreshed = radial.integrals_from_export(refreshed, potential=potential)
    refreshed["vsmt"] = vsmt
    local = dict(groundstate)
    local["vsig"] = interstitial_potential_transform(vsir, groundstate)
    local["vsir"] = vsir
    return refreshed, local


def density_at_potential(lapw, groundstate, densityk, vsmt, vsir):
    """One half-step: a potential to ``(rhomt packed, rhoir, mu, occsv, evalsv)``.

    All of ``rhomag``: the zone eigensolve, ``occupy``, ``rhomagk``,
    ``rhomagsh``, ``symrf``, the two interpolations, ``rhocore`` and
    ``rhonorm``.
    """
    refreshed, local = at_potential(lapw, groundstate, densityk, vsmt, vsir)
    vectors, evalsv = density.solve_zone(refreshed, local, densityk,
                                         eigenvalues=True)
    mu, occsv = occupations.occupy(evalsv, densityk)

    # concrete, not traced: `rhomagk`'s `epsocc` skip is a Python branch on the
    # occupation, and reproducing Elk means keeping it.  See the module note.
    occupied = dict(densityk)
    occupied["occsv"] = np.asarray(occsv)
    muffin, interstitial = density.converged_density(
        occupied, groundstate, refreshed, vectors=vectors)
    natmtot = int(groundstate["natmtot"])
    packed = jnp.stack([density.pack_fine(muffin[ias], groundstate, ias)
                        for ias in range(natmtot)])
    return dict(rhomt=packed, rhoir=interstitial, mu=mu, occsv=occsv,
                evalsv=evalsv)


# ------------------------------------------------------------- the fixed point


def pack(vsmt, vsir):
    """``mixpack``: the two halves of the potential as one flat real vector."""
    return jnp.concatenate([jnp.asarray(vsmt).reshape(-1),
                            jnp.asarray(vsir).reshape(-1)])


def unpack(vector, shape, size):
    """The inverse of :func:`pack`, given the muffin-tin shape and ``ngtot``."""
    count = int(np.prod(shape))
    return (jnp.asarray(vector)[:count].reshape(shape),
            jnp.asarray(vector)[count:count + size])


def step(vector, arguments):
    """One Kohn-Sham iteration as :math:`F(v)`, on the packed potential.

    Signature is ``(v, theta)`` so it drops straight into
    `elkjax.fixedpoint.iterate`; ``arguments`` is
    ``(lapw, groundstate, densityk)``.
    """
    lapw, groundstate, densityk = arguments
    shape = (int(groundstate["natmtot"]), int(np.asarray(lapw["vsmt"]).shape[1]))
    vsmt, vsir = unpack(vector, shape, int(groundstate["ngtot"]))
    got = density_at_potential(lapw, groundstate, densityk, vsmt, vsir)
    return pack(*potential_from_density(got["rhomt"], got["rhoir"],
                                        groundstate))


def total_energy(lapw, groundstate, densityk, vsmt, vsir):
    r"""``energy.f90`` at a given potential, with only :math:`E_{nn}` imported.

    The occupations come from this module's own ``occupy``, so
    :math:`\Sigma_\varepsilon` and :math:`E_{TS}` -- the two scalars §2f had to
    import -- are computed here.  What is still imported is the core half of
    :math:`\Sigma_\varepsilon` (patch 0023's ``evalsumcr``, an input at fixed
    potential like ``rhocr``) and :math:`E_{nn}`, which is a property of the
    lattice rather than of the density.
    """
    got = density_at_potential(lapw, groundstate, densityk, vsmt, vsir)
    local = dict(groundstate)
    local["rhomt"], local["rhoir"] = got["rhomt"], got["rhoir"]
    inputs = occupations.inputs(densityk)
    return energy.terms(
        local,
        evalsum=energy.eigenvalue_sum(got["evalsv"], got["occsv"],
                                      inputs["wkpt"],
                                      float(densityk["evalsumcr"])),
        engyts=energy.entropy_term(got["occsv"], inputs["wkpt"],
                                   inputs["swidth"], inputs["occmax"],
                                   inputs["stype"])), got


def run(lapw, groundstate, densityk, vsmt=None, vsir=None, mixing=0.4,
        tol=1e-7, maxiter=60, history=0):
    """Iterate :math:`v=F(v)` from a starting potential.

    ``vsmt``/``vsir`` default to Elk's converged potential, which is a fixed
    point and therefore tests nothing -- pass a perturbed one.  ``history=0``
    is linear mixing; ``history>0`` switches `elkjax.fixedpoint.iterate` to
    Anderson, which reaches a *different point inside the same tolerance ball*.

    Returns ``(vsmt, vsir, iterations, residual)``.  The residual is the norm
    of :math:`F(v)-v` on the packed vector, so it is not Elk's ``epspot`` (an
    RMS over the same arrays) and the two should not be compared as if they
    were.
    """
    check_scalar(densityk)
    vsmt = jnp.asarray(lapw["vsmt"]) if vsmt is None else jnp.asarray(vsmt)
    vsir = (jnp.asarray(groundstate["vsir"]) if vsir is None
            else jnp.asarray(vsir))
    shape = tuple(int(n) for n in vsmt.shape)
    solution, iterations, residual = fixedpoint.iterate(
        step, (lapw, groundstate, densityk), pack(vsmt, vsir),
        mixing=mixing, tol=tol, maxiter=maxiter, history=history)
    return unpack(solution, shape,
                  int(groundstate["ngtot"])) + (iterations, residual)
