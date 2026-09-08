r"""Phase 3 §3c: a calculation run from the input file, not from Elk's answer.

Everything in `elkjax.scf` iterates :math:`v = F(v)`, and until now the only
place to start it was Elk's own converged potential -- which is a fixed point,
so the loop had nothing to do.  Phase 3 §3b started it 0.30 away in potential
norm by perturbing that potential *by hand*, which tests the map but is not a
calculation: the starting point was still built out of the answer.

This module removes that.  ``elkpy``'s new task **9006**
(`patches/0024-initial-state.patch`) is `gndstate.f90`'s own
``trdstate=.false.`` branch --

.. code-block:: none

    init0; init1; rhoinit; potks(.true.); genvsig

-- followed by the top of its first self-consistent iteration --

.. code-block:: none

    gencore; linengy; genapwlofr; gensocfr; genevfsv; occupy

-- and nothing after it.  No ``rhomag``, no ``potks`` on a new density, no
``mixerifc``.  The export queries (patches 0013-0023) then describe the state
at the **start of Elk's first iteration**, with the density a superposition of
free atomic densities, and :func:`run` iterates from there to
self-consistency in JAX.

**What Elk still does, said plainly.**  The initialisation: the grids, the
species radial meshes, the G-vector and k-point sets, the Gaunt coefficients,
the symmetrisation operators, and the free-atom densities ``rhoinit``
superposes.  None of that is a functional of the density -- it is the same for
every iteration and for every potential -- but it is real work, and
transcribing it (``atom.f90``'s free-atom solver above all) is a separate
project.  What elkjax does here is the self-consistent field itself: every
map from a potential to the next potential, and the total energy of the
result.

**Three things are frozen that Elk recomputes each iteration**, and they are
evaluated once, at the *initial* potential rather than at the converged one:

* the core density ``rhocr`` and its kinetic energy ``engykncr``
  (``gencore``) -- the one that costs something, because the core relaxes as
  the valence density does.  Measured on bulk Si (§3c): **3.6e-4 Ha** in the
  total energy and **4.3e-5 Ha** in the Fermi level, and that is the WHOLE of
  the remaining gap -- swapping Elk's converged ``rhocr``/``engykncr`` into
  the same initial triple, changing nothing else, takes the same run to
  3.8e-8 Ha and 4.5e-9 Ha.  Note it is :math:`T_{\rm core}` that is frozen and
  not the core eigenvalue sum; `elkjax.energy.core_eigenvalue_sum` says why,
  and getting it the other way round is worth 2.0 Ha;
* the linearisation energies ``apwe``/``lorbe`` (``linengy``) -- **exact**
  whenever the species files set every ``apwve``/``lorbve`` flag false and
  ``autolinengy`` is off, which is Elk's default and silicon's species file.
  ``linengy`` then copies ``apwe0``/``lorbe0`` and does nothing else, so
  freezing it changes nothing at all.  :func:`check_linearisation_frozen` is
  the check, and it *raises* when the species file would have searched;
* the target charge ``chgtot``, which is :math:`\sum_a Z_a` and does not move.

**Memory.**  This runs a real eigensolve at every k-point of the set, so cap
the process (`elkjax.memory.limit_address_space`) and pin it with
``taskset -c 0-3`` -- see ``docs/jax_port_status.md``'s "Memory and CPU
discipline".  Bulk Si at ``rgkmax=7``, ``ngridk=(2,2,2)`` is
:math:`n_{\rm mat}\approx160` and fits easily; a monolayer with vacuum does
not.
"""

import collections

import numpy as np

from . import scf

__all__ = ["Result", "initial_state", "converged_state",
           "check_linearisation_frozen", "run"]


Result = collections.namedtuple(
    "Result",
    "energy mu vsmt vsir rhomt rhoir iterations residual terms")
Result.__doc__ = """The outcome of a self-consistent elkjax run.

``energy`` is the total energy in Hartree, ``mu`` the Fermi level in Hartree,
``vsmt``/``vsir`` the converged Kohn-Sham potential in Elk's own packing,
``rhomt``/``rhoir`` the density it came from, ``iterations`` and ``residual``
the fixed-point solver's own report (the residual is the norm of
:math:`F(v)-v` on the packed vector, NOT Elk's ``epspot``, which is an RMS),
and ``terms`` the full `elkjax.energy.terms` dictionary behind ``energy``.
"""


def initial_state(calculation):
    """The three export dicts at the top of Elk's first iteration.

    Runs task 9006 followed by the task-9002 session (see
    `elkpy.calculation.Calculation.initial_state_session`).  **No converged
    ground state is needed, and none is computed** -- this is the entry point
    that makes a run from the input file alone possible.

    Returns ``(groundstate, densityk, lapw)``, the same triple
    `converged_state` returns and the same one ``tests/test_calculation_scf.py``
    builds its fixtures from.
    """
    with calculation.initial_state_session() as session:
        return (session.ground_state(), session.density_k(),
                session.lapw_problem((0.0, 0.0, 0.0)))


def converged_state(calculation):
    """The same triple at Elk's own converged ground state.

    This is the Phase 2/3 path, kept here so a run can be compared against the
    answer it is supposed to reproduce without the caller having to know which
    session to open.  It DOES run Elk's full SCF (``ensure_ground_state``).
    """
    with calculation.eigenstate_session() as session:
        return (session.ground_state(), session.density_k(),
                session.lapw_problem((0.0, 0.0, 0.0)))


def check_linearisation_frozen(lapw):
    """Refuse a species whose linearisation energies Elk would have searched.

    ``linengy.f90`` only calls ``findband`` for an APW or local orbital whose
    species file sets ``apwve``/``lorbve`` true; otherwise it leaves
    ``apwe``/``lorbe`` at the ``apwe0``/``lorbe0`` that ``init1`` copied in
    (or, with ``autolinengy``, at :math:`E_F+\\Delta`).  Freezing the
    linearisation energies across this loop is therefore **exact** in the
    first case and an approximation in the other two.

    The flags themselves are exported (patch 0024), so this reads them
    rather than inferring anything from ``apwe``: a searched energy and a
    default one are the same kind of number, and comparing them would only
    say whether ``findband`` happened to move this one.
    """
    if lapw.get("apwve") is None:
        raise KeyError(
            "this LAPW export carries no apwve/lorbve flags -- rebuild the "
            "binary with patches/0024-initial-state.patch, or pass "
            "check_linearisation=False and accept that freezing the "
            "linearisation energies may not be exact")
    if lapw["autolinengy"]:
        raise ValueError(
            "autolinengy is on, so Elk sets every linearisation energy to "
            "efermi + dlefe and moves them as the Fermi level moves. "
            "linengy is not transcribed, so freezing them here is not exact "
            "-- see docs/jax_port_phase3.md #3c.")
    for name in ("apwve", "lorbve"):
        searched = np.asarray(lapw[name])
        if searched.any():
            raise ValueError(
                f"{searched.sum()} of this species file's {name} flags are "
                f"true, so Elk re-searches those linearisation energies "
                f"(linengy -> findband) every SCF iteration. linengy is not "
                f"transcribed, so freezing them here is not exact -- see "
                f"docs/jax_port_phase3.md #3c.")


def run(calculation=None, mixing=0.4, tol=1e-7, maxiter=100, history=0,
        state=None, check_linearisation=True):
    """Run the Kohn-Sham loop in JAX, starting from Elk's initialisation.

    ``calculation`` is an `elkpy.calculation.Calculation`; nothing about it
    needs to have been run.  ``state`` accepts an already-pulled
    ``(groundstate, densityk, lapw)`` triple so a session is not re-opened
    when several runs share one -- pass :func:`initial_state`'s output.

    ``mixing`` and ``history`` go to `elkjax.fixedpoint.iterate`:
    ``history=0`` is linear mixing, ``history>0`` Anderson.  The
    atomic-superposition start is much farther from the answer than §3b's
    hand-perturbed one, so Anderson is worth trying before tuning ``mixing``.

    Returns a :class:`Result`.
    """
    if state is None:
        if calculation is None:
            raise TypeError("pass a Calculation, or a `state` triple from "
                            "initial_state()")
        state = initial_state(calculation)
    groundstate, densityk, lapw = state
    scf.check_scalar(densityk)
    if check_linearisation:
        check_linearisation_frozen(lapw)

    vsmt, vsir, iterations, residual = scf.run(
        lapw, groundstate, densityk, mixing=mixing, tol=tol, maxiter=maxiter,
        history=history)
    terms, got = scf.total_energy(lapw, groundstate, densityk, vsmt, vsir)
    return Result(energy=float(terms["engytot"]), mu=float(got["mu"]),
                  vsmt=vsmt, vsir=vsir,
                  rhomt=got["rhomt"], rhoir=got["rhoir"],
                  iterations=iterations, residual=residual, terms=terms)
