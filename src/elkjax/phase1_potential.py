r"""Phase 1 item 1k: differentiate the LAPW spectrum with respect to the
muffin-tin Kohn-Sham potential.

`radial_functions.py` and `radial.py` between them make the first-variational
spectrum a function of `vsmt` rather than of a set of imported radial
integrals.  This is the experiment that measures what that buys, and what it
costs, on Elk's own matrices.

Three quantities, and the interesting one is the difference between the first
two.

**The frozen-basis derivative.**  Hold the radial functions fixed and let only
the radial integrals see the perturbation.  The overlap matrix is then
independent of the potential, so first-order perturbation theory for the
generalised eigenproblem collapses to

.. math::  \delta\varepsilon_n = c_n^\dagger\,\delta H\,c_n ,

with :math:`c_n` normalised as :math:`c_n^\dagger Oc_n=1`.  That is an exact,
independently computable closed form -- the ordinary
:math:`\langle\psi_n|\delta V|\psi_n\rangle` -- so this branch has an oracle
that owes nothing to finite differences.

**The full derivative.**  Let the perturbation reach the radial functions too:
`genapwfr` re-solves the radial Schrodinger equation in the perturbed
potential at the SAME linearisation energies, re-normalises and
re-orthogonalises, and the matching coefficients change with it.  The LAPW
basis is potential-dependent, so this is genuinely a different number.

**The basis response** is their difference, and it is the term any
frozen-basis argument drops.  It is the potential-space analogue of the
finding `docs/jax_port_phase1.md` §1e records for :math:`k`: LAPW's basis
depends on the parameter being differentiated, so Hellmann-Feynman does not
hold exactly and the correction is a real, measurable quantity rather than
roundoff.

The linearisation energies `apwe`/`lorbe` are held FIXED throughout.  Letting
them float would mean differentiating `linengy`, which re-solves them from the
potential by a bisection on the logarithmic derivative -- a different object,
and one Elk's own forces do not carry either.

Run as ``python3 -m elkjax.phase1_potential <workdir>`` after a ground state
exists there; with no argument it prints what it needs.
"""

import sys

import numpy as np

import jax
import jax.numpy as jnp

from . import hamiltonian as ham, radial, radial_functions as rf
from .memory import limit_address_space

limit_address_space()


def spectrum(export, packed, frozen_basis=False, nocc=None):
    """The first-variational spectrum as a function of the packed potential.

    With ``frozen_basis`` the radial functions are taken from the export and
    only the radial integrals respond; otherwise they are rebuilt from
    ``packed``.  ``nocc`` truncates to the lowest `nocc` eigenvalues, whose
    SUM is the safe scalar to differentiate at a multiplet.
    """
    potential = radial.potential_arrays(export, packed)
    if frozen_basis:
        source = export
    else:
        source = rf.radial_functions_from_export(export, potential=potential)
    rebuilt = radial.integrals_from_export(
        source, potential=potential, apwfr=source["apwfr_full"],
        lofr=source["lofr"], apwdfr=source["apwdfr"])
    h, o = ham.assemble_from_export(rebuilt)
    reduced, _ = ham.cholesky_reduce(h, o)
    evals = jnp.linalg.eigvalsh(reduced)
    return evals if nocc is None else evals[:nocc]


def band_energy(export, packed, nocc, frozen_basis=False):
    """:math:`\\sum_{n<n_{\\rm occ}}\\varepsilon_n` -- the scalar every
    derivative below is taken of.

    A SUM rather than one eigenvalue, because a degenerate group's individual
    branches are not differentiable and its trace is (`docs/jax_port_phase1.md`
    §1h); and because the trace of the occupied window is the quantity Phase 2
    actually needs.
    """
    return jnp.sum(spectrum(export, packed, frozen_basis=frozen_basis,
                            nocc=nocc))


def perturbation_theory_reference(export, direction, nocc):
    r"""The frozen-basis derivative in closed form: :math:`\sum_n c_n^\dagger\,
    \delta H\,c_n` over the occupied window, with Elk's own `evecfv`.

    At fixed basis the overlap does not respond, so this is exactly first-order
    perturbation theory, and it is an oracle for the frozen-basis AD branch
    that involves no finite difference and no eigensolve of a perturbed
    matrix.
    """
    dv = radial.potential_arrays(export, jnp.asarray(direction))
    zero = [jnp.zeros_like(v) for v in dv]
    integrals = radial.integrals_from_export(export, potential=dv)
    # olpfv does not see the potential at all, so dH is the muffin-tin
    # Hamiltonian built from the perturbation's own radial integrals; the
    # interstitial block is held fixed and contributes nothing.
    del zero
    dh = ham.muffin_tin_hamiltonian(integrals)
    nmatp = int(export["nmatp"])
    dh_full = jnp.zeros((nmatp, nmatp), dtype=complex).at[
        :dh.shape[0], :dh.shape[1]].set(dh)
    c = jnp.asarray(export["evecfv"])[:, :nocc]
    o = jnp.asarray(export["omat"])
    norm = jnp.einsum("in,ij,jn->n", c.conj(), o, c).real
    return float(jnp.sum(
        jnp.einsum("in,ij,jn->n", c.conj(), dh_full, c).real / norm))


def central_difference(fn, packed, direction, step):
    plus = fn(packed + step * direction)
    minus = fn(packed - step * direction)
    return float((plus - minus) / (2.0 * step))


def run(export, nocc=None, seed=17, steps=(1e-4, 1e-5, 1e-6)):
    """AD against central FD of the same function, both branches, plus the
    closed form and the basis-response term."""
    nocc = int(export["nstfv"]) // 2 if nocc is None else nocc
    packed = jnp.asarray(np.asarray(export["vsmt"]))
    rng = np.random.default_rng(seed)
    direction = jnp.asarray(rng.normal(size=packed.shape))
    direction = direction * (packed != 0.0)      # stay inside the packing
    out = {"nocc": nocc}
    for tag, frozen in (("frozen", True), ("full", False)):
        fn = lambda p, frozen=frozen: band_energy(export, p, nocc, frozen)
        grad = float(jax.jvp(fn, (packed,), (direction,))[1])
        out[f"ad_{tag}"] = grad
        out[f"fd_{tag}"] = [central_difference(fn, packed, direction, h)
                            for h in steps]
    out["closed_form"] = perturbation_theory_reference(export, direction, nocc)
    out["basis_response"] = out["ad_full"] - out["ad_frozen"]
    out["steps"] = list(steps)
    return out


def report(result):
    lines = [f"occupied window: lowest {result['nocc']} first-variational bands",
             "",
             f"  frozen-basis AD      {result['ad_frozen']: .12e}",
             f"  first-order PT       {result['closed_form']: .12e}",
             f"    relative           "
             f"{abs(result['ad_frozen'] - result['closed_form']) / abs(result['closed_form']):.3e}",
             ""]
    for tag in ("frozen", "full"):
        lines.append(f"  {tag} AD vs central FD")
        for h, fd in zip(result["steps"], result[f"fd_{tag}"]):
            rel = abs(fd - result[f"ad_{tag}"]) / abs(result[f"ad_{tag}"])
            lines.append(f"    h = {h:.0e}   FD {fd: .12e}   rel {rel:.3e}")
    lines += ["",
              f"  basis response       {result['basis_response']: .12e}"
              f"   ({abs(result['basis_response']) / abs(result['ad_full']):.3%}"
              " of the full derivative)"]
    return "\n".join(lines)


SI = (
    [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)],
    {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]},
    (4, 4, 4),
)
_A = 4.7443                      # monolayer h-BN, a = 2.511 Angstrom
HBN = (
    [(_A, 0.0, 0.0), (-_A / 2, _A * 3 ** 0.5 / 2, 0.0), (0.0, 0.0, 20.0)],
    {"B": [(0.0, 0.0, 0.0)], "N": [(1 / 3, 2 / 3, 0.0)]},
    (6, 6, 1),
)
CASES = [("Si generic", SI, (0.1, 0.2, 0.05)),
         ("hBN generic", HBN, (0.1, 0.2, 0.0))]


def _ground_state(workdir, label, structure):
    from elkpy.structure import Structure

    avec, species, ngridk = structure
    calculation = Structure(avec, species).get_calculation(
        f"{workdir}/{label.replace(' ', '_')}", xc="PW",
        ngridk=ngridk, rgkmax=7.0)
    calculation.ensure_ground_state()
    efermi = float((calculation.workdir / "EFERMI.OUT").read_text().split()[0])
    return calculation, efermi


def main(workdir="phase1_potential"):
    for label, structure, k in CASES:
        calculation, efermi = _ground_state(workdir, label.split()[0], structure)
        with calculation.eigenstate_session() as session:
            export = session.lapw_problem(k)
        nocc = ham.occupied_band_count(export["evalfv"], efermi)
        print(f"\n{'=' * 78}\n{label}   k={k}   nmat={export['nmatp']}"
              f"\n{'=' * 78}", flush=True)
        print(report(run(export, nocc=nocc)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "phase1_potential")
