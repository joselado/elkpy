r"""Item 0b(ii): $\kappa(O)$ for a REAL LAPW overlap, and the tolerance it sets.

Study §6 item 0b asks for the projector test to be repeated "with a real
Cholesky-reduced $S$, with a numeric criterion set at that size from the measured
$\kappa(S)$, since the tolerance scales as $\epsilon \kappa(S) \|H\|$ and $\kappa(S)$ is
unknown here".  Everything in `phase0b` uses a synthetic overlap with a *prescribed*
condition number, so the tolerance had no anchor.  This driver measures it on matrices
Elk actually built, through patch 0013's `LAPW` query (`docs/design.md` §33).

Three quantities per k-point:

* :math:`\kappa(O)=\lambda_{\max}/\lambda_{\min}` from a dense ``eigvalsh``.
* the Cholesky-diagonal estimate §8(b) proposes, :math:`\max_i L_{ii}^2/\min_i L_{ii}^2`,
  which is a provable *lower* bound (each :math:`L_{ii}^2` is a Schur-complement pivot,
  hence between the extreme eigenvalues) — the dangerous direction for a tolerance meant
  to bound from above.
* :math:`\|\tilde H\|` with :math:`\tilde H=L^{-1}HL^{-\dagger}`, since the reduced
  matrix is the one whose eigenproblem is actually differentiated, and it is
  consistently larger than :math:`\|H\|`.

Unlike the other Phase 0 drivers this one needs **no JAX at all** — it is NumPy, SciPy
and a real ``elk`` binary.  It lives here because it is the driver for a Phase 0 item;
the import of ``elkpy`` is the only place in ``elkjax`` that reaches into the sibling
package, and it is one-directional (nothing in ``elkpy`` imports ``elkjax``).

    PYTHONPATH=src taskset -c 0-3 python3 -m elkjax.phase0b_overlap <workdir>

Results and their five conclusions: ``docs/jax_port_phase0.md`` §0b(ii).
"""

import sys

import numpy as np
import scipy.linalg as sla

SI = (
    [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)],
    {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]},
    (4, 4, 4),
)
_A = 4.7443                      # monolayer h-BN, a = 2.511 Angstrom
HBN = (
    [(_A, 0.0, 0.0), (-_A / 2, _A * np.sqrt(3) / 2, 0.0), (0.0, 0.0, 30.0)],
    {"B": [(0.0, 0.0, 0.0)], "N": [(1 / 3, 2 / 3, 0.0)]},
    (4, 4, 1),
)
KPOINTS = [(0.1, 0.2, 0.05), (0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]
CASES = [("Si", SI, (7.0, 8.0, 9.0)), ("hBN", HBN, (7.0, 8.0))]


def measure(exported):
    """kappa(O), the Cholesky estimate, the two norms and the tolerance."""
    overlap, hamiltonian = exported["omat"], exported["hmat"]
    eigenvalues = np.linalg.eigvalsh(overlap)
    kappa = eigenvalues.max() / eigenvalues.min()
    factor = np.linalg.cholesky(overlap)
    pivots = np.abs(np.diag(factor)) ** 2
    # L^-1 H L^-H, taken as two triangular solves rather than an inverse
    reduced = sla.solve_triangular(
        factor,
        sla.solve_triangular(factor, hamiltonian, lower=True).conj().T,
        lower=True,
    ).conj().T
    norm_reduced = np.linalg.norm(reduced, 2)
    eigen = sla.eigh(hamiltonian, overlap, eigvals_only=True)
    return dict(
        kappa=kappa,
        cholesky_estimate=pivots.max() / pivots.min(),
        norm_h=np.linalg.norm(hamiltonian, 2),
        norm_reduced=norm_reduced,
        tolerance=np.finfo(float).eps * kappa * norm_reduced,
        # the export self-check: Elk's own evalfv came from eveqnfv BEFORE
        # elkpy_lapwexport disabled the tefvr real-matrix shortcut, so this is
        # a check on the export, not a tautology
        evalfv_error=np.abs(
            eigen[: exported["nstfv"]] - exported["evalfv"]).max(),
    )


def main(workdir):
    from elkpy.structure import Structure

    header = ("system", "rgkmax", "nmat", "k", "kappa(O)", "chol-est",
              "low by", "|H|", "|Ht|", "tol/Ha", "evalfv")
    print("{:5s} {:>6s} {:>5s} {:>15s} {:>9s} {:>8s} {:>7s} {:>6s} {:>6s} "
          "{:>9s} {:>8s}".format(*header), flush=True)
    for name, (avec, species, ngridk), cutoffs in CASES:
        for rgkmax in cutoffs:
            calculation = Structure(avec, species).get_calculation(
                f"{workdir}/kappa_{name}_{rgkmax}", xc="PW",
                ngridk=ngridk, rgkmax=rgkmax)
            with calculation.eigenstate_session() as session:
                for k in KPOINTS:
                    exported = session.lapw_problem(k)
                    m = measure(exported)
                    print("{:5s} {:6.1f} {:5d} {:>15s} {kappa:9.3g} "
                          "{cholesky_estimate:8.3g} {ratio:7.0f} {norm_h:6.2f} "
                          "{norm_reduced:6.2f} {tolerance:9.2e} "
                          "{evalfv_error:8.1e}".format(
                              name, rgkmax, exported["nmatp"], str(k),
                              ratio=m["kappa"] / m["cholesky_estimate"], **m),
                          flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "phase0b_overlap")
