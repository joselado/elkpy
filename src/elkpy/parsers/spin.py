"""Spin operators (sx, sy, sz) applicable to wavefunctions -- the spin-space
counterpart of parsers.quantum_geometry: needing no new Fortran, since every
number this consumes (evecsv) is already exposed by the task-9002
interactive session (session.py / patches/0003-eigenstate-session.patch),
which EIGENSTATES already returns for the OTHER reason documented in
docs/design.md #14 (inspecting a single diagonalization's own energies and
degeneracies).

See docs/design.md #17 and docs/physics.tex Part VI for the physics: why the
second-variational spinor basis makes the spin operator block-diagonal in
the first-variational spatial index, so S_a's matrix elements reduce to
plain inner products between evecsv's spin-up and spin-down blocks.
"""

import numpy as np


def compute_spin_operator(evecsv, nstfv, ist0, ist1):
    """S_x, S_y, S_z as Hermitian nst x nst matrices (hbar=1, so eigenvalues
    +-1/2 for a pure spin state) in the second-variational eigenbasis of one
    diagonalization, restricted to the contiguous 1-indexed band window
    [ist0, ist1].

    `evecsv`: (nstsv, nstsv) complex, evecsv[:, n] the n-th eigenvector, row
    index i = p + (ispn-1)*nstfv (Fortran 1-based convention, ispn in
    {1, 2}) -- eveqnsv.f90's own second-variational Hamiltonian layout, the
    same evecsv EigenstateSession.get_eigenstates() returns. `nstfv`: number
    of first-variational (spin-independent spatial) states; nstsv must equal
    2*nstfv (nspinor=2, i.e. a spin-polarized calculation -- see
    Calculation's spinpol/spinorb, which force this via Elk's own init0.f90)
    for a spin operator to be meaningful at all.

    Returns {"sx": .., "sy": .., "sz": ..}, each (nst, nst) complex
    Hermitian, nst = ist1 - ist0 + 1.
    """
    nstsv = evecsv.shape[0]
    if nstsv != 2 * nstfv:
        raise ValueError(
            f"evecsv has nstsv={nstsv} rows but nstfv={nstfv} implies nstsv=2*nstfv="
            f"{2 * nstfv} -- spin operators need a spin-polarized (nspinor=2) "
            f"diagonalization (Calculation(spinpol=True) or spinorb=True)"
        )
    window = slice(ist0 - 1, ist1)
    up = evecsv[:nstfv, window]
    down = evecsv[nstfv:, window]
    sz = 0.5 * (up.conj().T @ up - down.conj().T @ down)
    sx = 0.5 * (up.conj().T @ down + down.conj().T @ up)
    sy = 0.5j * (down.conj().T @ up - up.conj().T @ down)
    return {"sx": sx, "sy": sy, "sz": sz}
