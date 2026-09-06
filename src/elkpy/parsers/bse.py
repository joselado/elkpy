"""Parse the Bethe-Salpeter output of tasks 185/186/187
(src/writehmlbse.f90, src/writeevbse.f90, src/dielectric_bse.f90).

The BSE chain in Elk is strictly ordered and every step reads the previous
step's file (established from the Fortran, not the manual):

    120  writepmat     -> PMAT.OUT     (momentum matrix elements)
    180  writeepsinv   -> EPSINV.OUT   (inverse RPA dielectric matrix,
                                        read back by src/hmldbsek.f90:148
                                        via getcfgq)
    185  writehmlbse   -> HMLBSE.OUT   (the BSE Hamiltonian; its DIRECT
                                        term hmldbse needs EPSINV.OUT, and
                                        `hdbse` defaults to .true.)
    186  writeevbse    -> EVBSE.OUT, EIGVAL_BSE.OUT  (diagonalisation)
    187  dielectric_bse-> EPSILON_BSE_ij.OUT         (needs EVBSE.OUT and
                                                      PMAT.OUT)

Only EIGVAL_BSE.OUT and EPSILON_BSE_ij.OUT are formatted; HMLBSE.OUT and
EVBSE.OUT are sequential UNFORMATTED Fortran files (record markers, and
their record contents depend on nmbse), so they are treated as opaque
intermediates here.

Matrix size: nmbse = nvbse*ncbse*nkptnr, doubled when `bsefull` is true
(src/genidxbse.f90). The diagonalisation is dense, so cost grows as
nmbse^3 -- this is the expensive member of the family.
"""

import numpy as np


def parse_bse_eigenvalues(path):
    """Return the BSE excitation energies from EIGVAL_BSE.OUT (Hartree).

    Format, from src/writeevbse.f90's write statements:

        write(50,'(I6," : nmbse")') nmbse
        ! then, if bsefull (the full non-Hermitian matrix):
        write(50,'(I6,2G18.10)') a, dble(w(a)), aimag(w(a))
        ! otherwise (the Hermitian resonant block only):
        write(50,'(I6,G18.10)') a, evalbse(a)

    Returns a complex array of length nmbse either way, so a caller does
    not have to branch on `bsefull`: the Hermitian case simply has zero
    imaginary part. The branch is detected from the column count of the
    first data row, not from an input flag.

    Physically, each eigenvalue is the energy of one correlated
    electron-hole excitation (an exciton when it sits below the
    independent-particle absorption edge); the difference from the bare
    transition energies e_c - e_v is the excitonic binding.
    """
    values = []
    header = None
    with open(path) as fh:
        for line in fh:
            tokens = line.split()
            if not tokens:
                continue
            if header is None:
                # "  nmbse : nmbse"
                header = int(tokens[0])
                continue
            if len(tokens) == 2:
                values.append(complex(float(tokens[1]), 0.0))
            elif len(tokens) == 3:
                values.append(complex(float(tokens[1]), float(tokens[2])))
            else:
                raise ValueError(
                    f"unexpected column count {len(tokens)} in {path}; "
                    f"src/writeevbse.f90 writes 2 columns (Hermitian) or 3 (bsefull)"
                )
    if header is None:
        raise ValueError(f"{path} is empty; expected an 'I6 : nmbse' header")
    if len(values) != header:
        raise ValueError(
            f"{path} declares nmbse={header} but holds {len(values)} eigenvalues"
        )
    return np.array(values, dtype=complex)


def parse_epsilon_bse(path):
    """Return (energies, epsilon) for one EPSILON_BSE_ij.OUT file.

    src/dielectric_bse.f90 writes the same two-block layout as
    src/dielectric.f90 -- real part, blank line, imaginary part, both
    `(2G18.10)` -- so the splitting is reused from parsers/dielectric.py.

    NOTE the energy grid follows task 125's convention, NOT task 121's:
    `t1=wplot(2)/dble(nwplot)`, `w(iw)=t1*dble(iw-1)`, i.e. it starts at
    zero and ignores wplot(1).

    Units: omega in Hartree, epsilon dimensionless. Im eps is the
    excitonic absorption spectrum -- the electron-hole interaction shifts
    oscillator strength to lower energy relative to task 121's
    independent-particle Im eps.
    """
    from .dielectric import parse_two_blocks

    energies, real_part, imag_part = parse_two_blocks(path)
    return energies, real_part + 1j * imag_part
