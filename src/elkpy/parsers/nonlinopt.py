"""Parse the second-order (second-harmonic-generation) response files
written by task 125, src/nonlinopt.f90.

nonlinopt.f90 evaluates chi^{abc}(-2w; w, w), the second-order optical
susceptibility, in the length gauge following Sipe and Ghahramani, PRB 48,
11705 (1993) and Hughes and Sipe, PRB 53, 10751 (1996) (the two references
the subroutine itself names). It writes FOUR files per Cartesian triple
(a, b, c), all in the same two-block "real part, blank line, imaginary
part" layout src/dielectric.f90 uses for EPSILON_ij.OUT -- so the block
splitting is reused from parsers/dielectric.py rather than duplicated:

    CHI_II_2WWW_abc.OUT      chi_II   (interband)
    ETA_II_2WWW_abc.OUT      eta_II   (intraband/modulation)
    SIGMA_II_2WWW_abc.OUT    (i/2w) sigma_II
    CHI_2WWW_abc.OUT         the total, chi = chi_II + eta_II + i/2w sigma_II

The write statements are, in every case,

    write(50,'(2G18.10)') t1, dble(chi2w(iw))     ! nwplot lines
    write(50,*)                                   ! blank separator
    write(50,'(2G18.10)') t1, aimag(chi2w(iw))    ! nwplot lines

(src/nonlinopt.f90, the `if (mp_mpi) then` output section).

ENERGY GRID -- a real trap. Unlike task 121, nonlinopt builds its grid as

    t1 = wplot(2)/nwplot ;  w(iw) = t1*(iw-1)

i.e. it ALWAYS starts at zero and IGNORES wplot(1) entirely, and excludes
the upper endpoint. energies_from_wplot() below reproduces that exactly so
a caller can predict the abscissa without reading a file.

UNITS: omega in Hartree; chi in atomic units (the prefactor is
wkptnr/omega, i.e. per non-reduced k-point weight and per unit cell
volume, with the position-operator matrix elements r = p/(i*de) built from
PMAT.OUT).
"""

import numpy as np

from .dielectric import parse_two_blocks


def energies_from_wplot(wplot_max, nwplot):
    """The photon-energy grid task 125 uses, in Hartree.

    src/nonlinopt.f90: `t1=wplot(2)/dble(nwplot)` then
    `w(iw)=t1*dble(iw-1)` for iw = 1..nwplot -- zero-based, upper endpoint
    excluded, `wplot(1)` unused.
    """
    if nwplot < 2:
        raise ValueError(f"nwplot must be >= 2 (Elk's own check), got {nwplot}")
    step = float(wplot_max) / float(nwplot)
    return step * np.arange(nwplot, dtype=float)


def parse_chi2(path):
    """Return (energies, chi) for one CHI_2WWW_abc.OUT / CHI_II_2WWW_abc.OUT
    / ETA_II_2WWW_abc.OUT / SIGMA_II_2WWW_abc.OUT file.

    energies shape (nwplot,) in Hartree, chi shape (nwplot,) complex in
    atomic units -- real part first block, imaginary part second block,
    exactly as the write statements above emit them.
    """
    energies, real_part, imag_part = parse_two_blocks(path)
    return energies, real_part + 1j * imag_part
