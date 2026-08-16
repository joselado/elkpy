"""Parse KERR.OUT from task 122 (magneto-optic Kerr effect, src/moke.f90).

Format taken from moke.f90's own write statements (not the manual): two
blocks of `nwplot` "omega value" lines, separated by a line of blanks --
first the real part of the complex Kerr angle, then the imaginary part,
both in DEGREES (moke.f90 multiplies by 180/pi) and both on the same
non-negative energy grid in Hartree (src/dielectric.f90 builds the grid as
w1=max(wplot(1),0), so a negative wplot(1) is clipped to zero).

Same blank-line-separated block layout as SIGMA_ij.OUT/EPSILON_ij.OUT,
which dielectric.f90 writes with the identical (2G18.10) pairs.
"""

import numpy as np

# the two-block layout originates in dielectric.f90, which moke.f90 calls
# internally -- so the reader lives there and MOKE depends on it, not the
# reverse
from .dielectric import parse_two_blocks


def parse_kerr(path):
    """Return (energies, kerr): energies shape (nw,) in Hartree, kerr shape
    (nw,) complex in degrees.

    The real part is the Kerr rotation angle theta_K (the rotation of the
    polarization plane of light reflected off the magnetized surface), the
    imaginary part the Kerr ellipticity.
    """
    energies, real_part, imag_part = parse_two_blocks(path)
    return energies, real_part + 1j * imag_part
