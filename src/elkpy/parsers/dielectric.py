"""Parse EPSILON_ij.OUT / SIGMA_ij.OUT from task 121 (the dielectric
tensor and optical conductivity, src/dielectric.f90).

Format taken from dielectric.f90's own write statements (not the manual),
and verified against real output: `nwplot` lines of "omega value" pairs
in (2G18.10), then a `write(50,*)` blank line, then a second block of the
same length -- the REAL part first, the IMAGINARY part second, both on
the same non-negative energy grid in Hartree (dielectric.f90 builds it as
w1 = max(wplot(1), 0), so a negative wplot(1) is clipped to zero).

This is the same two-block layout src/moke.f90 writes KERR.OUT in, so the
block splitting itself lives in parsers/moke.py and is reused here.

Units: omega in Hartree; epsilon dimensionless; sigma in atomic units.
"""

import numpy as np



def parse_two_blocks(path):
    """Return (x, y_first, y_second) for a two-block Elk optical file: the
    shared abscissa and the two ordinate columns, in file order.

    Raises ValueError if the file doesn't hold exactly two blocks or if the
    two blocks disagree on the abscissa (they are written from the same
    `w` array, so a mismatch means a truncated/interleaved file).
    """
    blocks = []
    current = []
    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                if current:
                    blocks.append(current)
                    current = []
                continue
            x, y = stripped.split()
            current.append((float(x), float(y)))
    if current:
        blocks.append(current)
    if len(blocks) != 2:
        raise ValueError(
            f"expected 2 blocks (real then imaginary part) in {path}, got {len(blocks)}"
        )
    first = np.array(blocks[0])
    second = np.array(blocks[1])
    if first.shape != second.shape or not np.allclose(first[:, 0], second[:, 0]):
        raise ValueError(f"the two blocks in {path} are not on the same energy grid")
    return first[:, 0], first[:, 1], second[:, 1]


def parse_epsilon(path):
    """Return (energies, epsilon): energies shape (nw,) in Hartree,
    epsilon shape (nw,) complex and dimensionless.

    dielectric.f90 writes Re eps first, then Im eps -- with the delta_ij
    of the free-space term already added to the real part of a diagonal
    component. Im eps is the interband absorption spectrum.
    """
    energies, real_part, imag_part = parse_two_blocks(path)
    return energies, real_part + 1j * imag_part


def parse_sigma(path):
    """Return (energies, sigma): energies shape (nw,) in Hartree, sigma
    shape (nw,) complex in atomic units -- the optical conductivity, from
    which dielectric.f90 forms epsilon as
    eps_ij = delta_ij + 4 pi i sigma_ij / (omega + i swidth).
    """
    energies, real_part, imag_part = parse_two_blocks(path)
    return energies, real_part + 1j * imag_part
