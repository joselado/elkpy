"""Parse plot3d-family output (density/potential/ELF 3D plots -- tasks 33,
43, 53 -- share the exact same writer, src/plot3d.f90).

Format: one header line "nx ny nz : grid size" (the trailing text is a
literal Fortran format-string comment, not a separate line), then one line
per grid point: x y z (Cartesian Bohr) followed by 1-4 function values.
"""

import numpy as np


def parse_plot3d(path, nf=1):
    """Return (points, values): points shape (N,3) Cartesian Bohr, values
    shape (N,) if nf=1 else (N, nf)."""
    with open(path) as fh:
        fh.readline()  # header ("nx ny nz : grid size"), grid size not needed
        data = np.loadtxt(fh)
    points = data[:, :3]
    values = data[:, 3 : 3 + nf]
    if nf == 1:
        values = values[:, 0]
    return points, values


def parse_plot2d(path, nf=1):
    """Parse plot2d-family output (src/plot2d.f90 -- upstream tasks 62/162,
    elkpy task 9003).

    Format: one header line "n1 n2 : grid size", then one line per grid point:
    two in-plane Cartesian coordinates (Bohr, measured in the plotting
    parallelogram's own frame -- see plotpt2d.f90's `vppc`) followed by 1-4
    function values.

    Returns (points, values, grid): points shape (N,2), values shape (N,) if
    nf=1 else (N, nf), grid the (n1, n2) tuple from the header. The first
    plotting vector's index runs fastest (plotpt2d.f90's inner loop), so a
    column reshapes to an image as `values.reshape(n2, n1)`.
    """
    with open(path) as fh:
        header = fh.readline().split()
        grid = (int(header[0]), int(header[1]))
        data = np.loadtxt(fh)
    points = data[:, :2]
    values = data[:, 2 : 2 + nf]
    if nf == 1:
        values = values[:, 0]
    return points, values, grid
