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
