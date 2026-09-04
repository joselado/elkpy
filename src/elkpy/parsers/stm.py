"""Parse the spin-polarised STM task's output (elkpy tasks 9003/9004,
src/elkpy_stm.f90 -- docs/design.md #30).

The image itself is written by Elk's own generic plot2d/plot3d writers, so
`volumetric.parse_plot2d`/`parse_plot3d` read it; the only STM-specific file
is ELKPY_STMDOS.OUT, the cell integrals of the two plotted fields.
"""

import numpy as np


def parse_stm_dos(path):
    """Return a dict with the cell integrals and sampling parameters written
    by elkpy_stm.f90.

    In differential-conductance mode, `dos` is the total density of states at
    the sampling energy in states/Hartree/unit cell -- the same quantity
    src/occupy.f90 writes to FERMIDOS.OUT at zero bias -- and `spin_dos` its
    projection onto the tip direction (the spin DOS m.e, which is the total
    moment per Hartree, not per state).

    Keys: dos, spin_dos, energy, efermi, direction.
    """
    values = []
    with open(path) as fh:
        for line in fh:
            values.append(line.split(":")[0].split())
    return {
        "dos": float(values[0][0]),
        "spin_dos": float(values[1][0]),
        "energy": float(values[2][0]),
        "efermi": float(values[3][0]),
        "direction": np.array([float(x) for x in values[4]]),
    }
