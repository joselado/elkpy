"""Parse TDOS.OUT from task 10 (total density of states).

Verified against a real Elk 11.0.2 run (Si, examples/basic/Si + task 10):
two columns, energy (Hartree, relative to Fermi energy) and DOS
(states/Hartree/unit cell).
"""

import numpy as np


def parse_dos(tdos_out_path):
    data = np.loadtxt(tdos_out_path)
    energies, dos = data[:, 0], data[:, 1]
    return energies, dos
