"""Parse the "Forces :" section Elk appends to INFO.OUT when tforce is
.true. (src/writeforces.f90). Printed once, after the SCF loop ends -- not
per iteration.
"""

import re

import numpy as np

_TOTAL_FORCE_RE = re.compile(r"total force\s*:\s*(\S+)\s+(\S+)\s+(\S+)")


def parse_forces(info_out_path):
    """Return an (natoms, 3) array of total forces (Hartree/Bohr), in atom
    order as declared in the atoms block (species in order, atoms within
    each species in order -- matches src/writeforces.f90's loop order)."""
    text = open(info_out_path).read()
    idx = text.rfind("Forces :")
    if idx == -1:
        raise ValueError(f"no 'Forces :' section in {info_out_path} -- was tforce set?")
    section = text[idx:]
    forces = [tuple(float(x) for x in m.groups()) for m in _TOTAL_FORCE_RE.finditer(section)]
    if not forces:
        raise ValueError(f"'Forces :' section in {info_out_path} had no parseable force lines")
    return np.array(forces)
