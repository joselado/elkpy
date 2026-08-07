"""Parse GEOMETRY_OPT.OUT (src/geomopt.f90 / src/writegeom.f90): the
relaxed geometry after task 2/3, written in elk.in block syntax. writegeom
is called once per optimisation step, so the file holds one avec/atoms pair
per step -- the last occurrence is the final relaxed geometry.
"""

from collections import OrderedDict

from ..inputfile import read_blocks


def parse_last_geometry(geometry_opt_out_path):
    """Return (avec, species) for the final relaxed geometry, in the same
    shape Structure(avec, species) expects: avec as 3 tuples, species as an
    OrderedDict of symbol -> [(position, bfcmt), ...].

    writegeom always hardcodes scale/scale1/scale2/scale3 to 1.0 and bakes
    any scaling into avec directly (src/writegeom.f90), so those blocks are
    ignored here.
    """
    avec = None
    species = None
    for name, lines in read_blocks(geometry_opt_out_path):
        if name == "avec":
            avec = [tuple(float(x) for x in line) for line in lines]
        elif name == "atoms":
            species = _parse_atoms_block(lines)
    if avec is None or species is None:
        raise ValueError(f"no avec/atoms block found in {geometry_opt_out_path}")
    return avec, species


def _parse_atoms_block(lines):
    species = OrderedDict()
    idx = 0
    nspecies = int(lines[idx][0])
    idx += 1
    for _ in range(nspecies):
        spfname = lines[idx][0].strip("'\"")
        symbol = spfname.rsplit(".", 1)[0]
        idx += 1
        natoms = int(lines[idx][0])
        idx += 1
        positions = []
        for _ in range(natoms):
            values = [float(x) for x in lines[idx]]
            idx += 1
            positions.append((tuple(values[0:3]), tuple(values[3:6])))
        species[symbol] = positions
    return species
