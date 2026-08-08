"""Parse GEOMETRY_OPT.OUT (src/geomopt.f90 / src/writegeom.f90): the
relaxed geometry after task 2/3, written in elk.in block syntax. writegeom
is called once per optimisation step, so the file holds one avec/atoms pair
per step -- the last occurrence is the final relaxed geometry.
"""

from collections import OrderedDict

from ..inputfile import read_blocks


def parse_last_geometry(geometry_opt_out_path, species_order=None):
    """Return (avec, species) for the final relaxed geometry, in the same
    shape Structure(avec, species) expects: avec as 3 tuples, species as an
    OrderedDict of symbol -> [(position, bfcmt), ...].

    `species_order`: the element symbols in the same order they were
    declared in the original elk.in atoms block (e.g.
    list(structure.species.keys())). Elk echoes species back into
    GEOMETRY_OPT.OUT in that same declared order (src/writegeom.f90 loops
    `do is=1,nspecies` over the original array), but only the species
    *filename*, not the element -- so if the caller used a
    Structure.species_files override (species_filename(symbol) !=
    "{symbol}.in"), the filename alone doesn't recover the real element.
    Pass species_order to sidestep that; without it, the symbol is guessed
    from the filename stem, which is only correct for the unoverridden
    "{symbol}.in" convention.

    writegeom always hardcodes scale/scale1/scale2/scale3 to 1.0 and bakes
    any scaling into avec directly (src/writegeom.f90), so those blocks are
    ignored here -- the returned avec is already fully scaled.
    """
    avec = None
    species = None
    for name, lines in read_blocks(geometry_opt_out_path):
        if name == "avec":
            avec = [tuple(float(x) for x in line) for line in lines]
        elif name == "atoms":
            species = _parse_atoms_block(lines, species_order)
    if avec is None or species is None:
        raise ValueError(f"no avec/atoms block found in {geometry_opt_out_path}")
    return avec, species


def _parse_atoms_block(lines, species_order=None):
    species = OrderedDict()
    idx = 0
    nspecies = int(lines[idx][0])
    idx += 1
    if species_order is not None and len(species_order) != nspecies:
        raise ValueError(
            f"species_order has {len(species_order)} entries but the file declares "
            f"{nspecies} species"
        )
    for i in range(nspecies):
        spfname = lines[idx][0].strip("'\"")
        symbol = species_order[i] if species_order is not None else spfname.rsplit(".", 1)[0]
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
