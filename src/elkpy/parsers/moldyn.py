"""Parsers for the Born-Oppenheimer molecular-dynamics output of tasks
420/421 (src/moldyn.f90).

``moldyn`` runs the electrons to self-consistency at fixed nuclear
positions, takes the Hellmann-Feynman forces, and integrates the classical
nuclear equation of motion one step (``src/atptstep.f90``) -- the adiabatic
(Born-Oppenheimer) approximation, with no electronic dynamics at all
(that is tasks 460+). Times are in Hartree atomic units, hbar/E_h =
0.0241888 fs; forces in Hartree/Bohr; displacements in lattice
(ATDISPL_TD.OUT) or Cartesian Bohr (ATDISPC_TD.OUT) coordinates.

Every file here is opened by Elk with ``position='APPEND'`` and written
once per force step, so each is a *concatenation of blocks* rather than a
table -- a restart (task 421) appends to whatever is already there. Two
shapes occur:

- scalar-per-step, one line each: TOTENERGY_TD.OUT
  (``write(50,'(2G18.10)') times(itimes),engytot``), FORCEMAX_TD.OUT,
  MOMENT_TD.OUT, MOMENTM_TD.OUT, MOMENTIR_TD.OUT.
- per-atom blocks: ATDISPL_TD.OUT / ATDISPC_TD.OUT / FORCETOT_TD.OUT,
  each block opening with ``write(50,'(I8,G18.10)') itimes,times(itimes)``
  and followed by ``natmtot`` lines ``write(50,'(2I4,3G18.10)') is,ia,v``.

``ATDVC.OUT`` is the restart file rather than a trajectory: one line per
atom, ``write(50,'(2I4,6G18.10)') is,ia,atdvc(:,:,ia,is)`` -- three
displacement components followed by three velocity components, Cartesian.
"""

import numpy as np


def _lines(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [line.rstrip("\n") for line in fh if line.strip()]


def parse_scalar_series(path):
    """Parse a one-line-per-step MD file into an (nsteps, ncols) array.

    Covers TOTENERGY_TD.OUT and FORCEMAX_TD.OUT (2 columns: time, value),
    MOMENT_TD.OUT (1 + ndmag), MOMENTM_TD.OUT (2) and MOMENTIR_TD.OUT
    (1 + ndmag). Column 0 is always the time in atomic units.
    """
    rows = [[float(x) for x in line.split()] for line in _lines(path)]
    if not rows:
        raise ValueError(f"no data rows in {path}")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError(f"ragged rows in {path}")
    return np.array(rows)


def parse_atom_series(path):
    """Parse a per-atom-block MD file (ATDISPL_TD.OUT, ATDISPC_TD.OUT,
    FORCETOT_TD.OUT, MOMENTMT_TD.OUT).

    Returns {"step": (nsteps,) int, "time": (nsteps,), "values":
    (nsteps, natoms, ncomp), "species": (natoms,) int, "atom": (natoms,)
    int}.

    Block headers are ``(I8,G18.10)`` -> two fields (MOMENTMT_TD.OUT's is
    ``(G18.10)`` -> one), per-atom rows are ``(2I4,3G18.10)`` -> five, so
    the two are told apart by field count: a header never has three or
    more. ``ncomp`` is 3 for the displacement/force files but ``ndmag``
    for MOMENTMT_TD.OUT, which writes only ``mommt(1:ndmag,ias)`` -- one
    column for a collinear run -- hence the width is taken from the file
    rather than assumed. Atom order within a block is Elk's own (species
    in input order, atoms within a species in input order).
    """
    steps, times, blocks = [], [], []
    species, atoms = [], []
    current = None
    for line in _lines(path):
        fields = line.split()
        if len(fields) >= 3:
            if current is None:
                raise ValueError(f"{path}: atom row before any step header")
            if len(blocks) == 0:
                species.append(int(fields[0]))
                atoms.append(int(fields[1]))
            current.append([float(x) for x in fields[2:]])
        elif len(fields) in (1, 2):
            if current is not None:
                blocks.append(current)
            current = []
            steps.append(int(fields[0]) if len(fields) == 2 else len(steps) + 1)
            times.append(float(fields[-1]))
        else:
            raise ValueError(f"{path}: unexpected row with {len(fields)} fields: {line!r}")
    if current is not None:
        blocks.append(current)
    if not blocks:
        raise ValueError(f"no step blocks found in {path}")
    natoms = len(blocks[0])
    if any(len(b) != natoms for b in blocks):
        raise ValueError(f"{path}: steps have differing atom counts")
    return {
        "step": np.array(steps, dtype=int),
        "time": np.array(times),
        "values": np.array(blocks),
        "species": np.array(species, dtype=int),
        "atom": np.array(atoms, dtype=int),
    }


def parse_atdvc(atdvc_out_path):
    """Parse ATDVC.OUT, the MD restart state (src/writeatdvc.f90).

    ``write(50,'(2I4,6G18.10)') is,ia,atdvc(:,:,ia,is)`` -- Fortran array
    order runs the first index fastest, and ``atdvc(1:3,0:1,ia,is)`` holds
    the displacement at index 0 and the velocity at index 1, so the six
    numbers are (dx,dy,dz,vx,vy,vz) in Cartesian atomic units.

    Returns {"species": (n,), "atom": (n,), "displacement": (n,3),
    "velocity": (n,3)}.
    """
    species, atoms, disp, vel = [], [], [], []
    for line in _lines(atdvc_out_path):
        fields = line.split()
        if len(fields) != 8:
            raise ValueError(
                f"{atdvc_out_path}: expected 8 fields per atom, got {len(fields)}: {line!r}"
            )
        species.append(int(fields[0]))
        atoms.append(int(fields[1]))
        values = [float(x) for x in fields[2:]]
        disp.append(values[0:3])
        vel.append(values[3:6])
    if not species:
        raise ValueError(f"no atom rows in {atdvc_out_path}")
    return {
        "species": np.array(species, dtype=int),
        "atom": np.array(atoms, dtype=int),
        "displacement": np.array(disp),
        "velocity": np.array(vel),
    }


def parse_timestep(timestep_out_path):
    """Parse TIMESTEP.OUT (src/writetimes.f90): the single line
    ``write(50,'(I8,G18.10)') itimes,times(itimes)`` recording where a
    molecular-dynamics run stopped, which task 421 reads back to restart.

    Returns (step, time).
    """
    fields = _lines(timestep_out_path)[0].split()
    return int(fields[0]), float(fields[1])
