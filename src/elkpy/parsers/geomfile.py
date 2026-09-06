"""Parsers for the crystal-geometry files task 190 writes
(src/geomplot.f90): ``crystal.xsf`` (XCrySDen) and ``crystal.ascii``
(V_Sim). Task 420/421 additionally writes ``crystal.axsf``, the animated
XSF of a molecular-dynamics trajectory (src/writeaxsf.f90).

These carry no new physics -- they are Elk's own view of the structure it
actually ran, *after* symmetry analysis. That makes them worth reading
back rather than assuming: with the default ``tshift=.true.`` Elk shifts
the atomic basis so that, for a centrosymmetric crystal, the inversion
centre sits at the origin (``findsymcrys.f90``). Diamond silicon entered
as (0,0,0) and (1/4,1/4,1/4) therefore comes back as +-(3/8,3/8,3/8)
(measured against a real binary; the diamond lattice has several
inversion centres and which one Elk lands on is its own choice): the
same crystal, a different origin. Comparing these positions to the input
is a check on that shift, not a round-trip identity, and only the lattice
vectors are guaranteed unchanged.

The two files do NOT share units: ``geomplot`` multiplies by ``br_ang``
(the Bohr radius in Angstrom) on the XSF path only, so ``crystal.xsf`` is
in Angstrom while ``crystal.ascii`` -- like everything else in elkpy --
is in Bohr.
"""

import numpy as np

# src/modmain.f90: br_ang, the Bohr radius in Angstrom, as used by geomplot
BOHR_IN_ANGSTROM = 0.52917721092


def _lines(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [line.rstrip("\n") for line in fh]


def parse_xsf(xsf_path):
    """Parse an XCrySDen ``crystal.xsf`` written by src/geomplot.f90::

        write(50,'("PRIMVEC")')
        write(50,'(3G18.10)') avec(:,1)*br_ang     ! and (:,2), (:,3)
        write(50,'("PRIMCOORD")')
        write(50,'(2I8)') natmtot,1
        write(50,'(A,3G18.10)') trim(spsymb(is)),atposc(:,ia,is)*br_ang

    ``avec(:,j)`` is lattice vector j, so the three PRIMVEC lines are the
    lattice vectors as rows -- the same layout as elkpy's
    ``Structure.avec``, but in Angstrom.

    Returns {"avec": (3,3) array (Angstrom), "symbols": [str],
    "positions": (natoms,3) array (Cartesian Angstrom)}.
    """
    lines = _lines(xsf_path)
    avec, symbols, positions = None, [], []
    i = 0
    while i < len(lines):
        keyword = lines[i].strip()
        if keyword == "PRIMVEC":
            avec = np.array([[float(x) for x in lines[i + 1 + r].split()] for r in range(3)])
            i += 4
            continue
        if keyword == "PRIMCOORD":
            natoms = int(lines[i + 1].split()[0])
            for row in lines[i + 2 : i + 2 + natoms]:
                fields = row.split()
                symbols.append(fields[0])
                positions.append([float(x) for x in fields[1:4]])
            i += 2 + natoms
            continue
        i += 1
    if avec is None or not symbols:
        raise ValueError(f"{xsf_path}: missing PRIMVEC or PRIMCOORD section")
    return {"avec": avec, "symbols": symbols, "positions": np.array(positions)}


def parse_axsf(axsf_path):
    """Parse an animated XSF (``crystal.axsf``, src/writeaxsf.f90) into a
    list of per-frame dicts of the same shape ``parse_xsf`` returns, plus
    a ``"frame"`` index. Frames are delimited by ``PRIMCOORD <n>`` lines;
    the single leading ``PRIMVEC`` block applies to all of them.
    """
    lines = _lines(axsf_path)
    avec = None
    frames = []
    i = 0
    while i < len(lines):
        fields = lines[i].split()
        if fields and fields[0] == "PRIMVEC":
            avec = np.array([[float(x) for x in lines[i + 1 + r].split()] for r in range(3)])
            i += 4
            continue
        if fields and fields[0] == "PRIMCOORD":
            frame = int(fields[1]) if len(fields) > 1 else len(frames) + 1
            natoms = int(lines[i + 1].split()[0])
            symbols, positions = [], []
            for row in lines[i + 2 : i + 2 + natoms]:
                parts = row.split()
                symbols.append(parts[0])
                positions.append([float(x) for x in parts[1:4]])
            frames.append(
                {
                    "frame": frame,
                    "avec": avec,
                    "symbols": symbols,
                    "positions": np.array(positions),
                }
            )
            i += 2 + natoms
            continue
        i += 1
    if not frames:
        raise ValueError(f"{axsf_path}: no PRIMCOORD frames found")
    return frames


def parse_vsim_ascii(ascii_path):
    """Parse a V_Sim ``crystal.ascii`` written by src/geomplot.f90.

    The format is a rotated frame: geomplot builds an orthonormal triad
    (v1 along a1, v3 = a1 x a2 normalised, v2 = v3 x v1) and writes the
    six lower-triangular components of the lattice in that frame::

        write(50,'(3G18.10)') dxx,dyx,dyy
        write(50,'(3G18.10)') dzx,dzy,dzz
        write(50,'(3G18.10," ",A)') v4,trim(spsymb(is))

    with dxx = a1.v1 and so on. Lengths here are in Bohr -- unlike the XSF
    file, geomplot applies no ``br_ang`` conversion on this path.

    Returns {"lower_triangle": (dxx, dyx, dyy, dzx, dzy, dzz),
    "avec": (3,3) array in the rotated frame, "symbols": [str],
    "positions": (natoms,3) array}.
    """
    rows = [line for line in _lines(ascii_path) if line.strip()]
    dxx, dyx, dyy = (float(x) for x in rows[0].split())
    dzx, dzy, dzz = (float(x) for x in rows[1].split())
    symbols, positions = [], []
    for row in rows[2:]:
        fields = row.split()
        if len(fields) < 4:
            continue
        positions.append([float(x) for x in fields[0:3]])
        symbols.append(fields[3])
    avec = np.array([[dxx, 0.0, 0.0], [dyx, dyy, 0.0], [dzx, dzy, dzz]])
    return {
        "lower_triangle": (dxx, dyx, dyy, dzx, dzy, dzz),
        "avec": avec,
        "symbols": symbols,
        "positions": np.array(positions),
    }
