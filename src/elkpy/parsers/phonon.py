"""Parsers for Elk's phonon / Born-effective-charge output files.

One module for the whole "lattice dynamics" family that isn't already
covered elsewhere:

- ``BEC_Sss_Aaaa_Pp.OUT``     -- static Born effective charges (task 208,
                                src/bornechg.f90), and the DYNAMICAL ones
                                (task 478, src/bornecdyn.f90), which
                                unfortunately share the same filenames.
- ``PHONON.OUT``              -- phonon frequencies and eigenvectors at an
                                explicit list of q-points (task 230,
                                src/writephn.f90).
- ``GAMMAQ.OUT``/``LAMBDAQ.OUT`` -- phonon linewidths and mode-resolved
                                electron-phonon coupling constants on the
                                q-mesh (tasks 240/241, written by
                                src/writegamma.f90 / src/writelambda.f90).

Two families already have parsers and are deliberately NOT duplicated here:
``PHDISP.OUT``/``PHLWIDTH.OUT`` share BAND.OUT's blank-line-separated block
layout exactly (``parsers.band.parse_bands``), and ``PHDOS.OUT``/
``TDOS_EPH.OUT`` share TDOS.OUT's two-column layout (``parsers.dos``).

Every format below is transcribed from the cited ``write`` statement in
vendor/elk/src/, not from the manual. ``parse_born_charge_row`` is
additionally binary-verified against a real Elk 11.0.2 task-208 run on
bulk Si (see tests/test_calculation_phonons.py).
"""

import numpy as np


def split_blocks(path, ncols=None):
    """Split a blank-line-separated Elk text file into a list of float
    arrays, one per block.

    Elk writes ``write(50,*)`` (an empty record) between blocks. Blocks are
    allowed to have DIFFERENT lengths -- ELIASHBERG_IA.OUT genuinely does,
    since its Matsubara cut-off changes with temperature -- so this returns
    a list rather than one rectangular array. That is why it does not reuse
    ``parsers.dielectric.parse_two_blocks``, which requires exactly two
    blocks on a common abscissa.

    `ncols`, if given, is checked against every row.
    """
    blocks = []
    current = []
    with open(path) as fh:
        for line in fh:
            fields = line.split()
            if not fields:
                if current:
                    blocks.append(current)
                    current = []
                continue
            row = [float(x) for x in fields]
            if ncols is not None and len(row) != ncols:
                raise ValueError(
                    f"expected {ncols} columns in {path}, got {len(row)}: {line!r}"
                )
            current.append(row)
    if current:
        blocks.append(current)
    return [np.array(b) for b in blocks]


def born_charge_filename(ispecies, iatom, ipolar):
    """The BEC file name for species `ispecies`, atom `iatom`, displacement
    direction `ipolar` -- all 1-BASED, matching Elk's own indices.

    src/becfext.f90:
        write(fext,'("_S",I2.2,"_A",I3.3,"_P",I1,".OUT")') is,ia,ip
    and src/bectask.f90 prepends 'BEC'.
    """
    return f"BEC_S{ispecies:02d}_A{iatom:03d}_P{ipolar:d}.OUT"


def parse_born_charge_row(path):
    """Return one row of the Born effective charge tensor, shape (3,).

    src/bornechg.f90's write loop is

        do ip=1,3
          write(80,'(G18.10," : ip = ",I4)') becc(ip),ip
        end do

    i.e. three lines, each "value : ip = N". The file's own ``_P{ip}``
    suffix is the DISPLACEMENT direction; the three values inside are the
    resulting polarisation components. Elk has already added the core plus
    nuclear charge (``chgcr(isph)+spzn(isph)``) to the element whose
    polarisation index equals the displacement index, so what comes back is
    the full charge, not just the electronic part.
    """
    values = []
    with open(path) as fh:
        for line in fh:
            fields = line.split()
            if not fields:
                continue
            values.append(float(fields[0]))
    if len(values) != 3:
        raise ValueError(f"expected 3 Born-charge components in {path}, got {len(values)}")
    return np.array(values)


def parse_born_charges(directory, species_counts):
    """Assemble the full 3x3 Born effective charge tensor of every atom.

    `species_counts` is the number of atoms per species, in the same order
    the `atoms` block was written (i.e. ``Structure.species``' key order),
    so the (species, atom) indices here are Elk's own.

    Returns ``{(ispecies, iatom): Z}`` with 1-based keys and Z shape (3, 3),
    indexed ``Z[displacement, polarisation]``: ``Z[i][j]`` is the change in
    the j-th Cartesian polarisation component per unit displacement along
    Cartesian direction i. That is the TRANSPOSE of the common textbook
    convention Z*_{ab} = dP_a/du_b, and it is Elk's: src/readbec.f90 fills
    ``bec(ip,jp,ias)`` with ip the file's ``_P`` suffix (the displaced
    direction) and jp the line index within the file.
    """
    charges = {}
    for ispecies, natoms in enumerate(species_counts, start=1):
        for iatom in range(1, natoms + 1):
            rows = [
                parse_born_charge_row(directory / born_charge_filename(ispecies, iatom, ip))
                for ip in (1, 2, 3)
            ]
            charges[(ispecies, iatom)] = np.array(rows)
    return charges


def parse_born_charge_dynamical(path):
    """Return (frequencies, Z) for one dynamical Born effective charge row
    (task 478, src/bornecdyn.f90): frequencies shape (nw,) in Hartree, Z
    shape (nw, 3) complex.

    bornecdyn.f90 writes SIX blank-line-separated blocks into the same
    ``BEC_Sss_Aaaa_Pp.OUT`` name task 208 uses -- for each polarisation
    component i = 1, 2, 3, first the real part then the imaginary part:

        do i=1,3
          do iw=1,nwplot
            write(80,'(2G18.10)') w(iw),dble(becw(iw,i))+t2
          end do
          write(80,*)
          do iw=1,nwplot
            write(80,'(2G18.10)') w(iw),aimag(becw(iw,i))
          end do
          write(80,*)
        end do

    (``t2`` is the static muffin-tin plus nuclear charge, added only to the
    component matching the displacement direction.) The column index of the
    returned array is that polarisation component i, so the same
    [displacement, polarisation] convention as the static tensor holds:
    the file's ``_P`` suffix is the displacement.

    NOTE the filename collision: a task-478 output directory looks exactly
    like a task-208 one to ``readbec``/``tphnat``, which would then read the
    first three numbers of a frequency grid as a charge tensor. Keep the two
    in separate directories (elkpy does).
    """
    blocks = split_blocks(path, ncols=2)
    if len(blocks) != 6:
        raise ValueError(
            f"expected 6 blocks (Re/Im for each of 3 polarisations) in {path}, "
            f"got {len(blocks)}"
        )
    frequencies = blocks[0][:, 0]
    for b in blocks[1:]:
        if b.shape != blocks[0].shape or not np.allclose(b[:, 0], frequencies):
            raise ValueError(f"blocks in {path} are not on a common frequency grid")
    z = np.stack([blocks[2 * i][:, 1] + 1j * blocks[2 * i + 1][:, 1] for i in range(3)], axis=1)
    return frequencies, z


def parse_phonon_modes(path):
    """Parse PHONON.OUT (task 230, src/writephn.f90).

    Returns a list with one dict per requested q-point:

        {"index": int,                  # Elk's 1-based q index
         "qpoint": (3,) float,          # lattice coordinates, as requested
         "frequencies": (nbph,) float,  # Hartree
         "eigenvectors": (nbph, nbph) complex}

    ``eigenvectors[j]`` is mode j's vector, and its component ``i`` runs
    over (species, atom, Cartesian direction) in the `atoms` block order,
    fastest in the Cartesian index -- writephn.f90's own loop nesting:

        do j=1,nbph                                     ! mode
          write(50,'(I6,G18.10," : mode, frequency")') j,wq(j)
          i=0
          do is=1,nspecies; do ia=1,natoms(is); do ip=1,3
            i=i+1
            write(50,'(3I4,2G18.10)') is,ia,ip,ev(i,j)

    Two physics conventions come straight from src/dynev.f90, which
    produced ``wq``/``ev``, and are worth stating because neither is
    recoverable from the file:

    - ``ev`` diagonalises the MASS-WEIGHTED dynamical matrix
      D_ij / sqrt(M_i M_j), so its columns are orthonormal polarisation
      vectors, not displacements; the physical displacement pattern is
      ``eigenvectors[j][i] / sqrt(M_i)``.
    - a frequency is stored as ``sign(sqrt(|w2|), w2)``, so an unstable
      (imaginary) mode comes back as a NEGATIVE real number, not a complex
      one.
    """
    modes = []
    current = None
    with open(path) as fh:
        for line in fh:
            fields = line.split()
            if not fields:
                continue
            if "q-point, vqlwrt" in line:
                current = {
                    "index": int(fields[0]),
                    "qpoint": tuple(float(x) for x in fields[1:4]),
                    "frequencies": [],
                    "eigenvectors": [],
                }
                modes.append(current)
            elif "mode, frequency" in line:
                current["frequencies"].append(float(fields[1]))
                current["eigenvectors"].append([])
            else:
                # "is ia ip Re(ev) Im(ev)" -- the trailing comment on the
                # first such line of each mode is dropped by the split
                # below, since the numbers come first.
                current["eigenvectors"][-1].append(float(fields[3]) + 1j * float(fields[4]))
    for m in modes:
        m["frequencies"] = np.array(m["frequencies"])
        m["eigenvectors"] = np.array(m["eigenvectors"])
    return modes


def parse_qpoint_table(path):
    """Parse GAMMAQ.OUT or LAMBDAQ.OUT -- src/writegamma.f90 and
    src/writelambda.f90 write byte-identical layouts, only the meaning of
    the per-mode number differs:

        (blank)
        '(I4," : total number of atoms")'  natmtot
        '(I6," : number of q-points")'     nqpt
        (blank)
        for each q-point:
          '(I6," : q-point")'                                  iq
          '(3G18.10," : q-vector (lattice coordinates)")'      vql
          '(3G18.10," : q-vector (Cartesian coordinates)")'    vqc
          nbph lines of '(I4,G18.10)'                          imode, value
          (blank)

    Returns {"natoms": int, "qpoints": (nq, 3), "qpoints_cartesian":
    (nq, 3), "values": (nq, nbph)}. `qpoints` are lattice coordinates,
    `qpoints_cartesian` are in inverse Bohr.

    GAMMAQ's values are phonon linewidths gamma_{q,nu} in Hartree;
    LAMBDAQ's are the dimensionless mode couplings Elk forms as
    gamma / (pi N(E_F) omega^2) (writelambda.f90).
    """
    natoms = None
    nqpt = None
    qpoints = []
    qpoints_cartesian = []
    values = []
    with open(path) as fh:
        for line in fh:
            fields = line.split()
            if not fields:
                continue
            if "total number of atoms" in line:
                natoms = int(fields[0])
            elif "number of q-points" in line:
                nqpt = int(fields[0])
            elif ": q-point" in line:
                qpoints.append(None)
                qpoints_cartesian.append(None)
                values.append([])
            elif "lattice coordinates" in line:
                qpoints[-1] = tuple(float(x) for x in fields[:3])
            elif "Cartesian coordinates" in line:
                qpoints_cartesian[-1] = tuple(float(x) for x in fields[:3])
            else:
                values[-1].append(float(fields[1]))
    if nqpt is not None and len(qpoints) != nqpt:
        raise ValueError(f"{path} declares {nqpt} q-points but holds {len(qpoints)}")
    return {
        "natoms": natoms,
        "qpoints": np.array(qpoints),
        "qpoints_cartesian": np.array(qpoints_cartesian),
        "values": np.array(values),
    }


def parse_face_histogram(path):
    """Parse FACEEH.OUT (task 280, src/ephdos.f90): the fermionic anomalous
    correlation entropy of each Bogoliubov state against its energy.

        write(50,'(5G18.10)') evaluv(ist,ik), t1, v

    with t1 = -[v ln v + (1-v) ln(1-v)] the entropy of the state's V-norm
    and v the k-vector in Cartesian coordinates, mapped into the first
    Brillouin zone. Rows whose entropy is below 1e-4 are not written at
    all, so the file is sparse by construction.

    Returns (energies (n,), entropy (n,), kpoints (n, 3)).
    """
    data = np.loadtxt(path, ndmin=2)
    if data.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros((0, 3))
    return data[:, 0], data[:, 1], data[:, 2:5]
