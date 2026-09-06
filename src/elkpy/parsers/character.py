"""Parsers for the atom/l/m/spin-resolved *character* output: the band
character files of tasks 21-24 (``src/bandstr.f90``) and the partial /
interstitial density of states of task 10 (``src/dos.f90``).

Both come from the same machinery -- ``gendmatk``'s muffin-tin density
matrix, the same expansion ``docs/design.md`` sections 16/18/19 reuse for
the atom/orbital/angular-momentum operators -- so they live together here.

BAND_Sss_Aaaaa.OUT (src/bandstr.f90). One file per atom, named
``write(fname,'("BAND_S",I2.2,"_A",I4.4,".OUT")') is,ia``; blank-line
separated blocks, one per second-variational state, each block one line per
k-point along the ``plot1d`` path. Column layouts, transcribed from the
``write`` statements::

  case(21)   ! l character
    write(50,'(2G18.10,F12.6)',advance='NO') dpp1d(ik),evalsv(ist,ik),sm
    do l=0,lmaxdb
      write(50,'(F12.6)',advance='NO') bc(l,ias,ist,ik)
    end do
    write(50,*)
  case(22)   ! (l,m) character
    write(50,'(2G18.10)',advance='NO') dpp1d(ik),evalsv(ist,ik)
    do lm=1,lmmaxdb
      write(50,'(F12.6)',advance='NO') bc(lm,ias,ist,ik)
    end do
    write(50,*)
  case(23)   ! spin character
    write(50,'(2G18.10,2F12.6)') dpp1d(ik),evalsv(ist,ik), &
     (bc(ispn,ias,ist,ik),ispn=1,nspinor)
  case(24)   ! magnetic moment character
    write(50,'(2G18.10,3F12.6)') dpp1d(ik),evalsv(ist,ik), &
     (bc(idm,ias,ist,ik),idm=1,ndmag)

so the column count is 3+(lmaxdb+1) for task 21 (the third column being the
sum over l, i.e. the atom's total character), 2+(lmaxdb+1)^2 for task 22,
4 for task 23 and 2+ndmag for task 24. ``lmaxdb`` defaults to 3, i.e.
s, p, d, f. Elk has already subtracted the Fermi energy from ``evalsv``.

PDOS_Sss_Aaaaa.OUT / IDOS.OUT / TDOS.OUT (src/dos.f90)::

  do ispn=1,nsd
    do l=l0,l1
      do iw=1,nwplot
        write(50,'(2G18.10)') w(iw),dp(iw,l,ispn)*sps(ispn)
      end do
      write(50,*)

with ``sps(1)=1; sps(2)=-1``, ``nsd = 1`` when ``dosssum`` (or when the run
is not spin-polarised) and ``nspinor`` otherwise, and ``l0,l1 = 0,lmaxdb``
when ``dosmsum`` else ``1,lmmaxdb``.

TWO traps live in that loop. The SPIN-DOWN channel is written NEGATIVE
(``sps(2) = -1``) -- a plotting convention, not physics, so summing the
blocks naively gives the magnetisation rather than the total DOS. And the
partial-DOS channels are subtracted from the total as they are computed, so
IDOS.OUT holds the INTERSTITIAL remainder (total minus every muffin-tin
channel), not a second copy of the total.
"""

import numpy as np

#: column layout of each band-character task: (task code, description)
BAND_CHARACTER_TASKS = {
    "l": 21,
    "lm": 22,
    "spin": 23,
    "moment": 24,
}


def parse_band_character(path, kind="l"):
    """Parse one BAND_Sss_Aaaaa.OUT file (src/bandstr.f90, tasks 21-24).

    ``kind`` is "l", "lm", "spin" or "moment", matching the task that wrote
    the file; it only selects how the trailing columns are labelled, since
    every task shares the block layout.

    Returns a dict:

    - "distances" (npoints,): distance along the plot1d path, Bohr^-1.
    - "energies" (nbands, npoints): Hartree, Fermi energy already
      subtracted by Elk.
    - "characters" (nbands, npoints, ncolumns): the trailing columns as
      written, i.e. for kind="l" the total followed by l=0..lmaxdb, for
      kind="lm" the (l,m) channels in Elk's lm order, for kind="spin" the
      spin-up and spin-down weights, for kind="moment" the ndmag components
      of the moment.
    - for kind="l" additionally "total" (nbands, npoints) -- the sum over l,
      i.e. the whole muffin-tin weight of that atom -- and "l" (nbands,
      npoints, lmaxdb+1), the per-l channels with "total" stripped off.
    - "lmaxdb" for kind="l"/"lm": the l cut-off implied by the column count.
    """
    if kind not in BAND_CHARACTER_TASKS:
        raise ValueError(
            f"unknown band-character kind '{kind}'; use one of "
            f"{sorted(BAND_CHARACTER_TASKS)}"
        )
    blocks = []
    current = []
    with open(path) as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                if current:
                    blocks.append(current)
                    current = []
                continue
            current.append([float(t) for t in stripped.split()])
    if current:
        blocks.append(current)
    if not blocks:
        raise ValueError(f"{path} contains no band blocks")

    data = np.array(blocks, dtype=float)  # (nbands, npoints, ncols)
    result = {
        "distances": data[0, :, 0],
        "energies": data[:, :, 1],
        "characters": data[:, :, 2:],
        "kind": kind,
    }
    if kind == "l":
        result["total"] = data[:, :, 2]
        result["l"] = data[:, :, 3:]
        result["lmaxdb"] = data.shape[2] - 4
    elif kind == "lm":
        nlm = data.shape[2] - 2
        lmax = int(round(np.sqrt(nlm))) - 1
        result["lmaxdb"] = lmax
    return result


def parse_dos_blocks(path):
    """Parse any of Elk's blank-line-separated two-column DOS files
    (TDOS.OUT, PDOS_Sss_Aaaaa.OUT, IDOS.OUT -- src/dos.f90).

    Returns (energies, blocks): energies shape (nw,) in Hartree relative to
    the Fermi energy, blocks shape (nblocks, nw) in states/Hartree/unit
    cell, in file order.

    Note the spin-down blocks carry ``sps(2) = -1``, i.e. Elk writes them
    negative for plotting; parse_partial_dos() undoes that.
    """
    blocks = []
    current = []
    with open(path) as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                if current:
                    blocks.append(current)
                    current = []
                continue
            x, y = stripped.split()
            current.append((float(x), float(y)))
    if current:
        blocks.append(current)
    if not blocks:
        raise ValueError(f"{path} contains no DOS blocks")
    energies = np.array([p[0] for p in blocks[0]])
    return energies, np.array([[p[1] for p in block] for block in blocks])


def parse_partial_dos(path, nspin=1, undo_spin_sign=True):
    """Parse a PDOS_Sss_Aaaaa.OUT file into (energies, dos) with the spin
    axis split out.

    ``nspin`` is ``nsd`` in src/dos.f90's notation: 1 for a
    spin-unpolarised run or one with ``dosssum``, 2 otherwise. The file's
    blocks run spin-slowest, channel-fastest.

    Returns (energies, dos) with dos shape (nspin, nchannels, nw). The
    channel axis is l=0..lmaxdb when the run used ``dosmsum``, else the
    (l,m) channels lm=1..lmmaxdb -- the file itself carries no marker
    distinguishing the two, so the caller has to know which it asked for
    (the channel count settles it: lmaxdb+1 versus (lmaxdb+1)^2).

    With ``undo_spin_sign`` (the default) the spin-down channel's Elk
    plotting sign is removed, so both channels are non-negative and their
    sum is the spin-summed partial DOS.
    """
    energies, blocks = parse_dos_blocks(path)
    nblocks = blocks.shape[0]
    if nblocks % nspin != 0:
        raise ValueError(
            f"{path} has {nblocks} blocks, not divisible by nspin={nspin}"
        )
    dos = blocks.reshape(nspin, nblocks // nspin, -1)
    if undo_spin_sign and nspin == 2:
        dos = dos.copy()
        dos[1] = -dos[1]
    return energies, dos
