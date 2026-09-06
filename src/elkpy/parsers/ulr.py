"""Parsers for Elk's ultra-long-range (ULR) task family.

The ULR method (T. Mueller, S. Sharma, E. K. U. Gross and J. K. Dewhurst,
*Ultra-long-range order in the Fermi-Hubbard model...*, PRL **125**, 256402
(2020), arXiv:2008.12573) solves a Kohn-Sham problem on an *ultracell* -- a
supercell many unit cells across, defined by the ``avecu``/``scaleu`` input
blocks -- without ever building that supercell's basis. Instead the
long-range degrees of freedom enter through a small set of Q-vectors of the
ultracell (grid ``ngridq``) and the associated kappa-points (``ngridkpa``),
so the cost scales with the number of Q-points rather than with the
ultracell's atom count.

Files parsed here:

- ``TDOSULR.OUT`` -- task 710 (``vendor/elk/src/writedosu.f90``);
- ``BANDULR.OUT`` and ``BANDSFU.OUT`` -- tasks 720/725
  (``vendor/elk/src/bandstrulr.f90``);
- ``RHOU1D.OUT``/``VSU1D.OUT``/``MAGU1D.OUT`` and their ``*LINES.OUT``
  companions -- tasks 731/741/771 (``vendor/elk/src/plotu1d.f90``).

The 2D and 3D members of the plotting family (732/742/772 and 733/743/773)
write the *identical* layout as the ordinary ``plot2d``/``plot3d`` tasks --
compare ``plotu2d.f90``/``plotu3d.f90`` with ``plot2d.f90``/``plot3d.f90``:
the same ``write(fnum,'(2I6," : grid size")')`` header and the same
coordinate-then-values rows -- so they are read with the existing
``elkpy.parsers.volumetric.parse_plot2d`` / ``parse_plot3d`` and get no
duplicate reader here.
"""

import numpy as np


def parse_ulr_dos(path):
    """Ultra-long-range total density of states, from ``TDOSULR.OUT``
    (task 710).

    src/writedosu.f90::

        open(50,file='TDOSULR.OUT',form='FORMATTED',action='WRITE')
        do iw=1,nwplot
          write(50,'(2G18.10)') w(iw),g(iw)
        end do

    Returns ``(energies, dos)`` in Hartree and states/Hartree/unit cell; the
    Fermi energy has already been subtracted (``evalu = evalu - efermi``
    before the Brillouin-zone integration) and the DOS is normalised to the
    *unit* cell, not the ultracell (``f = 1/nkpa``).
    """
    data = np.atleast_2d(np.loadtxt(path))
    return data[:, 0], data[:, 1]


def parse_ulr_bands(path, nkappa=1):
    """Ultra-long-range band structure, from ``BANDULR.OUT`` (tasks 720/725).

    src/bandstrulr.f90 appends one blank-line-separated block per ULR state,
    for every kappa-point in turn::

        do ist=1,nstulr
          do ik0=1,nkpt0
            write(50,'(3G18.10)') dpp1d(ik0),evalu(ist,ik0),chkpa(ist,ik0)
          end do
          write(50,*)
        end do

    Three columns, not two -- so ``elkpy.parsers.band.parse_bands`` (which
    unpacks exactly two fields per line) cannot read this file. The third
    column is the *kappa-point character* of that state, the weight with
    which the ULR eigenvector projects onto the kappa-point currently being
    plotted (``charkpa.f90``); it is what turns the folded ULR spectrum back
    into an unfolded band structure.

    Task 720 plots only kappa = 0 (one pass); task 725 plots every
    kappa-point, and because the file is opened with ``position='APPEND'``
    the result is ``nkpa`` consecutive groups of ``nstulr`` blocks. Pass
    `nkappa` to split them: the returned arrays then have leading shape
    ``(nkappa, nstates, npoints)``.

    Returns ``(distances, energies, characters)``. Energies are Hartree with
    the Fermi energy subtracted.
    """
    blocks, current = [], []
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                if current:
                    blocks.append(current)
                    current = []
                continue
            fields = line.split()
            current.append((float(fields[0]), float(fields[1]), float(fields[2])))
    if current:
        blocks.append(current)
    if not blocks:
        raise ValueError(f"{path} contains no band blocks")
    arr = np.array(blocks)  # (nblocks, npoints, 3)
    distances = arr[0, :, 0]
    energies = arr[:, :, 1]
    characters = arr[:, :, 2]
    if nkappa > 1:
        if arr.shape[0] % nkappa:
            raise ValueError(
                f"{path}: {arr.shape[0]} blocks is not a multiple of nkappa={nkappa}"
            )
        nstates = arr.shape[0] // nkappa
        energies = energies.reshape(nkappa, nstates, -1)
        characters = characters.reshape(nkappa, nstates, -1)
    return distances, energies, characters


def parse_ulr_band_spectral(path):
    """Ultra-long-range spectral-function band structure, from
    ``BANDSFU.OUT`` (tasks 720/725).

    src/bandstrulr.f90::

        write(50,'(2I6," : grid size")') nkpt0,nwplot
        do iw=1,nwplot
          do ik0=1,nkpt0
            write(50,'(3G18.10)') dpp1d(ik0),w(iw),sfu(iw,ik0)
          end do
        end do

    As in ``GWBAND.OUT``, the header names the two grid sizes in the reverse
    of the loop order: frequencies run on the outside and k-path points on
    the inside, so the rows reshape as ``(nwplot, nkpt0)``.

    Returns ``(distances, frequencies, sf)`` with `distances` shape
    ``(nkpt0,)``, `frequencies` shape ``(nwplot,)`` and `sf` shape
    ``(nwplot, nkpt0)``.
    """
    with open(path) as fh:
        header = fh.readline().split()
        nkpt0, nwplot = int(header[0]), int(header[1])
        data = np.atleast_2d(np.loadtxt(fh))
    if data.shape[0] != nkpt0 * nwplot:
        raise ValueError(
            f"{path}: {data.shape[0]} rows, header says nkpt0={nkpt0} x "
            f"nwplot={nwplot} = {nkpt0 * nwplot}"
        )
    block = data.reshape(nwplot, nkpt0, 3)
    return block[0, :, 0], block[:, 0, 1], block[:, :, 2]


def parse_plotu1d(path, nf=1):
    """Parse a 1D ULR plot file -- ``RHOU1D.OUT`` (task 731),
    ``VSU1D.OUT`` (741) or ``MAGU1D.OUT`` (771).

    src/plotu1d.f90::

        do ip=1,npp1d
          write(fnum1,'(5G18.10)') dpp1d(ip),(fp(ip,jf),jf=1,nf)
        end do

    One row per plot point: the distance along the path (Bohr) followed by
    `nf` function values, and **no header line** (unlike the 2D/3D members
    of the family). `nf` is 1 for the density and the potential, and
    ``ndmag`` for the magnetisation -- 1 for a collinear (``cmagz``)
    calculation and 3 for a non-collinear one.

    Returns ``(distances, values)`` with `values` shape ``(npoints,)`` when
    ``nf == 1`` and ``(npoints, nf)`` otherwise.
    """
    data = np.atleast_2d(np.loadtxt(path))
    distances = data[:, 0]
    values = data[:, 1 : 1 + nf]
    if nf == 1:
        values = values[:, 0]
    return distances, values


def parse_plotu_lines(path):
    """Vertex positions from a ``*LINES.OUT`` companion file
    (``RHOULINES.OUT``, ``VSULINES.OUT``, ``MAGULINES.OUT``).

    src/plotu1d.f90 writes each vertex twice, once at the bottom and once at
    the top of the plotting range::

        do iv=1,nvp1d
          write(fnum2,'(2G18.10)') dvp1d(iv),fmin
          write(fnum2,'(2G18.10)') dvp1d(iv),fmax
          write(fnum2,*)
        end do

    Returns the distinct abscissae, in path order -- the x positions at
    which to draw vertical gridlines alongside :func:`parse_plotu1d`.
    """
    positions = []
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            x = float(line.split()[0])
            if not positions or positions[-1] != x:
                positions.append(x)
    return np.array(positions)
