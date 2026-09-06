"""Parsers for Elk's generic 1D/2D/3D plotting writers.

Elk's whole real-space plotting family is organised as triples in which the
last digit of the task code selects the dimensionality -- x1 writes a 1D line
plot, x2 a 2D plane, x3 a 3D parallelepiped -- and every member of every
triple goes through the SAME three writers, ``src/plot1d.f90``,
``src/plot2d.f90`` and ``src/plot3d.f90``. So one parser per writer covers
the density (31/32/33), potential (41/42/43), ELF (51/52/53), wavefunction
modulus (61/62/63), STM image (162), magnetisation (71/72/73), exchange-
correlation field (81/82/83), electric field (141/142/143), m x B_xc
(151/152/153), div B_xc (91/92/93), meta-GGA W_xc (341/342/343),
paramagnetic current (371/372/373) and static density (471) alike.

``src/elkpy/parsers/volumetric.py`` already covers plot2d and (without the
grid shape) plot3d; this module adds plot1d, the vertex-location "LINES"
companion file, and a grid-returning plot3d, and is the place any further
plot-writer knowledge should go.

FORMATS, transcribed from the Fortran ``write`` statements:

- plot1d (``src/plot1d.f90``)::

      do ip=1,npp1d
        write(fnum1,'(5G18.10)') dpp1d(ip),(fp(ip,jf),jf=1,nf)
      end do

  i.e. NO header line at all, one line per plotting point, the cumulative
  distance along the path (Bohr, or Bohr^-1 for a reciprocal-space plot)
  followed by 1-4 function values. ``npp1d`` points are laid out along the
  segments joining the ``plot1d`` block's vertices by ``src/plotpt1d.f90``.

- the vertex-location companion (same file, second unit)::

      do iv=1,nvp1d
        write(fnum2,'(2G18.10)') dvp1d(iv),fmin
        write(fnum2,'(2G18.10)') dvp1d(iv),fmax
        write(fnum2,*)
      end do

  i.e. each vertex is written as a two-point vertical line segment spanning
  the plot's value range, blank-line separated. Only the x-coordinate (the
  distance along the path) carries information a caller wants.

- plot3d (``src/plot3d.f90``)::

      write(fnum,'(3I6," : grid size")') np3d(:)
      do ip=1,np
        call r3mv(avec,vpl(:,ip),v1)
        write(fnum,'(7G18.10)') v1(:),(fp(ip,jf),jf=1,nf)
      end do

  The GRID ORDER is load-bearing. ``src/plotpt3d.f90`` runs::

      do i3=0,np3d(3)-1
        do i2=0,np3d(2)-1
          do i1=0,np3d(1)-1

so the FIRST plotting vector's index runs fastest and a column reshapes to
a volume as ``values.reshape(n3, n2, n1)`` -- getting it backwards permutes
the axes without changing a single number, exactly the trap
``volumetric.parse_plot2d`` already documents for the 2D case.
"""

import numpy as np


def parse_plot1d(path, nf=1):
    """Parse a plot1d-family output file (src/plot1d.f90).

    Returns (distances, values): distances shape (N,) -- cumulative distance
    along the plotting path, Bohr (real-space plots) or Bohr^-1 (reciprocal-
    space plots such as EMD1D.OUT) -- and values shape (N,) if nf=1 else
    (N, nf).
    """
    data = np.atleast_2d(np.loadtxt(path))
    distances = data[:, 0]
    values = data[:, 1 : 1 + nf]
    if nf == 1:
        values = values[:, 0]
    return distances, values


def parse_plot_lines(path):
    """Parse a vertex-location "LINES" file (RHOLINES.OUT, VLINES.OUT,
    ELFLINES.OUT, MAGLINES.OUT, ... -- src/plot1d.f90's second unit).

    Returns the distance-along-path of each ``plot1d`` vertex, in file
    order, for drawing vertical gridlines alongside parse_plot1d()'s output.
    Each vertex is written twice (once at the plot minimum, once at the
    maximum), so consecutive duplicates are collapsed.
    """
    positions = []
    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            x = float(stripped.split()[0])
            if not positions or positions[-1] != x:
                positions.append(x)
    return positions


def parse_plot3d(path, nf=1):
    """Parse a plot3d-family output file (src/plot3d.f90), returning the
    grid shape as well.

    Returns (points, values, grid): points shape (N, 3) in Cartesian Bohr
    (Bohr^-1 for a reciprocal-space plot such as EMD3D.OUT), values shape
    (N,) if nf=1 else (N, nf), and grid the (n1, n2, n3) triple from the
    header.

    The first plotting vector's index runs fastest (src/plotpt3d.f90's
    innermost loop), so a column reshapes to a volume as
    ``values.reshape(n3, n2, n1)`` -- see reshape_plot3d().
    """
    with open(path) as fh:
        header = fh.readline().split()
        grid = (int(header[0]), int(header[1]), int(header[2]))
        data = np.atleast_2d(np.loadtxt(fh))
    points = data[:, :3]
    values = data[:, 3 : 3 + nf]
    if nf == 1:
        values = values[:, 0]
    return points, values, grid


def reshape_plot3d(values, grid):
    """Reshape one plot3d column (N,) into a (n3, n2, n1) volume.

    ``grid`` is the (n1, n2, n3) triple parse_plot3d() returns. The axis
    order is reversed because src/plotpt3d.f90 runs i1 innermost.
    """
    n1, n2, n3 = grid
    return np.asarray(values).reshape(n3, n2, n1)


def reshape_plot2d(values, grid):
    """Reshape one plot2d column (N,) into a (n2, n1) image.

    ``grid`` is the (n1, n2) pair volumetric.parse_plot2d() returns;
    src/plotpt2d.f90 runs i1 innermost, same convention as above.
    """
    n1, n2 = grid
    return np.asarray(values).reshape(n2, n1)
