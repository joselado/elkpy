"""Parsers for the Fermi-surface family: tasks 100/101/103/104
(``src/fermisurf.f90``), task 102 (``src/fermisurfbxsf.f90``, the XCrySDen
``.bxsf`` format) and task 105 (``src/nesting.f90``).

None of these go through Elk's generic plot writers -- each writes its own
format -- so the exact ``write`` statements are transcribed below.

FERMISURF.OUT (src/fermisurf.f90). Two shapes, selected by the task:

- tasks 100 and 103, one scalar field per grid point::

      write(50,'(3I6," : grid size")') np3d(:)
      ...
      write(50,'(4G18.10)') vpc(:,i),fn

  task 100's ``fn`` is ``product(evalsv(ist0:ist1,ik)-efermi)``, the product
  of the Fermi-energy-referenced eigenvalues over the bands that cross E_F
  -- an isosurface at zero is the Fermi surface. Task 103's is
  ``sum_ist sdelta(stype,(evalsv-efermi)/swidth)/swidth``, a single smeared
  delta function at E_F.

- tasks 101 and 104, one column per band::

      write(50,'(4I6," : grid size, number of states")') np3d(:),ist1-ist0+1
      ...
      write(50,'(3G18.10)',advance='NO') vpc(:,i)
      do ist=ist0,ist1
        write(50,'(F14.8)',advance='NO') evalsv(ist,ik)-efermi   ! task 101
        write(50,'(F14.8)',advance='NO') sdelta(...)/swidth      ! task 104
      end do
      write(50,*)

  keeping the bands separate so each band's sheet can be plotted on its own.

  ``vpc`` is Cartesian (Bohr^-1): ``plotpt3d`` generates the grid in the
  ``plot3d`` block's fractional coordinates and ``r3mv(bvec,...)`` converts
  it. The write loop runs ``i3`` outermost and ``i1`` innermost over
  ``ngridk``, so -- exactly as for plot3d -- the first index runs fastest
  and a column reshapes as ``(n3, n2, n1)``.

  With collinear magnetism (``ndmag == 1``) fermisurf writes TWO files,
  FERMISURF_UP.OUT and FERMISURF_DN.OUT, splitting the second-variational
  states at ``nstfv``; otherwise a single FERMISURF.OUT.

FERMISURF.bxsf (src/fermisurfbxsf.f90), the XCrySDen band-grid format::

      write(50,'(" BEGIN_INFO")')
      ... "   Fermi Energy: ",G18.10  (written as 0, energies are shifted)
      write(50,'(" END_INFO")')
      write(50,'(" BEGIN_BLOCK_BANDGRID_3D")')
      write(50,'(" band_energies")')
      write(50,'(" BANDGRID_3D_BANDS")')
      write(50,'(I4)') nst
      write(50,'(3I6)') ngridk(:)+1
      write(50,'(3G18.10)') 0.d0,0.d0,0.d0
      do i=1,3; write(50,'(3G18.10)') bvec(:,i); end do
      do ist=ist0,ist1
        write(50,'(" BAND: ",I4)') ist
        do i1=0,ngridk(1); do i2=0,ngridk(2); do i3=0,ngridk(3)
          write(50,'(G18.10)') evalsv(ist,ik)-efermi

  Note the grid is ``ngridk+1`` in each direction (the periodic image of the
  first plane is repeated, which is what XCrySDen's isosurface needs) and
  that here ``i3`` is INNERMOST -- the opposite of every plot3d-family file
  -- so a band reshapes as ``(n1+1, n2+1, n3+1)`` in plain C order.

NEST3D.OUT / NESTING.OUT (src/nesting.f90)::

      write(50,'(3I6," : grid size")') ngridq(:)
      ...
      write(50,'(4G18.10)') vc(:),nq(iq)          ! NEST3D.OUT
      write(50,'(G18.10)') sm0                    ! NESTING.OUT

  ``vc`` is Cartesian (Bohr^-1), the loop again runs ``i3`` outermost so the
  first index is fastest. ``src/init2.f90`` FORCES ``ngridq = ngridk`` for
  task 105, so the nesting grid is always the k-mesh.
"""

import numpy as np


def _split_values(line, nvalues, leading=3, leading_width=18, value_width=14):
    """Split one fermisurf task-101/104 data line into floats.

    Whitespace splitting is correct for every physically plausible value,
    but the Fortran writes the per-band columns with ``F14.8`` and NO
    separator, so a magnitude of 10^5 or more (or five digits plus a sign)
    would fill the field completely and run into its neighbour. Fall back to
    fixed-width slicing when the token count disagrees, rather than raising
    on a file Elk considers perfectly well formed.
    """
    tokens = line.split()
    if len(tokens) == leading + nvalues:
        return [float(t) for t in tokens]
    values = []
    pos = 0
    for _ in range(leading):
        values.append(float(line[pos : pos + leading_width]))
        pos += leading_width
    for _ in range(nvalues):
        values.append(float(line[pos : pos + value_width]))
        pos += value_width
    return values


def parse_fermisurf(path):
    """Parse FERMISURF.OUT / FERMISURF_UP.OUT / FERMISURF_DN.OUT
    (src/fermisurf.f90, tasks 100/101/103/104).

    Returns a dict:

    - "points" (N, 3): Cartesian reciprocal-space coordinates, Bohr^-1.
    - "grid" (n1, n2, n3): the plotting grid from the header.
    - "values": (N,) for tasks 100/103 (a single scalar field), or
      (N, nstates) for tasks 101/104 (one column per band).
    - "nstates": the band count for tasks 101/104, else None. This is the
      number of bands that CROSS the Fermi energy (Elk's own ist0..ist1
      search), not the total, so it varies with the k-mesh.
    - "per_band": True for the tasks 101/104 shape.

    The first grid index runs fastest, so a scalar field reshapes to a
    volume as ``values.reshape(n3, n2, n1)`` (see plots.reshape_plot3d).
    """
    with open(path) as fh:
        header = fh.readline().split()
        # the header ends in a literal " : grid size[, number of states]"
        # comment, so take only the leading integers
        ints = []
        for token in header:
            try:
                ints.append(int(token))
            except ValueError:
                break
        # keep the lines UNSTRIPPED: the fixed-width fallback in
        # _split_values() counts characters from column 1, and Fortran's
        # G18.10 pads on the left
        rows = [raw.rstrip("\n") for raw in fh if raw.strip()]
    if len(ints) >= 4:
        grid = (ints[0], ints[1], ints[2])
        nstates = ints[3]
        data = np.array([_split_values(row, nstates) for row in rows], dtype=float)
        return {
            "points": data[:, :3],
            "grid": grid,
            "values": data[:, 3:],
            "nstates": nstates,
            "per_band": True,
        }
    grid = (ints[0], ints[1], ints[2])
    data = np.array([[float(t) for t in row.split()] for row in rows], dtype=float)
    return {
        "points": data[:, :3],
        "grid": grid,
        "values": data[:, 3],
        "nstates": None,
        "per_band": False,
    }


def parse_bxsf(path):
    """Parse an XCrySDen ``.bxsf`` band-grid file (src/fermisurfbxsf.f90,
    task 102).

    Returns a dict:

    - "grid" (n1, n2, n3): the band-grid shape AS WRITTEN, i.e. Elk's
      ``ngridk + 1`` -- the extra plane is the periodic image of the first,
      which is what an isosurface needs.
    - "origin" (3,): the reciprocal-space origin (Elk writes zero).
    - "bvec" (3, 3): the reciprocal lattice vectors, one per ROW, in
      Bohr^-1 (Elk writes ``bvec(:,i)`` for i=1..3, and stores them
      column-wise, so each written line is one full vector).
    - "band_indices": the Elk state index of each band written.
    - "energies" (nbands, n1, n2, n3): eigenvalues in Hartree relative to
      the Fermi energy (Elk subtracts E_F and declares "Fermi Energy: 0").
      The LAST index runs fastest here, unlike every plot3d-family file.
    - "fermi_energy": the value on the ``Fermi Energy:`` line, always 0.
    """
    with open(path) as fh:
        lines = [raw.rstrip("\n") for raw in fh]

    fermi_energy = 0.0
    for line in lines:
        if "Fermi Energy:" in line:
            fermi_energy = float(line.split(":", 1)[1])
            break

    i = next(i for i, line in enumerate(lines) if "BANDGRID_3D_BANDS" in line)
    nbands = int(lines[i + 1].split()[0])
    grid = tuple(int(t) for t in lines[i + 2].split()[:3])
    origin = np.array([float(t) for t in lines[i + 3].split()[:3]])
    bvec = np.array([[float(t) for t in lines[i + 4 + j].split()[:3]] for j in range(3)])

    npts = grid[0] * grid[1] * grid[2]
    band_indices = []
    energies = []
    j = i + 7
    while len(band_indices) < nbands:
        line = lines[j].strip()
        if not line.startswith("BAND:"):
            j += 1
            continue
        band_indices.append(int(line.split(":", 1)[1]))
        block = [float(lines[j + 1 + n]) for n in range(npts)]
        energies.append(np.array(block).reshape(grid))
        j += 1 + npts

    return {
        "grid": grid,
        "origin": origin,
        "bvec": bvec,
        "band_indices": band_indices,
        "energies": np.array(energies),
        "fermi_energy": fermi_energy,
    }


def parse_nesting(nest3d_path, nesting_path=None):
    """Parse NEST3D.OUT (and optionally NESTING.OUT) from task 105,
    src/nesting.f90.

    Returns a dict:

    - "points" (N, 3): Cartesian q-vectors, Bohr^-1.
    - "grid" (n1, n2, n3): the q-mesh, which src/init2.f90 forces equal to
      ``ngridk`` for this task.
    - "values" (N,): the nesting function N(q). The first index runs
      fastest, so it reshapes as ``values.reshape(n3, n2, n1)``.
    - "total": the Brillouin-zone integral of N(q) per unit volume, from
      NESTING.OUT, or None if that file was not given.
    """
    with open(nest3d_path) as fh:
        header = fh.readline().split()
        grid = (int(header[0]), int(header[1]), int(header[2]))
        data = np.atleast_2d(np.loadtxt(fh))
    total = None
    if nesting_path is not None:
        with open(nesting_path) as fh:
            total = float(fh.read().split()[0])
    return {
        "points": data[:, :3],
        "grid": grid,
        "values": data[:, 3],
        "total": total,
    }
