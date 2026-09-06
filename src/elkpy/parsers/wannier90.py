"""Parsers for Elk's Wannier90 export (task 550, ``vendor/elk/src/writew90.f90``).

Task 550 writes its files in this order::

    call initw90        ! reads STATE.OUT, EVALSV.OUT
    call writew90win    ! <seedname>.win
    call writew90eig    ! <seedname>.eig
    call setupw90       ! -> wannier_setup(), the Wannier90 LIBRARY
    call writew90amn    ! <seedname>.amn
    call writew90mmn    ! <seedname>.mmn
    call writew90spn    ! <seedname>.spn

so the two files parsed here -- the ``.win`` input template and the ``.eig``
eigenvalue file -- are complete *before* the library call, while ``.amn``,
``.mmn`` and ``.spn`` all need the nearest-neighbour tables that
``wannier_setup`` returns. This build links ``w90_stub.f90``
(``SRC_W90S = w90_stub.f90`` in ``build-config/make.inc``), whose
``wannier_setup`` prints ``Error(wannier_setup): libwannier not or improperly
installed`` and executes ``error stop``. See ``elkpy.tasks`` for how that
partial success is surfaced rather than hidden.
"""

import numpy as np


def parse_w90_eig(path):
    """Eigenvalues from ``<seedname>.eig``.

    src/writew90eig.f90::

        do ik=1,nkptnr
          jk=ivkik(ivk(1,ik),ivk(2,ik),ivk(3,ik))
          do i=1,num_bands
            ist=idxw90(i)
            t1=evalsv(ist,jk)-efermi
            write(50,'(2I6,G18.10)') i,ik,t1*ha_ev
          end do
        end do

    i.e. one line per (band, k-point) with the band index fastest, over the
    **non-reduced** k-point set. Energies are in **electronvolts** (the
    Wannier90 convention, ``ha_ev``) and Fermi-referenced -- both departures
    from elkpy's usual Hartree/absolute convention, kept because the file is
    Wannier90's, not elkpy's.

    Returns ``(energies, nkpt, num_bands)`` with `energies` shape
    ``(nkpt, num_bands)``.
    """
    data = np.atleast_2d(np.loadtxt(path))
    bands = data[:, 0].astype(int)
    kpoints = data[:, 1].astype(int)
    num_bands = int(bands.max())
    nkpt = int(kpoints.max())
    if nkpt * num_bands != data.shape[0]:
        raise ValueError(
            f"{path}: {data.shape[0]} rows but max band index {num_bands} and "
            f"max k index {nkpt} imply {nkpt * num_bands}"
        )
    energies = data[:, 2].reshape(nkpt, num_bands)
    return energies, nkpt, num_bands


def parse_w90_win(path):
    """A light reader for the ``<seedname>.win`` template Elk writes.

    src/writew90win.f90 emits ``key = value`` scalars, then ``begin
    <name>`` / ``end <name>`` blocks (``unit_cell_cart``, ``atoms_frac``,
    optionally ``projections``, and ``kpoints``). Lattice vectors and
    positions are written ``3G18.10``; k-points are the non-reduced set in
    lattice coordinates.

    Returns a dict with ``settings`` (every ``key = value`` line, values left
    as strings), ``unit_cell_cart`` (3x3 array, **Bohr** -- writew90win.f90
    emits ``length_unit = bohr`` and then ``avec`` unconverted),
    ``atoms_frac`` (list of ``(symbol, position)``) and ``kpoints``
    ((nkpt, 3) array). Blocks the file does not contain are omitted.
    """
    settings = {}
    blocks = {}
    name = None
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("begin "):
                name = low.split(None, 1)[1].strip()
                blocks[name] = []
                continue
            if low.startswith("end "):
                name = None
                continue
            if name is not None:
                blocks[name].append(line)
            elif "=" in line:
                key, value = line.split("=", 1)
                settings[key.strip()] = value.strip()
    result = {"settings": settings}
    if "unit_cell_cart" in blocks:
        rows = [r for r in blocks["unit_cell_cart"] if r.lower() not in ("bohr", "ang")]
        result["unit_cell_cart"] = np.array(
            [[float(x) for x in r.split()[:3]] for r in rows]
        )
    if "atoms_frac" in blocks:
        result["atoms_frac"] = [
            (r.split()[0], np.array([float(x) for x in r.split()[1:4]]))
            for r in blocks["atoms_frac"]
        ]
    if "kpoints" in blocks:
        result["kpoints"] = np.array(
            [[float(x) for x in r.split()[:3]] for r in blocks["kpoints"]]
        )
    if "projections" in blocks:
        result["projections"] = list(blocks["projections"])
    return result
