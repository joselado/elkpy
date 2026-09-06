"""Parsers for the small text reports of the atomic / expectation-value
tasks: 15 and 16 (``src/writelsj.f90``), 150 (``src/writeevsp.f90``), 65
(``src/wfcrplot.f90``), 14 (``src/writesf.f90``) and 140
(``src/elnes.f90``).

Every format below is transcribed from the ``write`` statement that
produces it.

LSJ.OUT (task 15, src/writelsj.f90)::

    write(50,*)
    write(50,'("Expectation values are computed only over the muffin-tin")')
    do is=1,nspecies
      write(50,*)
      write(50,'("Species : ",I0," (",A,")")') is,trim(spsymb(is))
      do ia=1,natoms(is)
        write(50,'(" atom : ",I0)') ia
        write(50,'("  L : ",3G18.10)') xl(:)
        write(50,'("  S : ",3G18.10)') xs(:)
        write(50,'("  J : ",3G18.10)') xj(:,ias)

LSJ_KST.OUT (task 16, same file, one group per (k-point, state, atom))::

    write(50,'("k-point : ",I0,3G18.10)') ik,vkl(:,ik)
    write(50,'("state : ",I0)') ist
    write(50,'("species : ",I0," (",A,"), atom : ",I0)') is,spsymb(is),ia
    write(50,'(" L : ",3G18.10)') xl(:)
    write(50,'(" S : ",3G18.10)') xs(:)
    write(50,'(" J : ",3G18.10)') xj(:,ias)

EVALSP.OUT (task 150, src/writeevsp.f90)::

    write(50,'("Exchange-correlation functional : ",3I6)') xctsp(:)
    do is=1,nspecies
      write(50,'("Species : ",I4," (",A,")",I4)') is,trim(spsymb(is))
      do ist=1,nstsp(is)
        write(50,'(" n = ",I2,", l = ",I2,", k = ",I2," : ",G18.10)') &
         nsp(ist,is),lsp(ist,is),ksp(ist,is),evalsp(ist,is)

(the trailing ``I4`` on the Species line has no matching output item, so
Fortran stops there and it prints nothing.)

WFCORE_Sss_Aaaaa.OUT (task 65, src/wfcrplot.f90)::

    do ist=1,nstsp(is)
      if (spcore(ist,is)) then
        do ir=1,nrsp(is)
          write(50,'(2G18.10)') rsp(ir,is),rwfcr(ir,1,ist,ias)
        end do
        write(50,*)

i.e. blank-line-separated blocks, ONE PER CORE STATE ONLY (valence states
are skipped by the ``spcore`` test), each holding the species' full radial
mesh. The second column is ``rwfcr``, the radial function ALREADY
MULTIPLIED BY r -- Elk stores u(r) = r R(r), so the physical radial
function is the column divided by r and the plotted quantity integrates to
1 as int |u|^2 dr, with no extra r^2 Jacobian.

SDELTA.OUT / STHETA.OUT (task 14, src/writesf.f90) and ELNES.OUT (task 140,
src/elnes.f90) are plain ``'(2G18.10)'`` two-column files with no header.
"""

import numpy as np


def _floats(text):
    return [float(t) for t in text.split()]


def parse_lsj(path):
    """Parse LSJ.OUT (task 15).

    Returns a list of dicts in file order, one per atom::

        {"species": 1, "symbol": "Si", "atom": 1,
         "L": array([...]), "S": array([...]), "J": array([...])}

    with L, S and J the Cartesian expectation values (dimensionless, in
    units of hbar) of the muffin-tin orbital angular momentum, spin and
    total angular momentum. They are muffin-tin quantities only: the
    interstitial contribution is not included, which is why L for a bulk
    sp-bonded solid is essentially zero rather than exactly so.
    """
    entries = []
    species = symbol = atom = None
    current = {}
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("Species :"):
                body = line.split(":", 1)[1].strip()
                species = int(body.split("(")[0].strip())
                symbol = body.split("(", 1)[1].split(")", 1)[0]
            elif line.startswith("atom :"):
                atom = int(line.split(":", 1)[1])
                current = {"species": species, "symbol": symbol, "atom": atom}
                entries.append(current)
            elif line[:1] in ("L", "S", "J") and ":" in line:
                key = line[0]
                current[key] = np.array(_floats(line.split(":", 1)[1]))
    return entries


def parse_lsj_kst(path):
    """Parse LSJ_KST.OUT (task 16).

    Returns a list of dicts in file order, one per (k-point, state, atom)
    group::

        {"ik": 1, "k": array([kx, ky, kz]), "ist": 4,
         "species": 1, "symbol": "Si", "atom": 1,
         "L": array([...]), "S": array([...]), "J": array([...])}

    ``k`` is in lattice (fractional reciprocal) coordinates, as Elk's
    ``vkl``. Unlike task 15 these are single-state expectation values: the
    occupancies are replaced by a delta on one (k, state) pair and
    symmetrisation is switched off, so equivalent atoms are NOT averaged.
    """
    entries = []
    ik = kvec = ist = None
    current = None
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("k-point :"):
                fields = line.split(":", 1)[1].split()
                ik = int(fields[0])
                kvec = np.array([float(t) for t in fields[1:4]])
            elif line.startswith("state :"):
                ist = int(line.split(":", 1)[1])
            elif line.startswith("species :"):
                body = line.split(":", 1)[1]
                species = int(body.split("(")[0].strip())
                symbol = body.split("(", 1)[1].split(")", 1)[0]
                atom = int(body.rsplit(":", 1)[1])
                current = {
                    "ik": ik,
                    "k": kvec,
                    "ist": ist,
                    "species": species,
                    "symbol": symbol,
                    "atom": atom,
                }
                entries.append(current)
            elif line[:1] in ("L", "S", "J") and ":" in line and current is not None:
                current[line[0]] = np.array(_floats(line.split(":", 1)[1]))
    return entries


def parse_evalsp(path):
    """Parse EVALSP.OUT (task 150).

    Returns a list of dicts, one per species::

        {"species": 1, "symbol": "Si",
         "states": [{"n": 1, "l": 0, "k": 1, "energy": -65.2...}, ...]}

    These are the eigenvalues of the free-ATOM Kohn-Sham-Dirac equation
    that Elk solves in ``init0`` to build each species' starting density
    and to decide which states are core -- not solid-state eigenvalues.

    ``k`` is Elk's ``ksp``, the MAGNITUDE of the Dirac quantum number,
    ``|kappa| = j + 1/2``, not the signed kappa: ``src/readspecies.f90``
    refuses ``ksp < 1`` and ``src/initoep.f90`` counts the shell degeneracy
    as ``2*ksp``. So k = l selects j = l - 1/2 and k = l + 1 selects
    j = l + 1/2 (the test ``ksp == lsp+1`` in ``src/gradwfcr2.f90`` is
    exactly that), and an l > 0 shell appears TWICE with the gap between
    the pair being the free-atom spin-orbit splitting. Confirmed against a
    real run: aluminium's 2p pair comes back as k = 1 and k = 2 with
    occupations 2 and 4.
    """
    species_list = []
    current = None
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("Species :"):
                body = line.split(":", 1)[1].strip()
                index = int(body.split("(")[0].strip())
                symbol = body.split("(", 1)[1].split(")", 1)[0]
                current = {"species": index, "symbol": symbol, "states": []}
                species_list.append(current)
            elif line.startswith("n =") and current is not None:
                quantum, energy = line.rsplit(":", 1)
                numbers = {}
                for field in quantum.split(","):
                    key, value = field.split("=")
                    numbers[key.strip()] = int(value)
                current["states"].append(
                    {
                        "n": numbers["n"],
                        "l": numbers["l"],
                        "k": numbers["k"],
                        "energy": float(energy),
                    }
                )
    return species_list


def parse_wfcore(path):
    """Parse a WFCORE_Sss_Aaaaa.OUT file (task 65).

    Returns (r, u): the species radial mesh (nr,) in Bohr and the core
    radial functions (ncore, nr), one row per core state in Elk's own
    species-file order. ``u(r) = r R(r)``, so int |u|^2 dr = 1.
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
        raise ValueError(f"{path} contains no core-state blocks")
    r = np.array([p[0] for p in blocks[0]])
    u = np.array([[p[1] for p in block] for block in blocks])
    return r, u


def parse_two_column(path):
    """Parse any of Elk's plain ``'(2G18.10)'`` two-column, header-free
    files: SDELTA.OUT / STHETA.OUT (task 14) and ELNES.OUT (task 140).

    Returns (x, y) as 1D arrays.
    """
    data = np.atleast_2d(np.loadtxt(path))
    return data[:, 0], data[:, 1]
