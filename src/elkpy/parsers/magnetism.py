"""Parsers for Elk's magnetism task family.

Three unrelated file shapes live here because they are produced by the three
tasks that a magnetic-anisotropy study uses together:

- **magnetic anisotropy energy**, tasks 28/29 (``vendor/elk/src/mae.f90``) --
  ``MAE.OUT``, ``MAEPUV.OUT`` and the human-readable ``MAE_INFO.OUT``;
- **the exchange-correlation torque**, task 160
  (``vendor/elk/src/torque.f90``) -- which writes **no output file at all**,
  only three numbers on standard output, so the "parser" reads the run log;
- **supercell spin spirals**, tasks 350/351/352
  (``vendor/elk/src/spiralsc.f90``) -- one ``SS_Q..._..._....OUT`` file per
  q-point, whose name encodes the reduced fraction of the q-vector
  (``vendor/elk/src/ssfext.f90``).

Every format below is transcribed from the ``write`` statement that produces
it; see the citation on each function.
"""

import math
import re
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# magnetic anisotropy energy (tasks 28/29, src/mae.f90)
# ---------------------------------------------------------------------------


def _single_value(path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                return float(line.split()[0])
    raise ValueError(f"{path} contains no numeric value")


def parse_mae(path):
    """Magnetic anisotropy energy in Hartree, from ``MAE.OUT``.

    src/mae.f90::

        open(50,file='MAE.OUT',form='FORMATTED')
        write(50,'(G18.10)') de

    ``de = e1 - e0`` is the spread between the highest- and lowest-energy
    magnetisation directions sampled, i.e. an *estimate* of the MAE whose
    quality is set by ``npmae`` (how many directions were tried).
    """
    return _single_value(path)


def parse_mae_per_volume(path):
    """MAE per unit volume, Hartree/Bohr^3, from ``MAEPUV.OUT``
    (src/mae.f90: ``write(50,'(G18.10)') de/omega``)."""
    return _single_value(path)


_MAE_POINT_RE = re.compile(
    r"Fixed spin moment direction point\s+(\d+)\s+of\s+(\d+)"
)


def _after_colon(line):
    return line.split(":", 1)[1].split()


def parse_mae_info(path):
    """Parse ``MAE_INFO.OUT`` (src/mae.f90, unit 71).

    The file is written by these statements, in this order::

        write(71,'("Scale factor of spin-orbit coupling term : ",G18.10)') socscf
        ! then, once per sampled direction:
        write(71,'("Fixed spin moment direction point ",I0," of ",I0)') i,npmae
        write(71,'("Spherical coordinates of direction : ",2G18.10)') tpmae(:,i)
        write(71,'("Direction vector (Cartesian coordinates) : ",3G18.10)') v2
        write(71,'("Calculated total moment magnitude : ",G18.10)') momtotm
        write(71,'("Total energy : ",G24.14)') engytot
        ! then, once at the end:
        write(71,'("Minimum energy point : ",I6)') i0
        write(71,'("Maximum energy point : ",I6)') i1
        write(71,'("Estimated magnetic anisotropy energy (MAE) : ",G18.10)') de
        write(71,'("MAE per unit volume : ",G18.10)') de/omega

    Returns a dict with keys ``socscf``, ``directions`` (a list of per-point
    dicts: ``index``, ``theta``, ``phi``, ``direction`` (3,), ``moment``,
    ``energy``), ``min_point``, ``max_point``, ``mae``, ``mae_per_volume``.
    Missing trailing entries (an interrupted run) simply come back absent
    from the dict rather than raising, so a partial sweep is still readable.
    """
    result = {"directions": []}
    current = None
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Scale factor of spin-orbit coupling term"):
                result["socscf"] = float(_after_colon(line)[0])
                continue
            match = _MAE_POINT_RE.match(line)
            if match:
                current = {"index": int(match.group(1))}
                result["directions"].append(current)
                result["npmae"] = int(match.group(2))
                continue
            if line.startswith("Spherical coordinates of direction"):
                theta, phi = (float(x) for x in _after_colon(line)[:2])
                current["theta"] = theta
                current["phi"] = phi
            elif line.startswith("Direction vector"):
                current["direction"] = np.array(
                    [float(x) for x in _after_colon(line)[:3]]
                )
            elif line.startswith("Calculated total moment magnitude"):
                current["moment"] = float(_after_colon(line)[0])
            elif line.startswith("Total energy"):
                current["energy"] = float(_after_colon(line)[0])
            elif line.startswith("Minimum energy point"):
                result["min_point"] = int(_after_colon(line)[0])
            elif line.startswith("Maximum energy point"):
                result["max_point"] = int(_after_colon(line)[0])
            elif line.startswith("Estimated magnetic anisotropy energy"):
                result["mae"] = float(_after_colon(line)[-1])
            elif line.startswith("MAE per unit volume"):
                result["mae_per_volume"] = float(_after_colon(line)[0])
    return result


# ---------------------------------------------------------------------------
# exchange-correlation torque (task 160, src/torque.f90)
# ---------------------------------------------------------------------------

_TORQUE_MARKER = "Total torque exerted by B_xc on the magnetisation"


def parse_torque(log_path):
    """Total torque exerted by B_xc on the magnetisation, from the run log.

    Task 160 opens **no file**; src/torque.f90 ends with::

        write(*,*)
        write(*,'("Info(torque):")')
        write(*,'(" Total torque exerted by B_xc on the magnetisation :")')
        write(*,'(3G18.10)') torq

    so the three Cartesian components (atomic units, Hartree per unit of
    magnetisation) are only available from standard output -- which
    ``LocalLauncher.run`` captures into ``elk.out`` in the run directory.

    Note that ``torque.f90`` short-circuits to exactly zero unless the
    calculation is non-collinear (``ncmag``, i.e. ``ndmag == 3``): a
    collinear run returns ``[0, 0, 0]`` by construction, not by measurement.
    """
    lines = Path(log_path).read_text().splitlines()
    for i, line in enumerate(lines):
        if _TORQUE_MARKER in line:
            for follow in lines[i + 1 :]:
                if follow.strip():
                    return np.array([float(x) for x in follow.split()[:3]])
            break
    raise ValueError(
        f"no torque output found in {log_path}: expected a line containing "
        f"{_TORQUE_MARKER!r} followed by three numbers (src/torque.f90)"
    )


# ---------------------------------------------------------------------------
# supercell spin spirals (tasks 350/351/352, src/spiralsc.f90)
# ---------------------------------------------------------------------------


def spin_spiral_filename(ivq, ngridq):
    """The ``SS_Q...`` filename Elk gives the spiral at integer q-grid point
    ``ivq`` on a ``ngridq`` mesh.

    src/ssfext.f90 reduces each component to its lowest terms and formats
    numerator/denominator pairs two digits wide::

        do i=1,3
          if (ivq(i,iq) /= 0) then
            j=gcd(ivq(i,iq),ngridq(i))
            m(i)=ivq(i,iq)/j
            n(i)=ngridq(i)/j
          else
            m(i)=0
            n(i)=0
          end if
        end do
        write(fext,'("_Q",2I2.2,"_",2I2.2,"_",2I2.2,".OUT")') m(1),n(1),m(2), &
         n(2),m(3),n(3)

    and src/sstask.f90 prefixes ``'SS'``. So q = (1/2, 0, 0) on a 2x2x2 mesh
    is ``SS_Q0102_0000_0000.OUT``: a zero component gives ``0000``, not
    ``0002``.

    ``ivq`` components must be non-negative -- Elk's own ``gcd`` (src/gcd.f90)
    hard-stops on an argument below 1, so a negative index never reaches this
    filename.
    """
    if len(ivq) != 3 or len(ngridq) != 3:
        raise ValueError("ivq and ngridq must both have three components")
    parts = []
    for i in range(3):
        q, n = int(ivq[i]), int(ngridq[i])
        if q < 0:
            raise ValueError(
                f"negative q-grid index {q}: src/gcd.f90 stops on arguments < 1, "
                "so Elk never forms such a filename"
            )
        if q == 0:
            parts.append((0, 0))
        else:
            g = math.gcd(q, n)
            parts.append((q // g, n // g))
    return "SS_Q{:02d}{:02d}_{:02d}{:02d}_{:02d}{:02d}.OUT".format(
        parts[0][0], parts[0][1], parts[1][0], parts[1][1], parts[2][0], parts[2][1]
    )


def parse_spin_spiral(path):
    """Parse one ``SS_Q..._..._....OUT`` file (src/spiralsc.f90, unit 80).

    Written by::

        write(80,'(I6,T20," : number of unit cells in supercell")') nscss
        write(80,'(G18.10,T20," : total energy per unit cell")') engytot/dble(nscss)
        write(80,*)
        write(80,'("q-point in lattice and Cartesian coordinates :")')
        write(80,'(3G18.10)') vql(:,iqss)
        write(80,'(3G18.10)') vqc(:,iqss)
        write(80,'(G18.10,T20," : length of q-vector")') q
        write(80,*)
        write(80,'(I6,T20," : number of equivalent q-points")') nq
        write(80,'("Equivalent q-points in lattice and Cartesian coordinates :")')
        ! then, per equivalent q-point, a lattice line, a Cartesian line and
        ! a blank line

    Returns a dict with ``ncells`` (number of unit cells in the supercell),
    ``energy`` (total energy **per unit cell**, Hartree), ``q_lattice`` (3,),
    ``q_cartesian`` (3,), ``q_length``, ``nequivalent`` and ``equivalent``
    (a list of ``(lattice, cartesian)`` pairs).

    An **empty** file means the q-point has been claimed but not finished:
    that is exactly what task 352 (dry run) writes, and also what a task
    350/351 run leaves behind while it is still working on that q-point.
    That case raises ``ValueError`` rather than returning a half-result.
    """
    path = Path(path)
    lines = [line.rstrip("\n") for line in path.read_text().splitlines()]
    body = [line for line in lines if line.strip()]
    if not body:
        raise ValueError(
            f"{path} is empty: task 352 writes empty SS files as placeholders, "
            "and tasks 350/351 leave one empty while that q-point is still "
            "being computed (src/sstask.f90)"
        )
    result = {
        "ncells": int(body[0].split()[0]),
        "energy": float(body[1].split()[0]),
    }
    for i, line in enumerate(body):
        if line.startswith("q-point in lattice and Cartesian coordinates"):
            result["q_lattice"] = np.array([float(x) for x in body[i + 1].split()[:3]])
            result["q_cartesian"] = np.array(
                [float(x) for x in body[i + 2].split()[:3]]
            )
            result["q_length"] = float(body[i + 3].split()[0])
        elif "number of equivalent q-points" in line:
            result["nequivalent"] = int(line.split()[0])
        elif line.startswith("Equivalent q-points in lattice and Cartesian"):
            rest = body[i + 1 :]
            pairs = []
            for j in range(0, len(rest) - 1, 2):
                pairs.append(
                    (
                        np.array([float(x) for x in rest[j].split()[:3]]),
                        np.array([float(x) for x in rest[j + 1].split()[:3]]),
                    )
                )
            result["equivalent"] = pairs
    return result


def collect_spin_spirals(directory):
    """Every finished ``SS_Q*.OUT`` in `directory`, sorted by |q|.

    Returns ``(results, pending)``: `results` is a list of
    :func:`parse_spin_spiral` dicts, each carrying an extra ``"file"`` key;
    `pending` is the list of empty (claimed-but-unfinished, or dry-run)
    filenames. Reporting both is the point -- an interrupted supercell sweep
    is a normal state for tasks 350/351, not an error.
    """
    directory = Path(directory)
    results, pending = [], []
    for path in sorted(directory.glob("SS_Q*.OUT")):
        try:
            record = parse_spin_spiral(path)
        except ValueError:
            pending.append(path.name)
            continue
        record["file"] = path.name
        results.append(record)
    results.sort(key=lambda r: r.get("q_length", 0.0))
    return results, pending
