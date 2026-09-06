"""Parsers for the strain/stress and polarisation-derivative response
tables: STRAIN.OUT (task 430), STRESS.OUT (task 440), PIEZOELT.OUT (task
380) and MAGNETOELT.OUT (task 390).

All four share one idea. Elk does not evaluate an analytic stress or
piezoelectric tensor; it builds an orthonormal *basis of strain tensors*
e_k (``src/genstrain.f90``) and then differentiates a scalar or a vector
numerically along each of them. ``src/strainabg.f90`` applies the strain as

    A -> A + deltast * e_k,

with A the 3x3 lattice-vector matrix, so the "component index" k in every
one of these files labels a member of that basis rather than a Cartesian
(ij) pair. The basis itself is therefore part of the answer, and each file
prints it alongside the derivative -- which is why every parser here
returns the strain tensors too.

Two conventions worth stating explicitly, both read off the Fortran rather
than the manual:

- **Row meaning.** ``src/writestress.f90`` writes each strain tensor as
  ``write(50,'(3G18.10)') (strain(i,j,k),i=1,3)`` inside ``do j=1,3``, i.e.
  printed line j holds the three *Cartesian* components of column j of
  Elk's ``strain`` array. Since ``strainabg`` adds that array to ``avec``
  and Elk stores lattice vector j in ``avec(1:3,j)``, printed line j is the
  Cartesian displacement added to lattice vector j. elkpy's
  ``Structure.avec`` is a list of lattice vectors (rows), so the parsed
  matrix indexes as ``[j, cartesian]`` and lines up with ``avec`` directly,
  with no transpose.
- **The first strain tensor is isotropic**: ``genstrain`` sets
  ``strain(:,:,1) = avec/||avec||_F`` before orthogonalising anything else
  against it (``writestrain.f90`` says so in its own info line). For tasks
  430/440 the ``latvopt`` branch that can remove it applies only to tasks
  2/3, so component 1 of a stress calculation is always the isotropic one
  -- which is what makes the hydrostatic pressure in
  ``pressure_from_stress()`` well defined.

The headers of STRESS.OUT/PIEZOELT.OUT contain non-ASCII characters (the
literal strings ``"     A → A + e_k dt,"`` and ``dP_i/dt`` with subscript
characters are in the Fortran source), so every file here is opened as
UTF-8.
"""

import numpy as np


def _lines(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [line.rstrip("\n") for line in fh]


def _matrix(lines, i):
    """Three consecutive `(3G18.10)` rows starting at index i."""
    return np.array([[float(x) for x in lines[i + r].split()] for r in range(3)])


def _after_colon(line):
    return line.split(":", 1)[1]


def parse_strain(strain_out_path):
    """Parse STRAIN.OUT (task 430, src/writestrain.f90).

    Written as, per tensor k::

        write(50,*)
        write(50,'("Strain tensor : ",I0)') k
        do j=1,3
          write(50,'(3G18.10)') (strain(i,j,k),i=1,3)
        end do

    Returns a list of (3,3) float arrays, one per strain-basis tensor, in
    Elk's own order (index 0 = the isotropic-expansion tensor).
    """
    lines = _lines(strain_out_path)
    tensors = []
    for i, line in enumerate(lines):
        if line.strip().startswith("Strain tensor :"):
            tensors.append(_matrix(lines, i + 1))
    if not tensors:
        raise ValueError(f"no 'Strain tensor :' blocks found in {strain_out_path}")
    return tensors


def parse_stress(stress_out_path):
    """Parse STRESS.OUT (task 440, src/writestress.f90).

    Written as, per component k::

        write(50,'("Strain tensor k : ",I1)') k
        do j=1,3
          write(50,'(3G18.10)') (strain(i,j,k),i=1,3)
        end do
        write(50,'("Stress : ",G18.10)') stress(k)

    with ``stress(k) = (engytot(strained) - engytot(unstrained))/deltast``
    from ``src/genstress.f90`` -- a total-energy derivative in Hartree per
    unit of the dimensionless strain parameter t, NOT a Cartesian stress
    tensor in pressure units.

    Returns {"strain": [(3,3), ...], "stress": (nstrain,) array}.
    """
    lines = _lines(stress_out_path)
    strains, stress = [], []
    for i, line in enumerate(lines):
        if line.strip().startswith("Strain tensor k :"):
            strains.append(_matrix(lines, i + 1))
        elif line.strip().startswith("Stress :"):
            stress.append(float(_after_colon(line)))
    if not stress:
        raise ValueError(f"no 'Stress :' entries found in {stress_out_path}")
    if len(strains) != len(stress):
        raise ValueError(
            f"{stress_out_path}: {len(strains)} strain tensors but {len(stress)} "
            f"stress values"
        )
    return {"strain": strains, "stress": np.array(stress)}


def pressure_from_stress(stress, avec):
    """Hydrostatic pressure in Hartree/Bohr^3 from component 1 of a
    ``parse_stress()`` result and the (unstrained) lattice vectors.

    Elk's first strain tensor is e_1 = A/||A||_F (``genstrain.f90``), so
    the strained lattice is A(t) = A + t A/||A||_F = A (1 + t/||A||_F): a
    pure isotropic scaling. The cell volume is then

        V(t) = V0 (1 + t/||A||_F)^3,   dV/dt|_0 = 3 V0 / ||A||_F,

    and since ``stress[0] = dE/dt``,

        P = -dE/dV = -stress[0] * ||A||_F / (3 V0).

    ``avec`` is the elkpy row-vector lattice (Bohr); ||A||_F is the
    Frobenius norm of the 3x3 matrix, matching ``genstrain``'s
    ``norm2(avec(1:3,1:3))``, and V0 = |det A|. Positive P means the cell
    wants to expand.
    """
    avec = np.asarray(avec, dtype=float)
    volume = abs(np.linalg.det(avec))
    frobenius = np.linalg.norm(avec)
    return -float(stress[0]) * frobenius / (3.0 * volume)


def _vector_after_colon(line):
    return np.array([float(x) for x in _after_colon(line).split()])


def parse_piezoelectric(piezoelt_out_path):
    """Parse PIEZOELT.OUT (task 380, src/piezoelt.f90).

    Per strain component k the file holds the strain tensor followed by::

        write(50,'("Piezoelectric tensor components dP_i/dt, i=1...3 :")')
        write(50,'(" lattice coordinates : ",3G18.10)') pelt(:,k)
        write(50,'(" Cartesian coordinates : ",3G18.10)') vc
        write(50,'("  length : ",G18.10)') norm2(vc(1:3))

    where ``pelt`` is the Berry-phase polarisation (``src/polar.f90``,
    King-Smith and Vanderbilt) differenced across the strain step. Returns
    a list of dicts, one per strain component:

        {"strain": (3,3), "lattice": (3,), "cartesian": (3,), "length": float}
    """
    lines = _lines(piezoelt_out_path)
    entries = []
    current = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("Strain tensor k :"):
            current = {"strain": _matrix(lines, i + 1)}
            entries.append(current)
        elif current is None:
            continue
        elif s.startswith("lattice coordinates :"):
            current["lattice"] = _vector_after_colon(line)
        elif s.startswith("Cartesian coordinates :"):
            current["cartesian"] = _vector_after_colon(line)
        elif s.startswith("length :"):
            current["length"] = float(_after_colon(line))
    if not entries:
        raise ValueError(f"no strain blocks found in {piezoelt_out_path}")
    return entries


def parse_magnetoelectric(magnetoelt_out_path):
    """Parse MAGNETOELT.OUT (task 390, src/magnetoelt.f90).

    One block per Cartesian magnetic-field component j::

        write(50,'("Magnetic field Cartesian component j : ",I1)') j
        write(50,'("Magnetoelectric tensor components dP_i/dB_j, i=1...3")')
        write(50,'(" lattice coordinates : ",3G18.10)') felt(:,j)
        write(50,'(" Cartesian coordinates : ",3G18.10)') vc
        write(50,'("  length : ",G18.10)') norm2(vc(1:3))

    Returns {"lattice": (3,3), "cartesian": (3,3), "length": (3,)}, each
    matrix indexed ``[j, i]`` -- row j is the polarisation derivative with
    respect to field component j, so ``cartesian[j, i] = dP_i/dB_j``.
    """
    lines = _lines(magnetoelt_out_path)
    lattice, cartesian, length = [], [], []
    for line in lines:
        s = line.strip()
        if s.startswith("lattice coordinates :"):
            lattice.append(_vector_after_colon(line))
        elif s.startswith("Cartesian coordinates :"):
            cartesian.append(_vector_after_colon(line))
        elif s.startswith("length :"):
            length.append(float(_after_colon(line)))
    if len(lattice) != 3 or len(cartesian) != 3 or len(length) != 3:
        raise ValueError(
            f"{magnetoelt_out_path}: expected 3 field components, got "
            f"{len(lattice)}/{len(cartesian)}/{len(length)}"
        )
    return {
        "lattice": np.array(lattice),
        "cartesian": np.array(cartesian),
        "length": np.array(length),
    }
