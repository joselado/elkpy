"""Parsers for the two per-nucleus hyperfine observables Elk writes:
EFG.OUT (task 115, src/writeefg.f90) and MOSSBAUER.OUT (task 110,
src/mossbauer.f90).

Both files are organised the same way -- one block per atom, opened by a
line written as

    write(50,'("Species : ",I4," (",A,"), atom : ",I4)') is,trim(spsymb(is)),ia

-- and the atom order is Elk's own (species in input order, atoms within
each species in input order), i.e. the same order ``parsers.forces`` uses.

The physics differs, though:

- The **electric field gradient** is the second derivative of the Coulomb
  potential at the nucleus with the l=m=0 part removed in the muffin tin,
  V_ij = d^2 V'_C / dr_i dr_j |_{r=r_alpha}, in Cartesian coordinates and
  atomic units (Hartree/Bohr^2). It is what nuclear quadrupole resonance
  and Moessbauer quadrupole splittings measure, via the coupling
  eQ V_zz / h. Elk symmetrises it and prints the eigenvalues from
  ``dsyev``, i.e. in ascending order, not the crystallographic
  |V_zz| >= |V_yy| >= |V_xx| convention -- reorder yourself if you want
  the asymmetry parameter eta = (V_xx - V_yy)/V_zz.
- The **Moessbauer** block gives the contact charge density (the l=m=0
  muffin-tin density sampled at, and averaged over, the nuclear and
  Thomson radii; Bohr^-3) which fixes the isomer shift, and -- only when
  the run is spin polarised -- the Fermi contact hyperfine field, printed
  both in atomic units and in tesla. The dipolar terms appear only when
  ``tbdip`` was set, so both extra sections are optional and this parser
  simply omits the keys that are absent.
"""

import numpy as np

_SPECIES_PREFIX = "Species :"


def _lines(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [line.rstrip("\n") for line in fh]


def _parse_species_header(line):
    """'Species :    1 (Si), atom :    4' -> (1, 'Si', 4)."""
    head, tail = line.split(",", 1)
    species_part = head.split(":", 1)[1]
    index = int(species_part.split("(", 1)[0])
    symbol = species_part.split("(", 1)[1].rsplit(")", 1)[0]
    atom = int(tail.split(":", 1)[1])
    return index, symbol.strip(), atom


def _blocks(lines):
    """Split a file into (header_line, body_lines) pairs, one per atom."""
    starts = [i for i, line in enumerate(lines) if line.strip().startswith(_SPECIES_PREFIX)]
    for n, i in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        yield lines[i], lines[i + 1 : end]


def parse_efg(efg_out_path):
    """Parse EFG.OUT (task 115, src/writeefg.f90).

    Per atom the file holds::

        write(50,'(" EFG tensor :")')
        do i=1,3
          write(50,'(3G18.10)') (efg(i,j),j=1,3)
        end do
        write(50,'(" trace : ",G18.10)') efg(1,1)+efg(2,2)+efg(3,3)
        write(50,'(" eigenvalues :")')
        write(50,'(3G18.10)') w

    Note the tensor rows here are genuine rows (``j`` is the inner index),
    unlike the strain tensors in ``parsers.stress``; the tensor is
    symmetrised by ``writeefg`` before printing anyway.

    Returns a list of dicts, one per atom, in Elk's atom order:

        {"species": int, "symbol": str, "atom": int,
         "tensor": (3,3) array, "trace": float, "eigenvalues": (3,) array}

    Units: Hartree/Bohr^2 (atomic units), Cartesian axes.
    """
    lines = _lines(efg_out_path)
    results = []
    for header, body in _blocks(lines):
        species, symbol, atom = _parse_species_header(header)
        entry = {"species": species, "symbol": symbol, "atom": atom}
        for i, line in enumerate(body):
            s = line.strip()
            if s.startswith("EFG tensor :"):
                entry["tensor"] = np.array(
                    [[float(x) for x in body[i + 1 + r].split()] for r in range(3)]
                )
            elif s.startswith("trace :"):
                entry["trace"] = float(line.split(":", 1)[1])
            elif s.startswith("eigenvalues :"):
                entry["eigenvalues"] = np.array([float(x) for x in body[i + 1].split()])
        results.append(entry)
    if not results:
        raise ValueError(f"no atom blocks found in {efg_out_path}")
    return results


# labels as written by src/mossbauer.f90, mapped to short keys
_CONTACT_DENSITY_KEYS = {
    "at nuclear center": "nuclear_center",
    "at nuclear surface": "nuclear_surface",
    "average in nuclear volume": "nuclear_volume_average",
    "at Thomson radius": "thomson_radius",
    "average in Thomson volume": "thomson_volume_average",
}


def parse_mossbauer(mossbauer_out_path):
    """Parse MOSSBAUER.OUT (task 110, src/mossbauer.f90).

    Per atom, always present::

        write(50,'(" approximate nuclear radius : ",G18.10)') rnucl(is)
        write(50,'(" number of mesh points to nuclear radius : ",I6)') nrn
        write(50,'(" Thomson radius : ",G18.10)') rtmsn(is)
        write(50,'(" number of mesh points to Thomson radius : ",I6)') nrt
        write(50,'(" Contact density :")')
        write(50,'("  at nuclear center         : ",G18.10)') r0
        ... (five labelled densities)

    Present only when ``spinpol``::

        write(50,'(" Contact average in nuclear volume :")')
        write(50,'("  moment (mu_B) : ",3G18.10)') mn(1:ndmag)
        write(50,'("  magnetic field : ",3G18.10)') bn(1:ndmag)
        write(50,'("   tesla : ",3G18.10)') b_si*bn(1:ndmag)
        ... and the same for the Thomson volume

    Present only when ``spinpol .and. tbdip``, the spin (and, with ``tjr``,
    orbital) dipole field::

        write(50,'(" Average dipole field in nuclear volume :")')
        write(50,'("  spin : ",3G18.10)') bn        (or "spin and orbital")
        write(50,'("   tesla : ",3G18.10)') b_si*bn

    Returns a list of dicts, one per atom, in Elk's atom order. Keys always
    present: ``species``, ``symbol``, ``atom``, ``nuclear_radius``,
    ``nuclear_mesh_points``, ``thomson_radius``, ``thomson_mesh_points``,
    ``contact_density`` (a dict with the five keys above, Bohr^-3).
    Present only for a spin-polarised run: ``contact_nuclear`` and
    ``contact_thomson``, each ``{"moment", "field", "field_tesla"}`` with
    ``ndmag``-length arrays (moment in mu_B, field in atomic units and in
    tesla). Present only with ``tbdip``: ``dipole_nuclear`` and
    ``dipole_thomson``, each ``{"field", "field_tesla", "includes_orbital"}``.

    Note the file's own closing remark: with ``tbdip`` the contact term is
    implicitly contained in the spin dipole field, so the two sections are
    not independent and need not agree exactly.
    """
    lines = _lines(mossbauer_out_path)
    results = []
    for header, body in _blocks(lines):
        species, symbol, atom = _parse_species_header(header)
        entry = {
            "species": species,
            "symbol": symbol,
            "atom": atom,
            "contact_density": {},
        }
        section = None
        for line in body:
            s = line.strip()
            if not s or ":" not in s:
                continue
            label, _, value = s.partition(":")
            label, value = label.strip(), value.strip()
            if label == "approximate nuclear radius":
                entry["nuclear_radius"] = float(value)
            elif label == "number of mesh points to nuclear radius":
                entry["nuclear_mesh_points"] = int(value)
            elif label == "Thomson radius":
                entry["thomson_radius"] = float(value)
            elif label == "number of mesh points to Thomson radius":
                entry["thomson_mesh_points"] = int(value)
            elif label == "Contact density":
                section = "contact_density"
            elif label == "Contact average in nuclear volume":
                section = entry.setdefault("contact_nuclear", {})
            elif label == "Contact average in Thomson volume":
                section = entry.setdefault("contact_thomson", {})
            elif label == "Average dipole field in nuclear volume":
                section = entry.setdefault("dipole_nuclear", {})
            elif label == "Average dipole field in Thomson volume":
                section = entry.setdefault("dipole_thomson", {})
            elif section == "contact_density" and label in _CONTACT_DENSITY_KEYS:
                entry["contact_density"][_CONTACT_DENSITY_KEYS[label]] = float(value)
            elif isinstance(section, dict):
                numbers = np.array([float(x) for x in value.split()])
                if label == "moment (mu_B)":
                    section["moment"] = numbers
                elif label == "magnetic field":
                    section["field"] = numbers
                elif label == "tesla":
                    section["field_tesla"] = numbers
                elif label in ("spin", "spin and orbital"):
                    section["field"] = numbers
                    section["includes_orbital"] = label == "spin and orbital"
        results.append(entry)
    if not results:
        raise ValueError(f"no atom blocks found in {mossbauer_out_path}")
    return results
