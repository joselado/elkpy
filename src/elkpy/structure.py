"""Structure: lattice vectors, species, and atomic positions.

Elk's public unit convention is atomic units throughout (Bohr, Hartree; see
docs/design.md #9) — avec and positions given directly are assumed to
already be in Bohr / fractional lattice coordinates. Use from_ase() for
Angstrom/Cartesian input, which converts.
"""

from collections import OrderedDict

BOHR_PER_ANGSTROM = 1.0 / 0.529177210903


def _normalize_atom(entry):
    """Accept either a bare (x, y, z) position, or a (position, bfcmt) pair
    -- the latter for atoms.py's bfcmt (muffin-tin external magnetic field,
    manual sec. 5.2), zero by default. Distinguished by length: a position
    is 3 scalars, a (position, bfcmt) pair is 2 length-3 tuples."""
    if len(entry) == 3 and all(isinstance(x, (int, float)) for x in entry):
        return tuple(entry), (0.0, 0.0, 0.0)
    position, bfcmt = entry
    return tuple(position), tuple(bfcmt)


class Structure:
    """A crystal: lattice vectors (avec, Bohr) plus species with fractional
    atomic positions (atposl, lattice coordinates).

    `species` is an ordered mapping of element symbol -> list of atoms,
    where each atom is either a bare (x, y, z) fractional position, or a
    ((x, y, z), (bx, by, bz)) pair giving a per-atom muffin-tin magnetic
    field (bfcmt) alongside the position -- e.g. for antiferromagnetic
    setups. e.g. {"Si": [(0, 0, 0), (0.25, 0.25, 0.25)]} or
    {"Fe": [((0, 0, 0), (0, 0, 1)), ((0.5, 0.5, 0.5), (0, 0, -1))]}.

    `species_files` optionally overrides the species filename looked up
    under `sppath` for one or more symbols (default: "{symbol}.in", the
    vendored species/ convention) -- e.g. {"Si": "Si_custom.in"}. The file
    must still exist under whatever sppath the Calculation resolves to;
    Elk itself only supports a single sppath directory, so a fully custom
    species file needs to live there (or sppath overridden to point at a
    directory containing it).
    """

    def __init__(self, avec, species, sppath=None, scale=1.0, species_files=None):
        self.avec = [tuple(v) for v in avec]
        self.species = OrderedDict(
            (symbol, [_normalize_atom(a) for a in atoms]) for symbol, atoms in species.items()
        )
        self.sppath = sppath
        self.scale = scale
        self.species_files = dict(species_files or {})

    def species_filename(self, symbol):
        return self.species_files.get(symbol, f"{symbol}.in")

    @classmethod
    def from_ase(cls, atoms, sppath=None):
        """Build a Structure from an ase.Atoms object (Angstrom, Cartesian
        cell/positions), converting to Elk's Bohr/fractional convention.
        ASE has no bfcmt equivalent -- all atoms get zero magnetic field."""
        cell_bohr = [
            tuple(c * BOHR_PER_ANGSTROM for c in row) for row in atoms.get_cell()
        ]
        species = OrderedDict()
        for symbol, frac in zip(atoms.get_chemical_symbols(), atoms.get_scaled_positions()):
            species.setdefault(symbol, []).append(tuple(frac))
        return cls(cell_bohr, species, sppath=sppath)

    def to_ase(self):
        try:
            from ase import Atoms
        except ImportError as exc:
            raise ImportError("to_ase() requires the optional 'ase' dependency") from exc
        cell_angstrom = [[c / BOHR_PER_ANGSTROM for c in row] for row in self.avec]
        symbols = []
        scaled_positions = []
        for symbol, atoms in self.species.items():
            for position, _bfcmt in atoms:
                symbols.append(symbol)
                scaled_positions.append(position)
        return Atoms(
            symbols=symbols, scaled_positions=scaled_positions, cell=cell_angstrom, pbc=True
        )

    def get_calculation(self, workdir, **params):
        from .calculation import Calculation

        return Calculation(self, workdir, **params)

    def atom_index(self, symbol, index=0):
        """0-based global atom index matching Elk's own atom ordering
        (species in declaration order, then atoms within each species in
        order -- the same convention get_forces() documents, and the order
        Calculation.get_atom_projection()'s per-atom matrices come back in).
        `index` is 0-based within `symbol`'s own atom list."""
        if symbol not in self.species:
            raise ValueError(f"species {symbol!r} not in structure (known: {sorted(self.species)})")
        if not (0 <= index < len(self.species[symbol])):
            raise ValueError(
                f"index {index} out of range for species {symbol!r} "
                f"({len(self.species[symbol])} atoms)"
            )
        offset = 0
        for s, atoms in self.species.items():
            if s == symbol:
                return offset + index
            offset += len(atoms)
