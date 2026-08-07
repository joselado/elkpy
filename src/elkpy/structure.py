"""Structure: lattice vectors, species, and atomic positions.

Elk's public unit convention is atomic units throughout (Bohr, Hartree; see
docs/design.md #9) — avec and positions given directly are assumed to
already be in Bohr / fractional lattice coordinates. Use from_ase() for
Angstrom/Cartesian input, which converts.
"""

from collections import OrderedDict

BOHR_PER_ANGSTROM = 1.0 / 0.529177210903


class Structure:
    """A crystal: lattice vectors (avec, Bohr) plus species with fractional
    atomic positions (atposl, lattice coordinates).

    `species` is an ordered mapping of element symbol -> list of (x, y, z)
    fractional positions, e.g. {"Si": [(0, 0, 0), (0.25, 0.25, 0.25)]}.
    """

    def __init__(self, avec, species, sppath=None, scale=1.0):
        self.avec = [tuple(v) for v in avec]
        self.species = OrderedDict(species)
        self.sppath = sppath
        self.scale = scale

    @classmethod
    def from_ase(cls, atoms, sppath=None):
        """Build a Structure from an ase.Atoms object (Angstrom, Cartesian
        cell/positions), converting to Elk's Bohr/fractional convention."""
        cell_bohr = [
            tuple(c * BOHR_PER_ANGSTROM for c in row) for row in atoms.get_cell()
        ]
        species = OrderedDict()
        for symbol, frac in zip(atoms.get_chemical_symbols(), atoms.get_scaled_positions()):
            species.setdefault(symbol, []).append(tuple(frac))
        return cls(cell_bohr, species)

    def to_ase(self):
        try:
            from ase import Atoms
        except ImportError as exc:
            raise ImportError("to_ase() requires the optional 'ase' dependency") from exc
        cell_angstrom = [[c / BOHR_PER_ANGSTROM for c in row] for row in self.avec]
        symbols = []
        scaled_positions = []
        for symbol, positions in self.species.items():
            for pos in positions:
                symbols.append(symbol)
                scaled_positions.append(pos)
        return Atoms(
            symbols=symbols, scaled_positions=scaled_positions, cell=cell_angstrom, pbc=True
        )

    def get_calculation(self, workdir, **params):
        from .calculation import Calculation

        return Calculation(self, workdir, **params)
