import pytest

from elkpy.structure import Structure

SI_AVEC = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
SI_SPECIES = {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}


def test_construct():
    s = Structure(SI_AVEC, SI_SPECIES)
    assert s.avec == SI_AVEC
    assert list(s.species["Si"]) == SI_SPECIES["Si"]


def test_from_ase_to_ase_roundtrip():
    ase = pytest.importorskip("ase")
    from ase.build import bulk

    atoms = bulk("Si", "diamond", a=5.43)
    s = Structure.from_ase(atoms)
    assert s.species.keys() == {"Si"}
    assert len(s.species["Si"]) == 2

    back = s.to_ase()
    assert back.get_chemical_symbols() == atoms.get_chemical_symbols()
    assert back.cell.array == pytest.approx(atoms.cell.array, rel=1e-8)
