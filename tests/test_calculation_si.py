"""Integration tests running the real Elk binary on bulk Si.

Skipped if the binary hasn't been built (see docs/design.md #8: copy
vendor/elk/ to build/elk/, drop in build-config/make.inc, `make`). These are
slow (seconds per Elk invocation) by nature -- they exist to catch drift
against Elk's actual output formats, not to run on every keystroke.
"""

import pytest

from elkpy import config
from elkpy.structure import Structure

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)

SI_AVEC = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
SI_SPECIES = {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}
SI_BAND_VERTICES = [
    (0.0, 0.0, 1.0),
    (0.5, 0.5, 1.0),
    (0.0, 0.0, 0.0),
    (0.5, 0.0, 0.0),
]


@pytest.fixture
def si_calculation(tmp_path):
    s = Structure(SI_AVEC, SI_SPECIES)
    return s.get_calculation(
        tmp_path / "si", xc="PW", ngridk=(2, 2, 2), vkloff=(0.25, 0.5, 0.625)
    )


def test_get_energy(si_calculation):
    e = si_calculation.get_energy()
    # bulk Si total energy is a few hundred Hartree; loose bound, just
    # catching gross regressions (wrong sign, wrong units, parse failure)
    assert -600 < e < -550


def test_get_energy_is_cached(si_calculation):
    import time

    t0 = time.time()
    e1 = si_calculation.get_energy()
    first_call = time.time() - t0

    t0 = time.time()
    e2 = si_calculation.get_energy()
    second_call = time.time() - t0

    assert e1 == e2
    assert second_call < first_call / 5


def test_get_bands(si_calculation):
    distances, energies = si_calculation.get_bands(SI_BAND_VERTICES, npoints=20)
    assert distances.shape == (20,)
    assert energies.shape[1] == 20
    assert energies.shape[0] > 0


def test_get_dos(si_calculation):
    energies, dos = si_calculation.get_dos()
    assert energies.shape == dos.shape
    assert len(energies) > 0


def test_get_dos_with_different_ngridk_does_not_corrupt_ground_state(si_calculation):
    """Regression test: get_dos(ngridk=...) at a denser mesh than the ground
    state must not mutate self.workdir's STATE.OUT/manifest -- a later
    get_energy() call must still report the original ground state's energy,
    not one silently recomputed at the denser mesh (see _run_resumed's
    docstring in calculation.py)."""
    e_before = si_calculation.get_energy()
    si_calculation.get_dos(ngridk=(6, 6, 6))
    e_after = si_calculation.get_energy()
    assert e_before == e_after


def test_get_bands_with_kpath(si_calculation):
    """Symbolic k-path resolution (ASE special points) as an alternative to
    raw vertices; self-skips if ase isn't installed."""
    pytest.importorskip("ase")
    distances, energies = si_calculation.get_bands(kpath="GX", npoints=10)
    assert distances.shape == (10,)
    assert energies.shape[1] == 10


def test_get_bands_requires_exactly_one_of_vertices_or_kpath(si_calculation):
    with pytest.raises(ValueError):
        si_calculation.get_bands()
    with pytest.raises(ValueError):
        si_calculation.get_bands(vertices=SI_BAND_VERTICES, kpath="GX")


def test_extra_blocks_reach_elk_in_and_affect_ground_state_cache(tmp_path):
    s = Structure(SI_AVEC, SI_SPECIES)
    calc = s.get_calculation(
        tmp_path / "si", xc="PW", ngridk=(2, 2, 2), extra_blocks={"maxscl": [50]}
    )
    calc.get_energy()
    elk_in = (calc.workdir / "elk.in").read_text()
    assert "maxscl" in elk_in

    # a differently-configured extra_blocks must not reuse this ground state
    calc2 = s.get_calculation(
        tmp_path / "si", xc="PW", ngridk=(2, 2, 2), extra_blocks={"maxscl": [60]}
    )
    assert not calc2._ground_state_valid()


def test_converged_status(si_calculation):
    assert si_calculation.converged is None
    si_calculation.get_energy()
    assert si_calculation.converged is True


def test_run_tasks_different_blocks_do_not_collide(si_calculation):
    d1 = si_calculation.run_tasks([15], blocks={"kstlist": [[1, 1]]})
    d2 = si_calculation.run_tasks([15], blocks={"kstlist": [[1, 2]]})
    assert d1 != d2
    assert (d1 / "LSJ.OUT").exists()
    assert (d2 / "LSJ.OUT").exists()


def test_get_forces(si_calculation):
    f = si_calculation.get_forces()
    assert f.shape == (2, 3)
    # the input positions are already the DFT-PBE/PW equilibrium geometry
    assert abs(f).max() < 1e-3


def test_get_relaxed(si_calculation):
    relaxed = si_calculation.get_relaxed()
    e_relaxed = relaxed.get_energy()
    e_original = si_calculation.get_energy()
    # already at equilibrium (see test_get_forces) -- relaxation should not
    # change the energy by more than a tight numerical tolerance
    assert abs(e_relaxed - e_original) < 1e-5
    assert "Si" in relaxed.structure.species


def test_get_effective_mass(si_calculation):
    states = si_calculation.get_effective_mass((0.0, 0.0, 0.0))
    assert len(states) > 0
    first = states[0]
    assert first["tensor"].shape == (3, 3)
    # cubic symmetry at Gamma -> isotropic tensor
    diag = first["tensor"].diagonal()
    assert diag.max() - diag.min() < 1e-4


def test_get_density(si_calculation):
    points, density = si_calculation.get_density(grid=(4, 4, 4))
    assert points.shape == (64, 3)
    assert density.shape == (64,)
    assert (density >= 0).all()


def test_species_file_override(tmp_path):
    import shutil

    custom_species_dir = tmp_path / "custom_species"
    custom_species_dir.mkdir()
    shutil.copyfile(
        config.default_species_path() / "Si.in", custom_species_dir / "Si_custom.in"
    )
    s = Structure(
        SI_AVEC, SI_SPECIES, sppath=custom_species_dir, species_files={"Si": "Si_custom.in"}
    )
    calc = s.get_calculation(tmp_path / "si", xc="PW", ngridk=(2, 2, 2))
    e = calc.get_energy()
    assert -600 < e < -550
    elk_in = (calc.workdir / "elk.in").read_text()
    assert "Si_custom.in" in elk_in
