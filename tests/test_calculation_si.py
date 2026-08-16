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
    assert calc2._load_valid_manifest() is None


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


def test_get_effective_mass_tensor_is_inverse_of_derivative_tensor(si_calculation):
    """Regression test: EFFMASS.OUT has two distinct 3x3 blocks per state --
    the raw eigenvalue-derivative matrix and its inverse, the actual
    effective mass tensor. `tensor` must be the latter, not the former (see
    parsers/effmass.py)."""
    import numpy as np

    first = si_calculation.get_effective_mass((0.0, 0.0, 0.0))[0]
    assert not np.allclose(first["tensor"], first["derivative_tensor"])
    product = first["tensor"] @ first["derivative_tensor"]
    assert np.allclose(product, np.eye(3), atol=1e-3)


def test_get_density(si_calculation):
    points, density = si_calculation.get_density(grid=(4, 4, 4))
    assert points.shape == (64, 3)
    assert density.shape == (64,)
    assert (density >= 0).all()


def test_get_potential(si_calculation):
    """Both components of the Kohn-Sham potential (task 43, VCL3D.OUT and
    VXC3D.OUT) on the same plot3d grid as get_density()."""
    import numpy as np

    points, v_coulomb = si_calculation.get_potential(grid=(4, 4, 4))
    assert points.shape == (64, 3)
    assert v_coulomb.shape == (64,)

    points_xc, v_xc = si_calculation.get_potential(grid=(4, 4, 4), component="xc")
    assert np.allclose(points, points_xc)

    # the two components are genuinely different files, not the same one
    # parsed twice (a plausible spec.py/filename mix-up)
    assert not np.allclose(v_coulomb, v_xc)


def test_get_potential_xc_matches_the_lda_functional_of_the_density(si_calculation):
    """Sharp cross-check tying VXC3D.OUT to RHO3D.OUT: xc='PW' is a LOCAL
    density functional, so v_xc(r) is a pointwise function of n(r) alone,
    dominated by Dirac exchange v_x = -(3n/pi)^(1/3) with the Perdew-Wang
    correlation potential adding a smaller negative correction on top.

    Catches what a shape/sign check cannot: a wrong file, a shifted column,
    or a unit error would break this pointwise relation immediately, and it
    compares two separate Elk runs' output (task 33 vs task 43) against an
    analytic formula rather than against each other."""
    import numpy as np

    points_rho, density = si_calculation.get_density(grid=(4, 4, 4))
    points_v, v_xc = si_calculation.get_potential(grid=(4, 4, 4), component="xc")
    assert np.allclose(points_rho, points_v)

    v_dirac = -((3 * density / np.pi) ** (1 / 3))
    ratio = v_xc / v_dirac
    # correlation makes v_xc 10-25% deeper than bare exchange across this
    # cell's density range (0.006 to 0.085 e/Bohr^3)
    assert (ratio > 1.0).all() and (ratio < 1.4).all()


def test_get_potential_rejects_unknown_component(si_calculation):
    with pytest.raises(ValueError):
        si_calculation.get_potential(grid=(2, 2, 2), component="hartree")


def test_get_elf(si_calculation):
    """ELF (task 53, ELF3D.OUT), same plot3d grid convention as
    get_density(). f_ELF = 1/(1 + (D/D0)^2) is bounded to [0, 1] by
    construction, and bulk Si's covalent bonds push it close to the upper
    bound somewhere in the cell -- a uniform-electron-gas-like result
    (everything near 1/2) would mean the bonding information was lost."""
    points, elf = si_calculation.get_elf(grid=(4, 4, 4))
    assert points.shape == (64, 3)
    assert elf.shape == (64,)
    assert (elf >= 0).all() and (elf <= 1).all()
    assert elf.max() > 0.8


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

    # regression: get_relaxed() must recover the real element symbol ("Si"),
    # not the filename stem ("Si_custom") -- see parsers/geometry.py
    relaxed = calc.get_relaxed()
    assert "Si" in relaxed.structure.species
    relaxed.structure.to_ase()  # raises if the species key isn't a real symbol


def test_get_relaxed_does_not_double_apply_scale(tmp_path):
    """Regression test: writegeom.f90 always bakes any scale factor into the
    avec it writes to GEOMETRY_OPT.OUT, so get_relaxed() must not reapply
    the original Structure's scale on top of that (see parsers/geometry.py's
    docstring and get_relaxed()'s scale=1.0 comment)."""
    # scale=2.565 with unit-ish direction vectors reproduces the same
    # physical cell as SI_AVEC (already in Bohr, scale=1.0 implicitly)
    s_scaled = Structure(
        [(2.0, 2.0, 0.0), (2.0, 0.0, 2.0), (0.0, 2.0, 2.0)], SI_SPECIES, scale=2.565
    )
    calc_scaled = s_scaled.get_calculation(tmp_path / "si_scaled", xc="PW", ngridk=(2, 2, 2))
    e_scaled = calc_scaled.get_energy()

    s_unscaled = Structure(SI_AVEC, SI_SPECIES)
    calc_unscaled = s_unscaled.get_calculation(tmp_path / "si_unscaled", xc="PW", ngridk=(2, 2, 2))
    e_unscaled = calc_unscaled.get_energy()
    # sanity check that the two setups really are the same physical cell
    assert abs(e_scaled - e_unscaled) < 1e-6

    relaxed = calc_scaled.get_relaxed()
    e_relaxed = relaxed.get_energy()
    # a scale^2 bug would blow this up to a wildly different (much smaller
    # cell -> much more negative, or a crash) energy, not a tiny numerical
    # relaxation shift
    assert abs(e_relaxed - e_scaled) < 1e-4


def test_nonconvergence_raises_on_cache_hit_too(tmp_path):
    """Regression test: ensure_ground_state()'s cache-hit path must apply
    raise_on_nonconvergence just like the fresh-run path -- previously it
    silently returned from a cached non-converged run instead of raising."""
    s = Structure(SI_AVEC, SI_SPECIES)
    workdir = tmp_path / "si_nonconv"

    calc = s.get_calculation(workdir, xc="PW", ngridk=(2, 2, 2), extra_blocks={"maxscl": [1]})
    with pytest.raises(RuntimeError):
        calc.get_energy()

    # re-instantiate on the same workdir -- manifest + STATE.OUT already on
    # disk with a matching (non-converged) basis signature: the cache-hit path
    calc2 = s.get_calculation(workdir, xc="PW", ngridk=(2, 2, 2), extra_blocks={"maxscl": [1]})
    with pytest.raises(RuntimeError):
        calc2.get_energy()

    # raise_on_nonconvergence=False must still allow inspecting .converged
    calc3 = s.get_calculation(
        workdir, xc="PW", ngridk=(2, 2, 2), extra_blocks={"maxscl": [1]},
        raise_on_nonconvergence=False,
    )
    calc3.get_energy()
    assert calc3.converged is False
