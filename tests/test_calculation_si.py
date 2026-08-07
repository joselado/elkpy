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
