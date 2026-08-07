"""Integration tests for get_phonon_dispersion()/get_phonon_dos() (DFPT,
tasks 205/220/210) against the real Elk binary.

Split out from test_calculation_si.py because these are a different order of
magnitude slower than every other integration test here: even at the
smallest meaningful grid (ngridq=(2,2,2), a 2-atom cell), a single DFPT
phonon dispersion run took ~11 minutes and phonon DOS ~13 minutes on the
machine these were last verified on -- DFPT cost is dominated by the number
of atomic-displacement perturbations times q-points, not by anything elkpy
controls. Skipped by default; set ELKPY_RUN_SLOW_TESTS=1 to run them.
"""

import os

import pytest

from elkpy import config
from elkpy.structure import Structure

pytestmark = [
    pytest.mark.skipif(
        not config.default_elk_binary().is_file(),
        reason="elk binary not built; see docs/design.md #8",
    ),
    pytest.mark.skipif(
        os.environ.get("ELKPY_RUN_SLOW_TESTS") != "1",
        reason="phonon DFPT is slow (~10+ min per test); set ELKPY_RUN_SLOW_TESTS=1 to run",
    ),
]

SI_AVEC = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
SI_SPECIES = {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}


@pytest.fixture
def si_calculation(tmp_path):
    s = Structure(SI_AVEC, SI_SPECIES)
    return s.get_calculation(
        tmp_path / "si", xc="PW", ngridk=(2, 2, 2), vkloff=(0.25, 0.5, 0.625)
    )


def test_get_phonon_dispersion(si_calculation):
    distances, frequencies = si_calculation.get_phonon_dispersion(
        vertices=[(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)], ngridq=(2, 2, 2), npoints=5
    )
    # 2 atoms -> 6 branches
    assert frequencies.shape == (6, 5)
    # 3 acoustic branches -> ~0 frequency at Gamma (the first path point)
    gamma_frequencies = sorted(frequencies[:, 0])
    assert all(abs(f) < 1e-6 for f in gamma_frequencies[:3])


def test_get_phonon_dos(si_calculation):
    frequencies, dos = si_calculation.get_phonon_dos(ngridq=(2, 2, 2))
    assert frequencies.shape == dos.shape
    assert len(frequencies) > 0
