"""Integration tests for the two medium-severity findings that need a real
Elk run (docs/review_findings.md 4 and 6).

Both are cheap by the standards of this directory: LiF is two atoms and the
stress comparison is two cubic cells, whose strain basis has one member.
"""

import numpy as np
import pytest

from elkpy import config
from elkpy.calculation import Calculation
from elkpy.structure import Structure

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)

# rocksalt LiF, a = 7.61 Bohr. Li's species file flags NO state spcore and
# F's flags the 1s -- which is the whole point of the fixture.
LIF_AVEC = [(3.805, 3.805, 0.0), (3.805, 0.0, 3.805), (0.0, 3.805, 3.805)]
LIF_SPECIES = {"Li": [(0.0, 0.0, 0.0)], "F": [(0.5, 0.5, 0.5)]}


def test_core_wavefunctions_skip_a_species_with_no_core_state(tmp_path):
    """wfcrplot.f90:21 opens the file unconditionally but writes only inside
    `if (spcore(ist,is))`, so lithium's is 0 bytes while fluorine's holds its
    1s. Parsing every file unconditionally killed the whole call and threw
    away the heavy atom's data, which had been computed correctly."""
    calc = Calculation(
        Structure(LIF_AVEC, LIF_SPECIES), tmp_path / "lif",
        xc="PW", ngridk=(2, 2, 2), rgkmax=6.0,
    )
    cores = calc.get_core_wavefunctions()

    assert ("Li", 1) not in cores          # no spcore state: absent, not an error
    r, u = cores[("F", 1)]
    assert u.shape[0] == 1                 # F.in flags exactly the 1s
    assert r.shape == (u.shape[1],)
    # Elk stores u = r R(r), so int |u|^2 dr = 1 with no r^2 Jacobian
    assert np.trapezoid(u[0] ** 2, r) == pytest.approx(1.0, abs=5e-3)


def test_stress_pressure_is_independent_of_how_the_cell_is_scaled(tmp_path):
    """readinput.f90:2275 multiplies avec by `scale` before any physics, so
    the pressure must come from the SCALED vectors. Using the raw ones is a
    factor of scale**2 -- 105x for silicon written the conventional way --
    and the isotropy guard cannot see it, avec/||avec|| being scale-invariant.

    The same crystal is given twice: once with the lattice vectors already in
    Bohr, once as the conventional fractional vectors with scale carrying the
    lattice constant."""
    a = 10.26
    fractional = [(0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]
    absolute = [tuple(a * x for x in row) for row in fractional]
    atoms = {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}

    plain = Calculation(
        Structure(absolute, atoms), tmp_path / "si_plain",
        xc="PW", ngridk=(2, 2, 2), rgkmax=6.0,
    ).get_stress()
    scaled = Calculation(
        Structure(fractional, atoms, scale=a), tmp_path / "si_scaled",
        xc="PW", ngridk=(2, 2, 2), rgkmax=6.0,
    ).get_stress()

    assert plain["pressure"] is not None and scaled["pressure"] is not None
    # same crystal, same Elk run: the stresses are identical and the pressure
    # must be too. Before the fix this differed by a**2 = 105.
    np.testing.assert_allclose(scaled["stress"], plain["stress"], rtol=1e-6)
    assert scaled["pressure"] == pytest.approx(plain["pressure"], rel=1e-6)
