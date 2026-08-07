"""Integration test for per-atom bfcmt (docs/roadmap.md Tier 1 #5), which
needs spinpol and a magnetic structure -- bulk Si isn't a useful fixture for
this, so a separate bcc Fe setup lives here.

Skipped if the binary hasn't been built, same as test_calculation_si.py.
"""

import pytest

from elkpy import config
from elkpy.structure import Structure

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)

FE_AVEC = [(-2.71, 2.71, 2.71), (2.71, -2.71, 2.71), (2.71, 2.71, -2.71)]


def test_per_atom_bfcmt_reaches_elk_in(tmp_path):
    species = {
        "Fe": [
            ((0.0, 0.0, 0.0), (0.0, 0.0, 4.0)),
            ((0.5, 0.5, 0.5), (0.0, 0.0, -4.0)),
        ]
    }
    s = Structure(FE_AVEC, species)
    calc = s.get_calculation(tmp_path / "fe", xc="PW", spinpol=True, ngridk=(2, 2, 2))
    e = calc.get_energy()
    # bcc Fe total energy is a few thousand Hartree; loose bound, just
    # catching gross regressions
    assert -3000 < e < -2000

    elk_in = (calc.workdir / "elk.in").read_text()
    assert "4.0000000000" in elk_in
    assert "-4.0000000000" in elk_in
