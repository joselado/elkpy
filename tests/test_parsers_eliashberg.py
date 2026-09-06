"""Unit tests for parsers/eliashberg.py.

Fixtures are transcribed from the `write` statements in
vendor/elk/src/alpha2f.f90 and vendor/elk/src/eliashberg.f90, cited inline.
MCMILLAN.OUT is stronger than format-derived: the fixture is the verbatim
file Elk ships with its own bcc-Nb example, at
vendor/elk/examples/phonons-superconductivity/Nb-DFPT/MCMILLAN.OUT --
including the literal UTF-8 lambda and mu characters in the labels.
"""

from pathlib import Path

import numpy as np
import pytest

from elkpy import config
from elkpy.parsers import eliashberg


# The real file Elk ships. Kept as a path rather than a copied string so a
# vendored-Elk version bump that changes the format breaks this test.
NB_MCMILLAN = (
    config.repo_root()
    / "vendor/elk/examples/phonons-superconductivity/Nb-DFPT/MCMILLAN.OUT"
)


# --------------------------------------------------------------------------
# ALPHA2F.OUT / MCMILLAN.OUT (task 250)
# --------------------------------------------------------------------------

def test_parse_alpha2f(tmp_path):
    # src/alpha2f.f90: write(50,'(2G18.10)') w(iw),a2f(iw)
    path = tmp_path / "ALPHA2F.OUT"
    path.write_text(
        " -0.1000000000E-04  0.000000000    \n"
        "  0.1000000000E-03  0.5000000000    \n"
        "  0.2000000000E-03  0.1250000000E+01\n"
    )
    frequencies, a2f = eliashberg.parse_alpha2f(path)
    np.testing.assert_allclose(frequencies, [-1e-5, 1e-4, 2e-4])
    np.testing.assert_allclose(a2f, [0.0, 0.5, 1.25])
    # alpha2f.f90 pads its own grid by 10% below the minimum phonon
    # frequency, so a slightly negative first point is normal, not corruption
    assert frequencies[0] < 0.0


@pytest.mark.skipif(not NB_MCMILLAN.is_file(), reason="vendored Elk example missing")
def test_parse_mcmillan_against_shipped_nb_example():
    values = eliashberg.parse_mcmillan(NB_MCMILLAN)
    assert values["lambda"] == pytest.approx(1.053424300)
    assert values["wlog"] == pytest.approx(0.6370229177e-03)
    assert values["wrms"] == pytest.approx(0.6993860014e-03)
    assert values["mustar"] == pytest.approx(0.15)
    assert values["tc"] == pytest.approx(12.44672586)
    # sanity on the physics the parser is carrying: Nb's measured T_c is
    # 9.2 K, w_rms >= w_log always (Cauchy-Schwarz on the same weight), and
    # lambda ~ 1 is the textbook intermediate-coupling value for Nb
    assert values["wrms"] >= values["wlog"]
    assert 5.0 < values["tc"] < 25.0


def test_parse_mcmillan_reports_missing_entries(tmp_path):
    path = tmp_path / "MCMILLAN.OUT"
    path.write_text("\nElectron-phonon coupling constant, l :    1.0\n")
    with pytest.raises(ValueError, match="missing entries"):
        eliashberg.parse_mcmillan(path)


def test_parse_mcmillan_survives_non_ascii_labels(tmp_path):
    """The shipped file has literal Greek characters in two labels; the
    parser must key on the ASCII part and open as UTF-8."""
    path = tmp_path / "MCMILLAN.OUT"
    path.write_text(
        "\nElectron-phonon coupling constant, λ :    2.0    \n"
        "\nLogarithmic average frequency :   0.1E-02\n"
        "\nRMS average frequency :   0.2E-02\n"
        "\nCoulomb pseudopotential, μ* :   0.1000000000    \n"
        "\nMcMillan-Allen-Dynes superconducting critical temperature\n"
        " [Eq. 34, Phys. Rev. B 12, 905 (1975)] (kelvin) :    30.0    \n",
        encoding="utf-8",
    )
    values = eliashberg.parse_mcmillan(path)
    assert values["lambda"] == pytest.approx(2.0)
    assert values["mustar"] == pytest.approx(0.1)
    assert values["tc"] == pytest.approx(30.0)


# --------------------------------------------------------------------------
# ELIASHBERG_*.OUT (task 260)
# --------------------------------------------------------------------------

def test_parse_gap_vs_temperature(tmp_path):
    # src/eliashberg.f90: write(64,'(3G18.10)') temp,d(0),z(0)
    path = tmp_path / "ELIASHBERG_GAP_T.OUT"
    path.write_text(
        "   2.000000000       0.5000000000E-03   2.000000000    \n"
        "   4.000000000       0.3000000000E-03   1.900000000    \n"
        "   6.000000000       0.1000000000E-03   1.800000000    \n"
    )
    temperatures, gap, z = eliashberg.parse_gap_vs_temperature(path)
    np.testing.assert_allclose(temperatures, [2.0, 4.0, 6.0])
    np.testing.assert_allclose(gap, [5e-4, 3e-4, 1e-4])
    np.testing.assert_allclose(z, [2.0, 1.9, 1.8])
    # the gap closes with rising temperature; that monotonicity is the whole
    # content of the file, and a column swap would break it
    assert np.all(np.diff(gap) < 0)


def test_parse_imaginary_axis_blocks_are_ragged(tmp_path):
    """src/eliashberg.f90 writes 2*nwf+1 lines per temperature, and nwf
    shrinks as T rises -- so the blocks are genuinely different lengths and
    must not be forced into one array."""
    path = tmp_path / "ELIASHBERG_IA.OUT"
    path.write_text(
        "  -0.3000000000E-02  0.5000000000E-03   2.000000000    \n"
        "  -0.1000000000E-02  0.6000000000E-03   2.100000000    \n"
        "   0.1000000000E-02  0.6000000000E-03   2.100000000    \n"
        "   0.3000000000E-02  0.5000000000E-03   2.000000000    \n"
        "\n"
        "  -0.2000000000E-02  0.2000000000E-03   1.500000000    \n"
        "   0.2000000000E-02  0.2000000000E-03   1.500000000    \n"
        "\n"
    )
    blocks = eliashberg.parse_imaginary_axis(path)
    assert [len(b[0]) for b in blocks] == [4, 2]
    matsubara, gap, z = blocks[0]
    np.testing.assert_allclose(matsubara, [-3e-3, -1e-3, 1e-3, 3e-3])
    # the functions are written symmetrically about zero (m = -n-1 for n < 0)
    np.testing.assert_allclose(gap, gap[::-1])
    np.testing.assert_allclose(z, z[::-1])


def test_parse_real_axis(tmp_path):
    # src/eliashberg.f90: write(65,'(3G18.10)') dble(zout(i)),a,b
    path = tmp_path / "ELIASHBERG_GAP_RA.OUT"
    path.write_text(
        "   0.000000000       0.5000000000E-03   0.000000000    \n"
        "   0.1000000000E-02  0.4000000000E-03  -0.1000000000E-04\n"
        "\n"
        "   0.000000000       0.2000000000E-03   0.000000000    \n"
        "   0.1000000000E-02  0.1000000000E-03  -0.2000000000E-04\n"
        "\n"
    )
    blocks = eliashberg.parse_real_axis(path)
    assert len(blocks) == 2
    frequencies, values = blocks[0]
    np.testing.assert_allclose(frequencies, [0.0, 1e-3])
    assert values[1] == pytest.approx(4e-4 - 1e-5j)
