"""Unit tests for parsers.sfac (SFACRHO.OUT / SFACMAG_j.OUT).

The charge fixture is verbatim output of a real compiled Elk binary (bulk
Si, diamond structure, ngridk 2x2x2, rgkmax 5, hmaxvr 6, task 195), so the
normalisation check below has teeth: Elk prints ``omega * F`` with the
imaginary part negated (the crystallographic convention), which makes the
H = 0 coefficient the total electron count of the cell -- 28 for two
silicon atoms, measured 28.005208, the density representation's own
0.02% charge error.

The magnetic fixture and the non-integer-hkl fixture are format-derived,
transcribed from the `write` statements cited above each one.
"""

import numpy as np
import pytest

from elkpy.parsers import sfac

# vendor/elk/src/sfacrho.f90:40-71
#   write(50,'("h k l indices transformed by vhmat matrix:")')
#   write(50,'(3G18.10)') vhmat(:,1)   ! and (:,2), (:,3)
#   write(50,'("      h      k      l  multipl.   |H|            Re(F)&
#    &            Im(F)           |F|")')
#   write(50,'(4I7,4G16.8)') iv(:),mulh(ih),hc(ih),a,b,r
# Real output, bulk Si, truncated after the first reflections.
SFACRHO_OUT = """
h k l indices transformed by vhmat matrix:
   1.000000000       0.000000000       0.000000000
   0.000000000       1.000000000       0.000000000
   0.000000000       0.000000000       1.000000000

      h      k      l  multipl.   |H|            Re(F)            Im(F)           |F|

      0      0      0      1   0.0000000       28.005208     -0.23483544E-16   28.005208
     -1      0      0      6   1.0607014      -15.277207      -0.0000000       15.277207
     -1     -1     -1      2   1.0607014       15.277207      -0.0000000       15.277207
     -1      0     -1      6   1.2247925     -0.31279263E-14  -0.0000000      0.31279263E-14
      1      0     -1      6   1.7321181       17.282369     -0.45006837E-16   17.282369
      2      1      1      6   1.7321181      -17.282369     -0.31404236E-16   17.282369
"""

# vendor/elk/src/sfacrho.f90:66-68, the non-integer-hkl branch taken when
# vhmat maps the H-vectors off the integer lattice. Format-derived.
#   write(50,'(3F7.2,I7,4G16.8)') v(:),mulh(ih),hc(ih),a,b,r
SFACRHO_OUT_NONINTEGER = """
h k l indices transformed by vhmat matrix:
  0.5000000000      0.5000000000       0.000000000
 -0.5000000000      0.5000000000       0.000000000
   0.000000000       0.000000000       1.000000000

      h      k      l  multipl.   |H|            Re(F)            Im(F)           |F|

   0.00   0.00   0.00      1   0.0000000       28.005208      -0.0000000       28.005208
  -0.50   0.50   0.00      4   1.0607014      -15.277207      -0.0000000       15.277207
"""

# vendor/elk/src/sfacmag.f90:43-72, same table format, one file per
# magnetisation component. Format-derived.
SFACMAG_OUT = """
h k l indices transformed by vhmat matrix:
   1.000000000       0.000000000       0.000000000
   0.000000000       1.000000000       0.000000000
   0.000000000       0.000000000       1.000000000

      h      k      l  multipl.   |H|            Re(F)            Im(F)           |F|

      0      0      0      1   0.0000000       2.2000000      -0.0000000       2.2000000
     -1      0      0      6   1.0607014      0.55000000      0.11000000      0.56089215
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_structure_factors_charge(tmp_path):
    result = sfac.parse_structure_factors(_write(tmp_path, "SFACRHO.OUT", SFACRHO_OUT))
    assert result["hkl"].shape == (6, 3)
    assert result["vhmat"] == pytest.approx(np.eye(3))
    # H = 0 comes first (genhvec sorts by |H|) and equals the electron count
    assert result["h"][0] == 0.0
    assert result["hkl"][0] == pytest.approx([0.0, 0.0, 0.0])
    assert result["F"][0].real == pytest.approx(28.0, abs=0.01)
    assert abs(result["F"][0].imag) < 1e-14
    # |H| is sorted ascending
    assert np.all(np.diff(result["h"]) >= 0)
    assert result["multiplicity"][1] == 6
    # the (1,0,1)-type reflection of the diamond lattice is extinct
    assert abs(result["F"][3]) < 1e-13


def test_parse_structure_factors_modulus_matches_printed_column(tmp_path):
    """The parser drops the printed |F| column and reconstructs it from
    Re/Im; check the two agree, which pins the column ordering."""
    result = sfac.parse_structure_factors(_write(tmp_path, "SFACRHO.OUT", SFACRHO_OUT))
    printed = [28.005208, 15.277207, 15.277207, 0.31279263e-14, 17.282369, 17.282369]
    assert np.abs(result["F"]) == pytest.approx(printed, rel=1e-6)


def test_parse_structure_factors_non_integer_hkl(tmp_path):
    result = sfac.parse_structure_factors(
        _write(tmp_path, "SFACRHO.OUT", SFACRHO_OUT_NONINTEGER)
    )
    assert result["hkl"].shape == (2, 3)
    assert result["hkl"][1] == pytest.approx([-0.5, 0.5, 0.0])
    assert result["multiplicity"][1] == 4
    assert result["vhmat"][0] == pytest.approx([0.5, 0.5, 0.0])


def test_parse_structure_factors_magnetic(tmp_path):
    result = sfac.parse_structure_factors(_write(tmp_path, "SFACMAG_1.OUT", SFACMAG_OUT))
    assert result["F"][0] == pytest.approx(2.2 + 0j)
    assert result["F"][1] == pytest.approx(0.55 + 0.11j)


def test_parse_structure_factors_rejects_headerless_file(tmp_path):
    with pytest.raises(ValueError):
        sfac.parse_structure_factors(_write(tmp_path, "SFACRHO.OUT", "nothing here\n"))
