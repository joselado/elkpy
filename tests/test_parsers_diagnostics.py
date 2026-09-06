"""Unit tests for parsers.diagnostics (task 68's RAM-disk report and the
TEST_nnn.OUT files of writetest/task 500).

Format-derived: transcribed from the `write` statements of
vendor/elk/src/modramdisk.f90 and vendor/elk/src/modtest.f90 cited above
each fixture. Neither task was run against the binary -- task 68 prints a
diagnostic and task 500 needs upstream reference files, so neither is a
physical observable this bucket's single run could check.
"""

import numpy as np
import pytest

from elkpy.parsers import diagnostics

# vendor/elk/src/modramdisk.f90:299-327 (rdstatus writes to STDOUT, which
# elkpy captures into elk.out)
#   write(*,'("Info(rdstatus):")')
#   write(*,'(" Filename : ",A)') file(i)%fname
#   write(*,'("  number of records : ",I0)') nr
#   write(*,'("  total number of bytes : ",I0)') 4*m
#   write(*,'(" Number of files on RAM disk : ",I0)') nf
#   write(*,'(" Total number of bytes used by RAM disk : ",I0)') 4*n
ELK_OUT = """
+----------------------+
| Current task :    68 |
+----------------------+

Info(rdstatus):

 Filename : EVALSV.OUT
  number of records : 3
  total number of bytes : 1440

 Filename : EVECSV.OUT
  number of records : 3
  total number of bytes : 34560

 Number of files on RAM disk : 2
 Total number of bytes used by RAM disk : 36000

+------------------+
| Elk code stopped |
+------------------+
"""

# vendor/elk/src/modramdisk.f90:303-306, the early-return branch
ELK_OUT_UNINITIALISED = """
Info(rdstatus):
 RAM disk not initialised

+------------------+
| Elk code stopped |
+------------------+
"""

# vendor/elk/src/modtest.f90:63-92
#   write(90,'("''",A,"''")') trim(descr)
#   write(90,'(2I8)') 2,nv                       ! vt=2 (real), nv values
#   write(90,'(G24.14)') tol
#   write(90,'(I8,G24.14)') j,rva(j)
# The description and tolerance are writestress.f90's own:
#   call writetest(440,'Stress tensor components',nv=nstrain,tol=5.d-2,rva=stress)
TEST_440_OUT = """'Stress tensor components'
       2       1
  0.50000000000000E-01
       1 -0.89383996060000E-02
"""

# vendor/elk/src/modtest.f90:88-93, the complex-array branch (vt=3), as
# used by sfacrho.f90:
#   call writetest(195,'density structure factors',nv=nhvec,tol=1.d-5,zva=zrhoh)
TEST_195_OUT = """'density structure factors'
       3       2
  0.10000000000000E-04
       1  0.28005208000000E+02  0.23483544000000E-16
       2 -0.15277207000000E+02  0.00000000000000E+00
"""

# vendor/elk/src/modtest.f90:77-81, the integer-array branch (vt=1): no
# tolerance line at all.
TEST_001_OUT = """'number of k-points'
       1       1
       1       3
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_ramdisk_status(tmp_path):
    result = diagnostics.parse_ramdisk_status(_write(tmp_path, "elk.out", ELK_OUT))
    assert result["initialised"] is True
    assert [f["filename"] for f in result["files"]] == ["EVALSV.OUT", "EVECSV.OUT"]
    assert result["files"][1]["records"] == 3
    assert result["files"][1]["bytes"] == 34560
    assert result["nfiles"] == 2
    # the per-file byte counts must add up to the reported total
    assert sum(f["bytes"] for f in result["files"]) == result["bytes"]


def test_parse_ramdisk_status_uninitialised_is_not_an_error(tmp_path):
    result = diagnostics.parse_ramdisk_status(
        _write(tmp_path, "elk.out", ELK_OUT_UNINITIALISED)
    )
    assert result["initialised"] is False
    assert result["files"] == []


def test_parse_ramdisk_status_requires_the_block(tmp_path):
    with pytest.raises(ValueError):
        diagnostics.parse_ramdisk_status(_write(tmp_path, "elk.out", "no such block\n"))


def test_parse_test_file_real(tmp_path):
    result = diagnostics.parse_test_file(_write(tmp_path, "TEST_440.OUT", TEST_440_OUT))
    assert result["description"] == "Stress tensor components"
    assert result["type"] == 2
    # writestress.f90 passes tol=5.d-2 -- upstream's own statement of how
    # reproducible the stress is
    assert result["tolerance"] == pytest.approx(0.05)
    assert result["values"] == pytest.approx([-0.008938399606])


def test_parse_test_file_complex(tmp_path):
    result = diagnostics.parse_test_file(_write(tmp_path, "TEST_195.OUT", TEST_195_OUT))
    assert result["type"] == 3
    assert result["values"].dtype == np.complex128
    assert result["values"][0].real == pytest.approx(28.005208)
    assert result["values"][1] == pytest.approx(-15.277207 + 0j)
    assert result["tolerance"] == pytest.approx(1e-5)


def test_parse_test_file_integer_has_no_tolerance(tmp_path):
    result = diagnostics.parse_test_file(_write(tmp_path, "TEST_001.OUT", TEST_001_OUT))
    assert result["type"] == 1
    assert result["tolerance"] is None
    assert result["values"].tolist() == [3]


def test_parse_test_file_rejects_a_truncated_file(tmp_path):
    text = TEST_195_OUT.rsplit("\n", 2)[0] + "\n"
    with pytest.raises(ValueError):
        diagnostics.parse_test_file(_write(tmp_path, "TEST_195.OUT", text))
