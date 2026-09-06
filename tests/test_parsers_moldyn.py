"""Unit tests for parsers.moldyn (the *_TD.OUT trajectory files, ATDVC.OUT
and TIMESTEP.OUT of tasks 420/421).

All fixtures are format-derived: transcribed from the `write` statements
cited above each one in vendor/elk/src/. Molecular dynamics was not run
against the binary (one force step is a full SCF and Elk's defaults ask
for a hundred of them), so nothing here is binary-verified.

Two shape details the parser has to get right and that these pin:

- the files are opened with ``position='APPEND'`` and written once per
  force step, so they are a concatenation of blocks, not a table; a
  restart appends to the same file;
- MOMENTMT_TD.OUT writes only ``mommt(1:ndmag,ias)``, one column for a
  collinear run and three for a non-collinear one, and its block header
  is a bare time rather than (step, time) -- so neither the row width nor
  the header width may be hardcoded.
"""

import numpy as np
import pytest

from elkpy.parsers import moldyn

# vendor/elk/src/writetdengy.f90:10
#   write(50,'(2G18.10)') times(itimes),engytot
TOTENERGY_TD_OUT = """   0.000000000      -577.5000000
   10.00000000      -577.4990000
   20.00000000      -577.4970000
"""

# vendor/elk/src/writetdforces.f90:13-21
#   write(50,'(I8,G18.10)') itimes,times(itimes)
#   write(50,'(2I4,3G18.10)') is,ia,forcetot(1:3,ias)
FORCETOT_TD_OUT = """       1   0.000000000
   1   1  0.1000000000E-02   0.000000000       0.000000000
   1   2 -0.1000000000E-02   0.000000000       0.000000000
     101   10.00000000
   1   1  0.9000000000E-03   0.000000000       0.000000000
   1   2 -0.9000000000E-03   0.000000000       0.000000000
"""

# vendor/elk/src/writeatdisp.f90:12-22 (ATDISPC_TD.OUT, Cartesian branch)
ATDISPC_TD_OUT = """       1   0.000000000
   1   1   0.000000000       0.000000000       0.000000000
   1   2   0.000000000       0.000000000       0.000000000
     101   10.00000000
   1   1  0.5000000000E-02   0.000000000       0.000000000
   1   2 -0.5000000000E-02   0.000000000       0.000000000
"""

# vendor/elk/src/writemomtd.f90:24-33, collinear (ndmag = 1): the per-atom
# row carries ONE component, and the block header is a bare time.
#   write(50,'(G18.10)') times(itimes)
#   write(50,'(2I4,3G18.10)') is,ia,mommt(1:ndmag,ias)
MOMENTMT_TD_OUT = """   0.000000000
   1   1   2.200000000
   1   2   2.200000000

   10.00000000
   1   1   2.190000000
   1   2   2.190000000

"""

# vendor/elk/src/writeatdvc.f90:12-17
#   write(50,'(2I4,6G18.10)') is,ia,atdvc(:,:,ia,is)
# Fortran runs the first index fastest, and atdvc(1:3,0:1,...) holds the
# displacement at index 0 and the velocity at index 1, so the six numbers
# are (dx,dy,dz,vx,vy,vz).
ATDVC_OUT = """   1   1  0.5000000000E-02   0.000000000       0.000000000      0.1000000000E-03   0.000000000       0.000000000
   1   2 -0.5000000000E-02   0.000000000       0.000000000     -0.1000000000E-03   0.000000000       0.000000000
"""

# vendor/elk/src/writetimes.f90:10
#   write(50,'(I8,G18.10)') itimes,times(itimes)
TIMESTEP_OUT = "     101   10.00000000\n"


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_scalar_series(tmp_path):
    data = moldyn.parse_scalar_series(_write(tmp_path, "TOTENERGY_TD.OUT", TOTENERGY_TD_OUT))
    assert data.shape == (3, 2)
    assert data[:, 0] == pytest.approx([0.0, 10.0, 20.0])
    assert data[-1, 1] == pytest.approx(-577.497)


def test_parse_scalar_series_rejects_ragged(tmp_path):
    text = TOTENERGY_TD_OUT + "   30.00000000\n"
    with pytest.raises(ValueError):
        moldyn.parse_scalar_series(_write(tmp_path, "TOTENERGY_TD.OUT", text))


def test_parse_atom_series_forces(tmp_path):
    data = moldyn.parse_atom_series(_write(tmp_path, "FORCETOT_TD.OUT", FORCETOT_TD_OUT))
    assert data["values"].shape == (2, 2, 3)
    assert data["step"] == pytest.approx([1, 101])
    assert data["time"] == pytest.approx([0.0, 10.0])
    assert data["species"] == pytest.approx([1, 1])
    assert data["atom"] == pytest.approx([1, 2])
    assert data["values"][0, 0, 0] == pytest.approx(1e-3)
    # Newton's third law for a two-atom cell: the forces are equal and opposite
    assert data["values"][0].sum(axis=0) == pytest.approx([0.0, 0.0, 0.0])


def test_parse_atom_series_displacements(tmp_path):
    data = moldyn.parse_atom_series(_write(tmp_path, "ATDISPC_TD.OUT", ATDISPC_TD_OUT))
    assert data["values"].shape == (2, 2, 3)
    # t = 0 starts undisplaced
    assert data["values"][0] == pytest.approx(np.zeros((2, 3)))
    assert data["values"][1, 0, 0] == pytest.approx(5e-3)


def test_parse_atom_series_collinear_moment_width(tmp_path):
    """MOMENTMT_TD.OUT writes ndmag components, not 3, and its header is a
    bare time -- neither width may be hardcoded."""
    data = moldyn.parse_atom_series(_write(tmp_path, "MOMENTMT_TD.OUT", MOMENTMT_TD_OUT))
    assert data["values"].shape == (2, 2, 1)
    assert data["time"] == pytest.approx([0.0, 10.0])
    assert data["values"][0, 0, 0] == pytest.approx(2.2)


def test_parse_atdvc(tmp_path):
    data = moldyn.parse_atdvc(_write(tmp_path, "ATDVC.OUT", ATDVC_OUT))
    assert data["displacement"].shape == (2, 3)
    assert data["velocity"].shape == (2, 3)
    assert data["displacement"][0] == pytest.approx([5e-3, 0.0, 0.0])
    assert data["velocity"][0] == pytest.approx([1e-4, 0.0, 0.0])
    assert data["velocity"][1] == pytest.approx([-1e-4, 0.0, 0.0])


def test_parse_atdvc_rejects_wrong_width(tmp_path):
    with pytest.raises(ValueError):
        moldyn.parse_atdvc(_write(tmp_path, "ATDVC.OUT", "   1   1   0.0   0.0   0.0\n"))


def test_parse_timestep(tmp_path):
    step, time = moldyn.parse_timestep(_write(tmp_path, "TIMESTEP.OUT", TIMESTEP_OUT))
    assert step == 101
    assert time == pytest.approx(10.0)
