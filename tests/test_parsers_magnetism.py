"""Unit tests for elkpy.parsers.magnetism.

Every fixture string is transcribed from the Fortran `write` statement that
produces it (file and statement cited above each fixture). This is
format-derived verification: it pins the parser against the source of truth
for the layout, but it is NOT evidence that a real run produces these values.
"""

import numpy as np
import pytest

from elkpy.parsers import magnetism


# vendor/elk/src/mae.f90:
#   open(50,file='MAE.OUT',form='FORMATTED')
#   write(50,'(G18.10)') de
MAE_OUT = "  0.1234500000E-04\n"

# vendor/elk/src/mae.f90:
#   open(50,file='MAEPUV.OUT',form='FORMATTED')
#   write(50,'(G18.10)') de/omega
MAEPUV_OUT = "  0.7890000000E-07\n"

# vendor/elk/src/mae.f90, unit 71 (MAE_INFO.OUT), in file order:
#   write(71,'("Scale factor of spin-orbit coupling term : ",G18.10)') socscf
#   write(71,'("Fixed spin moment direction point ",I0," of ",I0)') i,npmae
#   write(71,'("Spherical coordinates of direction : ",2G18.10)') tpmae(:,i)
#   write(71,'("Direction vector (Cartesian coordinates) : ",3G18.10)') v2
#   write(71,'("Calculated total moment magnitude : ",G18.10)') momtotm
#   write(71,'("Total energy : ",G24.14)') engytot
#   write(71,'("Minimum energy point : ",I6)') i0
#   write(71,'("Maximum energy point : ",I6)') i1
#   write(71,'("Estimated magnetic anisotropy energy (MAE) : ",G18.10)') de
#   write(71,'("MAE per unit volume : ",G18.10)') de/omega
MAE_INFO_OUT = """
Scale factor of spin-orbit coupling term :   1.000000000

Fixed spin moment direction point 1 of 2
Spherical coordinates of direction :   1.570796327       0.000000000
Direction vector (Cartesian coordinates) :   1.000000000       0.000000000       0.000000000
Calculated total moment magnitude :   3.256789012
Total energy :      -2545.1234567890123

Fixed spin moment direction point 2 of 2
Spherical coordinates of direction :   0.000000000       0.000000000
Direction vector (Cartesian coordinates) :   0.000000000       0.000000000       1.000000000
Calculated total moment magnitude :   3.257123456
Total energy :      -2545.1234691340123

Minimum energy point :      2
Maximum energy point :      1

Estimated magnetic anisotropy energy (MAE) :   0.1234500000E-04

MAE per unit volume :   0.7890000000E-07
"""

# vendor/elk/src/torque.f90 (standard output, captured by LocalLauncher into
# elk.out):
#   write(*,*)
#   write(*,'("Info(torque):")')
#   write(*,'(" Total torque exerted by B_xc on the magnetisation :")')
#   write(*,'(3G18.10)') torq
TORQUE_LOG = """ Elk code version 11.0.02 started

Info(torque):
 Total torque exerted by B_xc on the magnetisation :
 -0.1234000000E-03  0.5678000000E-04  0.9000000000E-06

Elk code stopped
"""

# vendor/elk/src/spiralsc.f90, unit 80 (SS_Q..._..._....OUT):
#   write(80,'(I6,T20," : number of unit cells in supercell")') nscss
#   write(80,'(G18.10,T20," : total energy per unit cell")') engytot/dble(nscss)
#   write(80,*)
#   write(80,'("q-point in lattice and Cartesian coordinates :")')
#   write(80,'(3G18.10)') vql(:,iqss)
#   write(80,'(3G18.10)') vqc(:,iqss)
#   write(80,'(G18.10,T20," : length of q-vector")') q
#   write(80,*)
#   write(80,'(I6,T20," : number of equivalent q-points")') nq
#   write(80,'("Equivalent q-points in lattice and Cartesian coordinates :")')
#   ! then per equivalent q-point: 3G18.10 lattice, 3G18.10 Cartesian, blank
SS_OUT = """     2             : number of unit cells in supercell
 -1272.345678                 : total energy per unit cell

q-point in lattice and Cartesian coordinates :
  0.5000000000       0.000000000       0.000000000
  0.4652000000       0.4652000000       0.000000000
  0.6579000000                 : length of q-vector

     3             : number of equivalent q-points
Equivalent q-points in lattice and Cartesian coordinates :
  0.5000000000       0.000000000       0.000000000
  0.4652000000       0.4652000000       0.000000000

  0.000000000       0.5000000000       0.000000000
  0.4652000000       0.000000000       0.4652000000

  0.000000000       0.000000000       0.5000000000
  0.000000000       0.4652000000       0.4652000000

"""


def test_parse_mae(tmp_path):
    path = tmp_path / "MAE.OUT"
    path.write_text(MAE_OUT)
    assert magnetism.parse_mae(path) == pytest.approx(1.2345e-05)


def test_parse_mae_per_volume(tmp_path):
    path = tmp_path / "MAEPUV.OUT"
    path.write_text(MAEPUV_OUT)
    assert magnetism.parse_mae_per_volume(path) == pytest.approx(7.89e-08)


def test_parse_mae_info(tmp_path):
    path = tmp_path / "MAE_INFO.OUT"
    path.write_text(MAE_INFO_OUT)
    info = magnetism.parse_mae_info(path)

    assert info["socscf"] == pytest.approx(1.0)
    assert info["npmae"] == 2
    assert info["min_point"] == 2
    assert info["max_point"] == 1
    assert info["mae"] == pytest.approx(1.2345e-05)
    assert info["mae_per_volume"] == pytest.approx(7.89e-08)

    assert len(info["directions"]) == 2
    first, second = info["directions"]
    assert first["index"] == 1
    # theta = pi/2, phi = 0 -> the x direction
    assert first["theta"] == pytest.approx(np.pi / 2, abs=1e-6)
    assert first["phi"] == pytest.approx(0.0)
    assert first["direction"] == pytest.approx(np.array([1.0, 0.0, 0.0]))
    assert first["moment"] == pytest.approx(3.256789012)
    assert first["energy"] == pytest.approx(-2545.1234567890123)
    # theta = 0 -> the z direction, and the lower of the two energies, which
    # is what makes point 2 the minimum reported above
    assert second["direction"] == pytest.approx(np.array([0.0, 0.0, 1.0]))
    assert second["energy"] < first["energy"]
    # the reported MAE is the spread between max and min energies
    assert first["energy"] - second["energy"] == pytest.approx(info["mae"], rel=1e-3)


def test_parse_mae_info_tolerates_an_interrupted_run(tmp_path):
    """A sweep killed partway leaves the summary lines unwritten; the
    per-direction records already flushed must still be readable."""
    partial = MAE_INFO_OUT.split("Minimum energy point")[0]
    path = tmp_path / "MAE_INFO.OUT"
    path.write_text(partial)
    info = magnetism.parse_mae_info(path)
    assert len(info["directions"]) == 2
    assert "mae" not in info


def test_parse_torque(tmp_path):
    path = tmp_path / "elk.out"
    path.write_text(TORQUE_LOG)
    torque = magnetism.parse_torque(path)
    assert torque == pytest.approx(np.array([-1.234e-04, 5.678e-05, 9.0e-07]))


def test_parse_torque_without_the_marker_raises(tmp_path):
    path = tmp_path / "elk.out"
    path.write_text("Elk code version 11.0.02 started\nElk code stopped\n")
    with pytest.raises(ValueError, match="no torque output"):
        magnetism.parse_torque(path)


def test_spin_spiral_filename_reduces_the_fraction():
    """src/ssfext.f90 divides each component by gcd(ivq_i, ngridq_i), so
    q = (2/4, 0, 0) is written as the reduced 1/2 and NOT as 0204."""
    assert (
        magnetism.spin_spiral_filename((2, 0, 0), (4, 4, 4))
        == "SS_Q0102_0000_0000.OUT"
    )
    assert (
        magnetism.spin_spiral_filename((1, 0, 0), (2, 2, 2))
        == "SS_Q0102_0000_0000.OUT"
    )


def test_spin_spiral_filename_zero_component_is_0000_not_00nn():
    """The `else` branch of ssfext.f90 sets BOTH m(i) and n(i) to zero, so a
    vanishing component prints 0000 rather than 00 followed by ngridq."""
    assert (
        magnetism.spin_spiral_filename((0, 0, 0), (4, 4, 4))
        == "SS_Q0000_0000_0000.OUT"
    )
    assert (
        magnetism.spin_spiral_filename((0, 1, 3), (4, 4, 4))
        == "SS_Q0000_0104_0304.OUT"
    )


def test_spin_spiral_filename_rejects_negative_indices():
    with pytest.raises(ValueError, match="negative q-grid index"):
        magnetism.spin_spiral_filename((-1, 0, 0), (4, 4, 4))


def test_parse_spin_spiral(tmp_path):
    path = tmp_path / "SS_Q0102_0000_0000.OUT"
    path.write_text(SS_OUT)
    record = magnetism.parse_spin_spiral(path)
    assert record["ncells"] == 2
    assert record["energy"] == pytest.approx(-1272.345678)
    assert record["q_lattice"] == pytest.approx(np.array([0.5, 0.0, 0.0]))
    assert record["q_cartesian"] == pytest.approx(np.array([0.4652, 0.4652, 0.0]))
    assert record["q_length"] == pytest.approx(0.6579)
    assert record["nequivalent"] == 3
    assert len(record["equivalent"]) == 3
    assert record["equivalent"][2][0] == pytest.approx(np.array([0.0, 0.0, 0.5]))


def test_parse_spin_spiral_rejects_an_empty_placeholder(tmp_path):
    """Task 352 (dry run) writes every SS file empty; so does a task-350 run
    on the q-point it is currently working on (src/sstask.f90)."""
    path = tmp_path / "SS_Q0102_0000_0000.OUT"
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        magnetism.parse_spin_spiral(path)


def test_collect_spin_spirals_separates_finished_from_pending(tmp_path):
    (tmp_path / "SS_Q0102_0000_0000.OUT").write_text(SS_OUT)
    (tmp_path / "SS_Q0104_0000_0000.OUT").write_text(
        SS_OUT.replace("0.6579000000", "0.3290000000")
    )
    (tmp_path / "SS_Q0000_0000_0000.OUT").write_text("")  # claimed, unfinished
    results, pending = magnetism.collect_spin_spirals(tmp_path)
    assert [r["file"] for r in results] == [
        "SS_Q0104_0000_0000.OUT",  # sorted by |q|, so the shorter one first
        "SS_Q0102_0000_0000.OUT",
    ]
    assert pending == ["SS_Q0000_0000_0000.OUT"]
