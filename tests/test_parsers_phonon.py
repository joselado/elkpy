"""Unit tests for parsers/phonon.py.

Every fixture below is transcribed from the `write` statement in
vendor/elk/src/ that produces the file, cited inline. The BEC fixture is
stronger than that: it is the verbatim output of a real Elk 11.0.2 task-208
run on bulk Si (2-atom diamond cell, ngridk=(2,2,2), rgkmax=6, nkspolar=2),
which is also why parse_born_charge_row is labelled binary-verified.
"""

import numpy as np
import pytest

from elkpy.parsers import phonon


# --------------------------------------------------------------------------
# Born effective charges (task 208)
# --------------------------------------------------------------------------

# src/bornechg.f90:
#     do ip=1,3
#       write(80,'(G18.10," : ip = ",I4)') becc(ip),ip
#     end do
# Verbatim BEC_S01_A001_P1.OUT from the real Si run described above.
BEC_P1 = """\
 -0.1295320358E-02 : ip =    1
  0.5290518330E-03 : ip =    2
  0.1021657080E-02 : ip =    3
"""
BEC_P2 = """\
 -0.6523677399E-03 : ip =    1
 -0.1217333205E-02 : ip =    2
  0.2015669917E-02 : ip =    3
"""
BEC_P3 = """\
  0.2808982859E-03 : ip =    1
 -0.2058398090E-03 : ip =    2
 -0.1991603976E-02 : ip =    3
"""


def test_born_charge_filename_matches_becfext():
    # src/becfext.f90: write(fext,'("_S",I2.2,"_A",I3.3,"_P",I1,".OUT")') is,ia,ip
    assert phonon.born_charge_filename(1, 1, 1) == "BEC_S01_A001_P1.OUT"
    assert phonon.born_charge_filename(2, 13, 3) == "BEC_S02_A013_P3.OUT"


def test_parse_born_charge_row(tmp_path):
    path = tmp_path / "BEC_S01_A001_P1.OUT"
    path.write_text(BEC_P1)
    row = phonon.parse_born_charge_row(path)
    assert row.shape == (3,)
    np.testing.assert_allclose(row, [-1.295320358e-03, 5.290518330e-04, 1.021657080e-03])


def test_parse_born_charge_row_rejects_wrong_length(tmp_path):
    path = tmp_path / "BEC_S01_A001_P1.OUT"
    path.write_text(BEC_P1 + "  0.1E-02 : ip =    4\n")
    with pytest.raises(ValueError, match="expected 3"):
        phonon.parse_born_charge_row(path)


def test_parse_born_charges_assembles_tensors(tmp_path):
    for ip, text in zip((1, 2, 3), (BEC_P1, BEC_P2, BEC_P3)):
        (tmp_path / phonon.born_charge_filename(1, 1, ip)).write_text(text)
        # a second, identical atom so the species_counts loop is exercised
        (tmp_path / phonon.born_charge_filename(1, 2, ip)).write_text(text)
    charges = phonon.parse_born_charges(tmp_path, [2])
    assert set(charges) == {(1, 1), (1, 2)}
    z = charges[(1, 1)]
    assert z.shape == (3, 3)
    # row index = the file's _P suffix (displacement), column = line in file
    assert z[0, 0] == pytest.approx(-1.295320358e-03)
    assert z[1, 2] == pytest.approx(2.015669917e-03)
    assert z[2, 1] == pytest.approx(-2.058398090e-04)
    # diamond Si: every element is numerically zero against a nuclear charge
    # of 14, which is what makes this file a null test of the whole task
    assert np.max(np.abs(z)) < 1e-2


# --------------------------------------------------------------------------
# dynamical Born effective charges (task 478)
# --------------------------------------------------------------------------

def _dynamical_bec_text(nw=3):
    """src/bornecdyn.f90:
        do i=1,3
          do iw=1,nwplot
            write(80,'(2G18.10)') w(iw),dble(becw(iw,i))+t2
          end do
          write(80,*)
          do iw=1,nwplot
            write(80,'(2G18.10)') w(iw),aimag(becw(iw,i))
          end do
          write(80,*)
        end do
    """
    w = [0.0, 0.5, 1.0][:nw]
    lines = []
    for i in range(3):
        for part in (0, 1):
            for iw, wi in enumerate(w):
                value = (i + 1) * 10.0 + part * 100.0 + iw
                lines.append(f"{wi:18.10G}{value:18.10G}")
            lines.append("")
    return "\n".join(lines) + "\n"


def test_parse_born_charge_dynamical(tmp_path):
    path = tmp_path / "BEC_S01_A001_P1.OUT"
    path.write_text(_dynamical_bec_text())
    frequencies, z = phonon.parse_born_charge_dynamical(path)
    np.testing.assert_allclose(frequencies, [0.0, 0.5, 1.0])
    assert z.shape == (3, 3)
    # column i is polarisation component i: Re from block 2i, Im from 2i+1
    assert z[0, 0] == pytest.approx(10.0 + 110.0j)
    assert z[2, 1] == pytest.approx(22.0 + 122.0j)
    assert z[1, 2] == pytest.approx(31.0 + 131.0j)


def test_parse_born_charge_dynamical_rejects_static_file(tmp_path):
    """A task-208 file has three lines and no blank separators, so it must
    not be silently read as a task-478 one -- the filenames collide."""
    path = tmp_path / "BEC_S01_A001_P1.OUT"
    path.write_text(BEC_P1)
    with pytest.raises(ValueError):
        phonon.parse_born_charge_dynamical(path)


# --------------------------------------------------------------------------
# PHONON.OUT (task 230)
# --------------------------------------------------------------------------

# src/writephn.f90, for a 1-species / 1-atom cell (nbph = 3) and two
# q-points. Note the trailing comment on the FIRST eigenvector line of every
# mode, which the parser has to tolerate.
PHONON_OUT = """
     1  0.000000000       0.000000000       0.000000000      : q-point, vqlwrt

     1  0.000000000     : mode, frequency
   1   1   1  1.000000000       0.000000000      : species, atom, polarisation, eigenvector
   1   1   2  0.000000000       0.000000000
   1   1   3  0.000000000       0.000000000

     2  0.1000000000E-02 : mode, frequency
   1   1   1  0.000000000       0.000000000      : species, atom, polarisation, eigenvector
   1   1   2  0.7071067812       0.000000000
   1   1   3  0.000000000       0.7071067812

     3  0.2000000000E-02 : mode, frequency
   1   1   1  0.000000000       0.000000000      : species, atom, polarisation, eigenvector
   1   1   2  0.000000000       0.7071067812
   1   1   3  0.7071067812       0.000000000

     2  0.5000000000      0.000000000       0.000000000      : q-point, vqlwrt

     1 -0.5000000000E-03 : mode, frequency
   1   1   1  1.000000000       0.000000000      : species, atom, polarisation, eigenvector
   1   1   2  0.000000000       0.000000000
   1   1   3  0.000000000       0.000000000

     2  0.3000000000E-02 : mode, frequency
   1   1   1  0.000000000       0.000000000      : species, atom, polarisation, eigenvector
   1   1   2  1.000000000       0.000000000
   1   1   3  0.000000000       0.000000000

     3  0.4000000000E-02 : mode, frequency
   1   1   1  0.000000000       0.000000000      : species, atom, polarisation, eigenvector
   1   1   2  0.000000000       0.000000000
   1   1   3  1.000000000       0.000000000

"""


def test_parse_phonon_modes(tmp_path):
    path = tmp_path / "PHONON.OUT"
    path.write_text(PHONON_OUT)
    modes = phonon.parse_phonon_modes(path)
    assert len(modes) == 2
    assert modes[0]["index"] == 1
    assert modes[0]["qpoint"] == (0.0, 0.0, 0.0)
    assert modes[1]["qpoint"] == (0.5, 0.0, 0.0)
    np.testing.assert_allclose(modes[0]["frequencies"], [0.0, 1e-3, 2e-3])
    ev = modes[0]["eigenvectors"]
    assert ev.shape == (3, 3)
    # eigenvectors[j] is mode j; its components run over (species, atom, ip)
    np.testing.assert_allclose(ev[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(ev[1], [0.0, 0.7071067812, 0.7071067812j])
    np.testing.assert_allclose(ev[2], [0.0, 0.7071067812j, 0.7071067812])
    # src/dynev.f90 stores sign(sqrt(|w^2|), w^2), so an unstable mode is a
    # NEGATIVE frequency rather than a complex one
    assert modes[1]["frequencies"][0] < 0.0


def test_parse_phonon_modes_eigenvectors_are_orthonormal(tmp_path):
    """dynev.f90 diagonalises a Hermitian matrix with zheevdi, so the mode
    vectors are orthonormal -- a property the parser must not scramble by
    transposing the (i, j) indexing."""
    path = tmp_path / "PHONON.OUT"
    path.write_text(PHONON_OUT)
    ev = phonon.parse_phonon_modes(path)[0]["eigenvectors"]
    np.testing.assert_allclose(ev.conj() @ ev.T, np.eye(3), atol=1e-9)


# --------------------------------------------------------------------------
# GAMMAQ.OUT / LAMBDAQ.OUT (tasks 240/241)
# --------------------------------------------------------------------------

# src/writegamma.f90 (src/writelambda.f90 writes the identical layout):
#     write(50,*)
#     write(50,'(I4," : total number of atoms")') natmtot
#     write(50,'(I6," : number of q-points")') nqpt
#     write(50,*)
#     do iq=1,nqpt
#       write(50,'(I6," : q-point")') iq
#       write(50,'(3G18.10," : q-vector (lattice coordinates)")') vql(:,iq)
#       write(50,'(3G18.10," : q-vector (Cartesian coordinates)")') vqc(:,iq)
#       do i=1,nbph
#         write(50,'(I4,G18.10)') i,gq(i,iq)
#       end do
#       write(50,*)
#     end do
GAMMAQ_HEADER = """
   1 : total number of atoms
     2 : number of q-points

"""

GAMMAQ_BLOCK_1 = """\
     1 : q-point
  0.000000000       0.000000000       0.000000000      : q-vector (lattice coordinates)
  0.000000000       0.000000000       0.000000000      : q-vector (Cartesian coordinates)
   1  0.1000000000E-05
   2  0.2000000000E-05
   3  0.3000000000E-05

"""

GAMMAQ_BLOCK_2 = """\
     2 : q-point
  0.5000000000      0.000000000       0.000000000      : q-vector (lattice coordinates)
  0.3141592654      0.000000000       0.000000000      : q-vector (Cartesian coordinates)
   1  0.4000000000E-05
   2  0.5000000000E-05
   3  0.6000000000E-05

"""

GAMMAQ_OUT = GAMMAQ_HEADER + GAMMAQ_BLOCK_1 + GAMMAQ_BLOCK_2


def test_parse_qpoint_table(tmp_path):
    path = tmp_path / "GAMMAQ.OUT"
    path.write_text(GAMMAQ_OUT)
    table = phonon.parse_qpoint_table(path)
    assert table["natoms"] == 1
    np.testing.assert_allclose(table["qpoints"], [[0, 0, 0], [0.5, 0, 0]])
    np.testing.assert_allclose(
        table["qpoints_cartesian"], [[0, 0, 0], [0.3141592654, 0, 0]]
    )
    assert table["values"].shape == (2, 3)
    np.testing.assert_allclose(table["values"][1], [4e-6, 5e-6, 6e-6])


def test_parse_qpoint_table_detects_truncation(tmp_path):
    path = tmp_path / "GAMMAQ.OUT"
    # drop the second q-point block while leaving the declared count at 2 --
    # exactly what a killed run leaves behind
    path.write_text(GAMMAQ_HEADER + GAMMAQ_BLOCK_1)
    with pytest.raises(ValueError, match="declares 2 q-points"):
        phonon.parse_qpoint_table(path)


# --------------------------------------------------------------------------
# block splitter and FACEEH.OUT
# --------------------------------------------------------------------------

def test_split_blocks_allows_ragged_blocks(tmp_path):
    """ELIASHBERG_IA.OUT genuinely has blocks of different lengths, so the
    splitter must not assume a rectangle."""
    path = tmp_path / "ragged.OUT"
    path.write_text("1.0 2.0\n3.0 4.0\n\n5.0 6.0\n")
    blocks = phonon.split_blocks(path, ncols=2)
    assert [b.shape for b in blocks] == [(2, 2), (1, 2)]


def test_split_blocks_checks_column_count(tmp_path):
    path = tmp_path / "bad.OUT"
    path.write_text("1.0 2.0\n3.0\n")
    with pytest.raises(ValueError, match="expected 2 columns"):
        phonon.split_blocks(path, ncols=2)


def test_parse_face_histogram(tmp_path):
    # src/ephdos.f90: write(50,'(5G18.10)') evaluv(ist,ik),t1,v
    path = tmp_path / "FACEEH.OUT"
    path.write_text(
        "  0.1000000000E-02  0.6931471806      0.000000000       0.000000000       0.000000000\n"
        " -0.1000000000E-02  0.5000000000      0.100000000       0.200000000       0.300000000\n"
    )
    energies, entropy, kpoints = phonon.parse_face_histogram(path)
    np.testing.assert_allclose(energies, [1e-3, -1e-3])
    np.testing.assert_allclose(entropy, [0.6931471806, 0.5])
    assert kpoints.shape == (2, 3)
    np.testing.assert_allclose(kpoints[1], [0.1, 0.2, 0.3])
