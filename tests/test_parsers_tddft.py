"""Unit tests for parsers/tddft.py -- pure text parsing, no Elk run needed.

Every fixture is transcribed from the write statement that produces the
file, with the source cited on the test:

  AFIELDT.OUT   vendor/elk/src/genafieldt.f90, closing block:
                  write(50,'(I8," : number of time steps")') ntimes
                  write(50,'(I8,4G18.10)') its, times(its), afieldt(:,its)
  AFSPT.OUT     same file: write(50,'(I8,10G18.10)') its, times, afspt(:,:,its)
  AFPDT.OUT     vendor/elk/src/writeafpdt.f90:
                  write(50,'(2G18.10)') times(its), pd(its)
  AFTED.OUT     same file: write(50,'("Total energy density : ",G18.10)') ed
  EFIELDW.OUT   vendor/elk/src/writeefieldw.f90:
                  do i=1,3 ; do iw ; write(50,'(3G18.10)') w(iw), ew(iw,i)
                  ; end do ; write(50,*) ; end do
  JTOT_TD.OUT   vendor/elk/src/writetddft.f90:
                  write(50,'(4G18.10)') times(itimes), jtot(:)
  TOTENERGY_TD  vendor/elk/src/writetdengy.f90:
                  write(50,'(2G18.10)') times(itimes), engytot
  CHARGEMT_TD   vendor/elk/src/writetddft.f90:
                  write(50,'(G18.10)') times(itimes)
                  write(50,'(2I4,G18.10)') is, ia, chgmt(ias)
                  write(50,*)
  EPSILON_TDDFT / CHI_ij / EPSILON_TDRT: the standard two-block
                (2G18.10) layout of src/tddftlr.f90, src/tddftsplr.f90 and
                src/dielectric_tdrt.f90.
"""

import numpy as np
import pytest

from elkpy.parsers.tddft import (
    parse_afieldt,
    parse_afpdt,
    parse_afspt,
    parse_afted,
    parse_atom_time_series,
    parse_column_blocks,
    parse_efieldw,
    parse_epsilon_tdrt,
    parse_jtotw,
    parse_spin_response,
    parse_tddft_response,
    parse_time_series,
    tddftlr_energies,
)


def _write_two_blocks(path, x, first, second):
    with open(path, "w") as fh:
        for a, b in zip(x, first):
            fh.write(f"{a:18.10G}{b:18.10G}\n")
        fh.write("\n")
        for a, b in zip(x, second):
            fh.write(f"{a:18.10G}{b:18.10G}\n")


# --------------------------------------------------------------------------
# frequency grid
# --------------------------------------------------------------------------

def test_tddftlr_energies_drops_the_first_point():
    # src/init3.f90 builds nwplot frequencies; src/tddftlr.f90 writes
    # `do iw=2,nwrf`, so one fewer row reaches the file
    w = tddftlr_energies((0.1, 0.5), 8)
    assert len(w) == 7
    assert w[0] == pytest.approx(0.1 + (0.5 - 0.1) / 8)
    # unlike task 121, wplot(1) is NOT clipped to zero
    w = tddftlr_energies((-0.2, 0.2), 4)
    assert w[0] == pytest.approx(-0.1)


# --------------------------------------------------------------------------
# generic block reader
# --------------------------------------------------------------------------

def test_parse_column_blocks_splits_on_blank_lines(tmp_path):
    path = tmp_path / "x.out"
    path.write_text("1.0 2.0\n3.0 4.0\n\n5.0 6.0\n\n")
    blocks = parse_column_blocks(path, ncols=2)
    assert len(blocks) == 2
    assert blocks[0].shape == (2, 2)
    assert blocks[1].shape == (1, 2)


def test_parse_column_blocks_rejects_ragged_rows(tmp_path):
    path = tmp_path / "x.out"
    path.write_text("1.0 2.0\n3.0 4.0 5.0\n")
    with pytest.raises(ValueError, match="ragged block"):
        parse_column_blocks(path)


# --------------------------------------------------------------------------
# task 450 -- A(t)
# --------------------------------------------------------------------------

def test_parse_afieldt_round_trip(tmp_path):
    times = 0.1 * np.arange(5)
    afield = np.column_stack([np.zeros(5), np.zeros(5), np.linspace(0, 0.05, 5)])
    path = tmp_path / "AFIELDT.OUT"
    with open(path, "w") as fh:
        fh.write(f"{len(times):8d} : number of time steps\n")
        for i, (t, a) in enumerate(zip(times, afield), start=1):
            fh.write(f"{i:8d}{t:18.10G}{a[0]:18.10G}{a[1]:18.10G}{a[2]:18.10G}\n")

    parsed_t, parsed_a = parse_afieldt(path)
    assert parsed_t == pytest.approx(times)
    assert parsed_a == pytest.approx(afield)


def test_parse_afieldt_checks_declared_ntimes(tmp_path):
    path = tmp_path / "AFIELDT.OUT"
    path.write_text("       9 : number of time steps\n"
                    "       1   0.000000000       0.000000000"
                    "       0.000000000       0.000000000    \n")
    with pytest.raises(ValueError, match="declares ntimes=9"):
        parse_afieldt(path)


def test_parse_afspt_unpacks_fortran_order(tmp_path):
    # afspt(3,3,ntimes) written as afspt(:,:,its): Cartesian index fastest,
    # spin index slowest, so afspt[a, j] multiplies sigma_j along axis a
    times = np.array([0.0, 0.1])
    frames = np.array([np.arange(9.0).reshape((3, 3), order="F"),
                       np.arange(9.0, 18.0).reshape((3, 3), order="F")])
    path = tmp_path / "AFSPT.OUT"
    with open(path, "w") as fh:
        fh.write(f"{len(times):8d} : number of time steps\n")
        for i, (t, frame) in enumerate(zip(times, frames), start=1):
            flat = frame.flatten(order="F")
            fh.write(f"{i:8d}{t:18.10G}" + "".join(f"{v:18.10G}" for v in flat) + "\n")

    parsed_t, parsed = parse_afspt(path)
    assert parsed_t == pytest.approx(times)
    assert parsed.shape == (2, 3, 3)
    assert parsed == pytest.approx(frames)


# --------------------------------------------------------------------------
# tasks 455/456
# --------------------------------------------------------------------------

def test_parse_afpdt_and_afted(tmp_path):
    times = 0.1 * np.arange(4)
    power = np.array([0.0, 1e-14, 4e-14, 9e-14])
    afpdt = tmp_path / "AFPDT.OUT"
    with open(afpdt, "w") as fh:
        for t, p in zip(times, power):
            fh.write(f"{t:18.10G}{p:18.10G}\n")
    parsed_t, parsed_p = parse_afpdt(afpdt)
    assert parsed_t == pytest.approx(times)
    assert parsed_p == pytest.approx(power)

    afted = tmp_path / "AFTED.OUT"
    afted.write_text("\nTotal energy density :   0.1775534501E-05\n"
                     " in J/cm²            :   0.2764317418E-06\n")
    # the J/cm^2 line must NOT be picked up -- it carries a different number
    assert parse_afted(afted) == pytest.approx(1.775534501e-06)


def test_parse_efieldw_three_direction_blocks(tmp_path):
    energies = 0.05 * np.arange(4)
    values = np.array([
        [1 + 1j, 2 - 1j, 3 + 0j, 4 + 2j],
        [0 + 0j, 1 + 1j, 0 - 1j, 2 + 0j],
        [5 + 5j, 0 + 0j, 1 - 1j, 0 + 3j],
    ])
    path = tmp_path / "EFIELDW.OUT"
    with open(path, "w") as fh:
        for i in range(3):
            for w, z in zip(energies, values[i]):
                fh.write(f"{w:18.10G}{z.real:18.10G}{z.imag:18.10G}\n")
            fh.write("\n")

    parsed_w, parsed = parse_efieldw(path)
    assert parsed_w == pytest.approx(energies)
    assert parsed.shape == (4, 3)
    assert parsed == pytest.approx(values.T)
    # JTOTW.OUT has the identical shape (src/dielectric_tdrt.f90)
    assert parse_jtotw(path)[1] == pytest.approx(values.T)


def test_parse_efieldw_rejects_wrong_block_count(tmp_path):
    path = tmp_path / "EFIELDW.OUT"
    path.write_text("0.0 1.0 2.0\n\n0.0 1.0 2.0\n\n")
    with pytest.raises(ValueError, match="expected 3 blocks"):
        parse_efieldw(path)


# --------------------------------------------------------------------------
# tasks 460-463 -- observable time series
# --------------------------------------------------------------------------

def test_parse_time_series_scalar_and_vector(tmp_path):
    times = 0.1 * np.arange(3)
    energy = np.array([-580.1, -580.0, -579.9])
    path = tmp_path / "TOTENERGY_TD.OUT"
    with open(path, "w") as fh:
        for t, e in zip(times, energy):
            fh.write(f"{t:18.10G}{e:18.10G}\n")
    parsed_t, parsed_e = parse_time_series(path, ncomponents=1)
    assert parsed_t == pytest.approx(times)
    assert parsed_e == pytest.approx(energy)

    current = np.array([[0.0, 0.0, 0.0], [1e-4, 0.0, 2e-4], [2e-4, 0.0, 4e-4]])
    path = tmp_path / "JTOT_TD.OUT"
    with open(path, "w") as fh:
        for t, j in zip(times, current):
            fh.write(f"{t:18.10G}{j[0]:18.10G}{j[1]:18.10G}{j[2]:18.10G}\n")
    parsed_t, parsed_j = parse_time_series(path, ncomponents=3)
    assert parsed_j == pytest.approx(current)


def test_parse_time_series_rejects_wrong_component_count(tmp_path):
    # MOMENT_TD.OUT carries ndmag columns (1 collinear, 3 non-collinear) --
    # asking for the wrong one must fail loudly, not silently reshape
    path = tmp_path / "MOMENT_TD.OUT"
    path.write_text("   0.000000000       2.200000000    \n")
    with pytest.raises(ValueError, match="expected 3 value columns"):
        parse_time_series(path, ncomponents=3)
    assert parse_time_series(path, ncomponents=1)[1] == pytest.approx([2.2])


def test_parse_atom_time_series(tmp_path):
    path = tmp_path / "CHARGEMT_TD.OUT"
    with open(path, "w") as fh:
        for step, (t, q1, q2) in enumerate(
            [(0.0, 11.9, 11.9), (0.1, 11.8, 12.0)]
        ):
            fh.write(f"{t:18.10G}\n")
            fh.write(f"{1:4d}{1:4d}{q1:18.10G}\n")
            fh.write(f"{1:4d}{2:4d}{q2:18.10G}\n")
            fh.write("\n")

    times, values, labels = parse_atom_time_series(path)
    assert times == pytest.approx([0.0, 0.1])
    assert labels == [(1, 1), (1, 2)]
    assert values.shape == (2, 2, 1)
    assert values[1, 1, 0] == pytest.approx(12.0)


# --------------------------------------------------------------------------
# two-block spectra (320, 330/331, 480/481)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "parser", [parse_tddft_response, parse_spin_response, parse_epsilon_tdrt]
)
def test_two_block_spectra_round_trip(tmp_path, parser):
    x = np.linspace(0.0, 0.5, 5)
    re = np.linspace(1.0, 3.0, 5)
    im = np.linspace(0.0, -2.0, 5)
    path = tmp_path / "SPECTRUM.OUT"
    _write_two_blocks(path, x, re, im)
    parsed_x, values = parser(path)
    assert parsed_x == pytest.approx(x)
    assert values == pytest.approx(re + 1j * im)
