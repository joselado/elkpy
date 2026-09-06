"""Unit tests for the band-character and partial-DOS parsers (no Elk run).

Fixtures transcribe the ``write`` statements of ``src/bandstr.f90`` (tasks
21-24, BAND_Sss_Aaaaa.OUT) and ``src/dos.f90`` (task 10, TDOS.OUT /
PDOS_Sss_Aaaaa.OUT / IDOS.OUT) -- FORMAT-DERIVED verification, not
binary-verified.

The checks with teeth are the two things a shape assertion cannot see: that
task 21's THIRD column is the sum over l and not the l=0 channel (reading it
as s-character would silently overstate every s weight by the whole
muffin-tin total), and that dos.f90 writes the SPIN-DOWN channel negated
(``sps(2) = -1``), so a naive sum over blocks returns the magnetisation
rather than the DOS.
"""

import math

import numpy as np
import pytest

from elkpy.parsers import character


def g(v, w=18, d=10):
    """One Fortran ``Gw.d`` field (see tests/test_parsers_plots.py)."""
    if v == 0.0:
        n = 0
    else:
        n = math.floor(math.log10(abs(v))) + 1
    if 0 <= n <= d:
        return f"{v:{w - 4}.{d - n}f}" + "    "
    return f"{v:{w}.{d - 1}E}"


def write_band_character(path, nbands, npoints, ncharacter, with_total):
    """src/bandstr.f90, tasks 21-24::

        do ist=1,nstsv
          do ik=1,nkpt
            write(50,'(2G18.10,F12.6)',advance='NO') dpp1d(ik),evalsv(ist,ik),sm
            do l=0,lmaxdb
              write(50,'(F12.6)',advance='NO') bc(l,ias,ist,ik)
            end do
            write(50,*)
          end do
          write(50,*)
        end do

    (tasks 22-24 drop the ``sm`` total column and write their own trailing
    count of ``F12.6`` fields; the block structure is identical.) Values
    encode their (band, point, column) indices.
    """
    with open(path, "w") as fh:
        for ist in range(nbands):
            for ik in range(npoints):
                distance = 0.5 * ik
                energy = -1.0 + ist + 0.01 * ik
                fh.write(g(distance) + g(energy))
                columns = [0.001 * (100 * ist + 10 * ik + j) for j in range(ncharacter)]
                if with_total:
                    fh.write(f"{sum(columns):12.6f}")
                fh.write("".join(f"{c:12.6f}" for c in columns))
                fh.write("\n")
            fh.write("\n")


def write_dos(path, nblocks, nw, sign_per_block=None):
    """src/dos.f90::

        do ispn=1,nsd
          do l=l0,l1
            do iw=1,nwplot
              write(50,'(2G18.10)') w(iw),dp(iw,l,ispn)*sps(ispn)
            end do
            write(50,*)

    with ``sps(1)=1; sps(2)=-1``.
    """
    with open(path, "w") as fh:
        for ib in range(nblocks):
            sign = 1.0 if sign_per_block is None else sign_per_block[ib]
            for iw in range(nw):
                w = -0.5 + iw * 0.1
                value = sign * (ib + 1) * (iw + 1)
                fh.write(g(w) + g(value) + "\n")
            fh.write("\n")


def test_parse_band_character_l(tmp_path):
    """Task 21: distance, energy, TOTAL, then lmaxdb+1 l channels."""
    path = tmp_path / "BAND_S01_A0001.OUT"
    write_band_character(path, nbands=3, npoints=4, ncharacter=4, with_total=True)
    result = character.parse_band_character(path, kind="l")
    assert result["distances"].shape == (4,)
    assert result["energies"].shape == (3, 4)
    assert result["characters"].shape == (3, 4, 5)  # total + s,p,d,f
    assert result["l"].shape == (3, 4, 4)
    assert result["lmaxdb"] == 3
    np.testing.assert_allclose(result["distances"], [0.0, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(result["energies"][1], -1.0 + 1 + 0.01 * np.arange(4))


def test_parse_band_character_total_is_the_sum_over_l_not_the_s_channel(tmp_path):
    """The pin that matters: bandstr.f90 writes ``sm = sum(bc(0:lmaxdb,...))``
    BEFORE the per-l columns, so column 3 is the atom's whole muffin-tin
    weight. Mistaking it for l=0 would inflate every reported s character."""
    path = tmp_path / "BAND_S01_A0001.OUT"
    write_band_character(path, nbands=2, npoints=3, ncharacter=4, with_total=True)
    result = character.parse_band_character(path, kind="l")
    np.testing.assert_allclose(
        result["total"], result["l"].sum(axis=2), rtol=0, atol=1e-6
    )
    assert not np.allclose(result["total"], result["l"][:, :, 0])


def test_parse_band_character_lm(tmp_path):
    """Task 22: distance, energy, then (lmaxdb+1)^2 (l,m) channels."""
    path = tmp_path / "BAND_S01_A0001.OUT"
    write_band_character(path, nbands=2, npoints=3, ncharacter=16, with_total=False)
    result = character.parse_band_character(path, kind="lm")
    assert result["characters"].shape == (2, 3, 16)
    assert result["lmaxdb"] == 3


def test_parse_band_character_spin_and_moment(tmp_path):
    """Task 23 writes two columns (up, down); task 24 writes ndmag."""
    spin = tmp_path / "spin.OUT"
    write_band_character(spin, nbands=2, npoints=3, ncharacter=2, with_total=False)
    assert character.parse_band_character(spin, kind="spin")["characters"].shape == (
        2, 3, 2
    )
    moment = tmp_path / "moment.OUT"
    write_band_character(moment, nbands=2, npoints=3, ncharacter=3, with_total=False)
    assert character.parse_band_character(moment, kind="moment")["characters"].shape == (
        2, 3, 3
    )


def test_parse_band_character_rejects_an_unknown_kind(tmp_path):
    path = tmp_path / "BAND_S01_A0001.OUT"
    write_band_character(path, nbands=1, npoints=2, ncharacter=2, with_total=False)
    with pytest.raises(ValueError, match="unknown band-character kind"):
        character.parse_band_character(path, kind="j")


def test_parse_dos_blocks(tmp_path):
    path = tmp_path / "TDOS.OUT"
    write_dos(path, nblocks=2, nw=5)
    energies, blocks = character.parse_dos_blocks(path)
    assert energies.shape == (5,)
    assert blocks.shape == (2, 5)
    np.testing.assert_allclose(blocks[1], 2.0 * np.arange(1, 6))


def test_parse_partial_dos_splits_spin_slowest(tmp_path):
    """dos.f90's loop is ``do ispn`` OUTSIDE ``do l``, so with nsd=2 and
    four l channels the eight blocks are (up s,p,d,f) then (dn s,p,d,f)."""
    path = tmp_path / "PDOS_S01_A0001.OUT"
    write_dos(path, nblocks=8, nw=3, sign_per_block=[1, 1, 1, 1, -1, -1, -1, -1])
    energies, dos = character.parse_partial_dos(path, nspin=2)
    assert energies.shape == (3,)
    assert dos.shape == (2, 4, 3)


def test_parse_partial_dos_undoes_the_spin_down_negation(tmp_path):
    """``sps(2) = -1``: Elk writes the spin-down channel negative so a
    plotter can mirror it about the axis. Summing the raw blocks therefore
    gives the MAGNETISATION, not the DOS -- the parser flips it back."""
    path = tmp_path / "PDOS_S01_A0001.OUT"
    write_dos(path, nblocks=4, nw=3, sign_per_block=[1, 1, -1, -1])
    _e, raw = character.parse_partial_dos(path, nspin=2, undo_spin_sign=False)
    assert (raw[1] < 0).all()
    _e, dos = character.parse_partial_dos(path, nspin=2)
    assert (dos >= 0).all()
    np.testing.assert_allclose(dos[1], -raw[1])


def test_parse_partial_dos_rejects_an_inconsistent_spin_count(tmp_path):
    path = tmp_path / "PDOS_S01_A0001.OUT"
    write_dos(path, nblocks=3, nw=2)
    with pytest.raises(ValueError, match="not divisible by nspin"):
        character.parse_partial_dos(path, nspin=2)
