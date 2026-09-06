"""Unit tests for the Fermi-surface family parsers (no Elk run).

Fixtures are Python transcriptions of the ``write`` statements in
``src/fermisurf.f90``, ``src/fermisurfbxsf.f90`` and ``src/nesting.f90``,
cited on each writer -- FORMAT-DERIVED verification, not binary-verified.

Two things here are worth more than shape checks:

* the two DIFFERENT header/record shapes fermisurf writes (a 3-integer
  header plus one scalar column for tasks 100/103, a 4-integer header plus
  one column per band for tasks 101/104), which a parser that assumed one
  of them would silently misread as the other;
* the AXIS ORDER, which is i1-fastest in FERMISURF.OUT and NEST3D.OUT (like
  every plot3d-family file) but i3-fastest in the ``.bxsf`` band grid.
  Getting either backwards permutes a volume without changing a number.
"""

import math

import numpy as np
import pytest

from elkpy.parsers import fermisurface


def g(v, w=18, d=10):
    """One Fortran ``Gw.d`` field (see tests/test_parsers_plots.py)."""
    if v == 0.0:
        n = 0
    else:
        n = math.floor(math.log10(abs(v))) + 1
    if 0 <= n <= d:
        return f"{v:{w - 4}.{d - n}f}" + "    "
    return f"{v:{w}.{d - 1}E}"


def write_fermisurf_scalar(path, n1, n2, n3):
    """src/fermisurf.f90, tasks 100 and 103::

        write(50,'(3I6," : grid size")') np3d(:)
        do i3=0,ngridk(3)-1
          do i2=0,ngridk(2)-1
            do i1=0,ngridk(1)-1
              write(50,'(4G18.10)') vpc(:,i),fn
    """
    with open(path, "w") as fh:
        fh.write(f"{n1:6d}{n2:6d}{n3:6d} : grid size\n")
        for i3 in range(n3):
            for i2 in range(n2):
                for i1 in range(n1):
                    row = [float(i1), float(i2), float(i3), i1 + 10.0 * i2 + 100.0 * i3]
                    fh.write("".join(g(v) for v in row) + "\n")


def write_fermisurf_bands(path, n1, n2, n3, nst, value=None):
    """src/fermisurf.f90, tasks 101 and 104::

        write(50,'(4I6," : grid size, number of states")') np3d(:),ist1-ist0+1
        ...
              write(50,'(3G18.10)',advance='NO') vpc(:,i)
              do ist=ist0,ist1
                write(50,'(F14.8)',advance='NO') evalsv(ist,ik)-efermi
              end do
              write(50,*)

    Note the per-band columns carry NO separator of their own -- ``F14.8``
    supplies whatever leading blanks are left over.
    """
    if value is None:
        def value(i1, i2, i3, ist):
            return i1 + 10.0 * i2 + 100.0 * i3 + 1000.0 * ist
    with open(path, "w") as fh:
        fh.write(f"{n1:6d}{n2:6d}{n3:6d}{nst:6d} : grid size, number of states\n")
        for i3 in range(n3):
            for i2 in range(n2):
                for i1 in range(n1):
                    line = "".join(g(v) for v in (float(i1), float(i2), float(i3)))
                    line += "".join(
                        f"{value(i1, i2, i3, ist):14.8f}" for ist in range(nst)
                    )
                    fh.write(line + "\n")


def write_bxsf(path, ngridk, bands, bvec):
    """src/fermisurfbxsf.f90, task 102. Note the grid written is
    ``ngridk(:)+1`` and that the band loop runs i1 OUTERMOST, i3 innermost
    -- the opposite of every plot3d-family file."""
    n = [g_ + 1 for g_ in ngridk]
    with open(path, "w") as fh:
        fh.write(" BEGIN_INFO\n")
        fh.write(" # Band-XCRYSDEN-Structure-File for Fermi surface plotting\n")
        fh.write(" # created by Elk version 11.0.2\n")
        fh.write(" # Launch as: xcrysden --bxsf FERMISURF(_UP/_DN).bxsf\n")
        fh.write("   Fermi Energy: " + g(0.0) + "\n")
        fh.write(" END_INFO\n")
        fh.write(" BEGIN_BLOCK_BANDGRID_3D\n")
        fh.write(" band_energies\n")
        fh.write(" BANDGRID_3D_BANDS\n")
        fh.write(f"{len(bands):4d}\n")
        fh.write(f"{n[0]:6d}{n[1]:6d}{n[2]:6d}\n")
        fh.write("".join(g(v) for v in (0.0, 0.0, 0.0)) + "\n")
        for row in bvec:
            fh.write("".join(g(v) for v in row) + "\n")
        for ist in bands:
            fh.write(f" BAND: {ist:4d}\n")
            for i1 in range(n[0]):
                for i2 in range(n[1]):
                    for i3 in range(n[2]):
                        fh.write(g(i1 + 10.0 * i2 + 100.0 * i3 + 1000.0 * ist) + "\n")
        fh.write(" END_BANDGRID_3D\n")
        fh.write(" END_BLOCK_BANDGRID_3D\n")


def write_nesting(nest3d, nesting, n1, n2, n3, total):
    """src/nesting.f90::

        write(50,'(3I6," : grid size")') ngridq(:)
        ... write(50,'(4G18.10)') vc(:),nq(iq)      ! NEST3D.OUT
        write(50,'(G18.10)') sm0                    ! NESTING.OUT
    """
    with open(nest3d, "w") as fh:
        fh.write(f"{n1:6d}{n2:6d}{n3:6d} : grid size\n")
        for i3 in range(n3):
            for i2 in range(n2):
                for i1 in range(n1):
                    row = [0.1 * i1, 0.2 * i2, 0.3 * i3, i1 + 10.0 * i2 + 100.0 * i3]
                    fh.write("".join(g(v) for v in row) + "\n")
    with open(nesting, "w") as fh:
        fh.write(g(total) + "\n")


def test_parse_fermisurf_scalar_form(tmp_path):
    path = tmp_path / "FERMISURF.OUT"
    write_fermisurf_scalar(path, 3, 4, 2)
    result = fermisurface.parse_fermisurf(path)
    assert result["grid"] == (3, 4, 2)
    assert result["per_band"] is False
    assert result["nstates"] is None
    assert result["values"].shape == (24,)
    assert result["points"].shape == (24, 3)


def test_parse_fermisurf_scalar_axis_order(tmp_path):
    """i1 innermost, as in every plot3d-family file."""
    from elkpy.parsers import plots

    path = tmp_path / "FERMISURF.OUT"
    write_fermisurf_scalar(path, 3, 4, 2)
    result = fermisurface.parse_fermisurf(path)
    volume = plots.reshape_plot3d(result["values"], result["grid"])
    assert volume.shape == (2, 4, 3)
    assert volume[1, 3, 2] == pytest.approx(2 + 30 + 100)


def test_parse_fermisurf_per_band_form(tmp_path):
    path = tmp_path / "FERMISURF.OUT"
    write_fermisurf_bands(path, 2, 2, 2, nst=3)
    result = fermisurface.parse_fermisurf(path)
    assert result["grid"] == (2, 2, 2)
    assert result["nstates"] == 3
    assert result["per_band"] is True
    assert result["values"].shape == (8, 3)
    # last point (i1,i2,i3)=(1,1,1), third band (ist index 2)
    assert result["values"][-1, 2] == pytest.approx(1 + 10 + 100 + 2000)


def test_parse_fermisurf_per_band_survives_touching_f14_8_columns(tmp_path):
    """``F14.8`` supplies its own padding, so a value needing all fourteen
    characters butts straight against its neighbour with no space. Whitespace
    splitting then under-counts the columns; the parser must fall back to
    fixed-width slicing rather than raise on a file Elk considers well
    formed."""
    path = tmp_path / "FERMISURF.OUT"

    def value(i1, i2, i3, ist):
        # 99999.99999999 fills all fourteen characters, so two adjacent ones
        # run together with no separating blank at all
        return 99999.99999999 if ist < 2 else 3.0

    write_fermisurf_bands(path, 2, 1, 1, nst=3, value=value)
    with open(path) as fh:
        fh.readline()
        first = fh.readline().rstrip("\n")
    assert len(first.split()) != 6, "fixture no longer exercises the touching case"
    result = fermisurface.parse_fermisurf(path)
    assert result["values"].shape == (2, 3)
    np.testing.assert_allclose(
        result["values"][0], [99999.99999999, 99999.99999999, 3.0]
    )


def test_parse_bxsf(tmp_path):
    path = tmp_path / "FERMISURF.bxsf"
    bvec = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    write_bxsf(path, (2, 3, 4), bands=[5, 6], bvec=bvec)
    result = fermisurface.parse_bxsf(path)
    assert result["grid"] == (3, 4, 5)  # ngridk + 1 in each direction
    assert result["band_indices"] == [5, 6]
    assert result["fermi_energy"] == pytest.approx(0.0)
    np.testing.assert_allclose(result["origin"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(result["bvec"], bvec)
    assert result["energies"].shape == (2, 3, 4, 5)


def test_parse_bxsf_axis_order_is_i3_fastest(tmp_path):
    """The bxsf band loop runs i1 outermost, so the LAST index is fastest --
    the opposite of FERMISURF.OUT. Values encode their indices, so a
    transposed read is a wrong number here."""
    path = tmp_path / "FERMISURF.bxsf"
    write_bxsf(path, (2, 3, 4), bands=[5], bvec=np.eye(3).tolist())
    result = fermisurface.parse_bxsf(path)
    energies = result["energies"][0]
    for i1 in range(3):
        for i2 in range(4):
            for i3 in range(5):
                assert energies[i1, i2, i3] == pytest.approx(
                    i1 + 10 * i2 + 100 * i3 + 5000
                )


def test_parse_nesting(tmp_path):
    nest3d = tmp_path / "NEST3D.OUT"
    nesting = tmp_path / "NESTING.OUT"
    write_nesting(nest3d, nesting, 2, 3, 2, total=1.2345)
    result = fermisurface.parse_nesting(nest3d, nesting)
    assert result["grid"] == (2, 3, 2)
    assert result["values"].shape == (12,)
    assert result["total"] == pytest.approx(1.2345)
    assert result["points"][-1] == pytest.approx([0.1, 0.4, 0.3])


def test_parse_nesting_without_the_integral_file(tmp_path):
    nest3d = tmp_path / "NEST3D.OUT"
    write_nesting(nest3d, tmp_path / "NESTING.OUT", 2, 2, 2, total=0.0)
    result = fermisurface.parse_nesting(nest3d)
    assert result["total"] is None
