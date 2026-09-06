"""Unit tests for elkpy.parsers.ulr (the ultra-long-range family).

Fixtures transcribed from the Fortran `write` statements in
vendor/elk/src/writedosu.f90, bandstrulr.f90 and plotu1d.f90 (cited inline).
Format-derived verification.
"""

import numpy as np
import pytest

from elkpy.parsers import ulr, volumetric


# vendor/elk/src/writedosu.f90:
#   open(50,file='TDOSULR.OUT',form='FORMATTED',action='WRITE')
#   do iw=1,nwplot
#     write(50,'(2G18.10)') w(iw),g(iw)
#   end do
TDOSULR_OUT = """ -0.5000000000       0.000000000
 -0.2500000000       1.250000000
  0.000000000       4.500000000
  0.2500000000       2.100000000
"""

# vendor/elk/src/bandstrulr.f90:
#   do ist=1,nstulr
#     do ik0=1,nkpt0
#       write(50,'(3G18.10)') dpp1d(ik0),evalu(ist,ik0),chkpa(ist,ik0)
#     end do
#     write(50,*)
#   end do
# opened with position='APPEND', so task 725 repeats this for every kappa.
BANDULR_OUT = """  0.000000000      -0.3000000000       0.9500000000
  0.5000000000     -0.2500000000       0.9300000000
  1.000000000      -0.2000000000       0.9100000000

  0.000000000       0.1000000000       0.4000000000
  0.5000000000      0.1500000000       0.4200000000
  1.000000000      0.2000000000       0.4400000000

"""

# vendor/elk/src/bandstrulr.f90:
#   open(50,file='BANDSFU.OUT',form='FORMATTED')
#   write(50,'(2I6," : grid size")') nkpt0,nwplot
#   do iw=1,nwplot
#     do ik0=1,nkpt0
#       write(50,'(3G18.10)') dpp1d(ik0),w(iw),sfu(iw,ik0)
#     end do
#   end do
# NOTE the header names nkpt0 first while the loops put iw on the OUTSIDE.
BANDSFU_OUT = """     3     2 : grid size
  0.000000000      -0.5000000000       0.1000000000
  0.5000000000     -0.5000000000       0.2000000000
  1.000000000      -0.5000000000       0.3000000000
  0.000000000       0.5000000000       0.4000000000
  0.5000000000      0.5000000000       0.5000000000
  1.000000000      0.5000000000       0.6000000000
"""

# vendor/elk/src/plotu1d.f90:
#   do ip=1,npp1d
#     write(fnum1,'(5G18.10)') dpp1d(ip),(fp(ip,jf),jf=1,nf)
#   end do
# no header line, unlike the 2D/3D members of the family.
MAGU1D_OUT = """  0.000000000       0.000000000       0.000000000       0.5000000000
  1.000000000       0.1000000000      -0.2000000000       0.4000000000
  2.000000000       0.2000000000      -0.4000000000       0.3000000000
"""

RHOU1D_OUT = """  0.000000000       0.1234000000
  1.000000000       0.2345000000
  2.000000000       0.3456000000
"""

# vendor/elk/src/plotu1d.f90:
#   do iv=1,nvp1d
#     write(fnum2,'(2G18.10)') dvp1d(iv),fmin
#     write(fnum2,'(2G18.10)') dvp1d(iv),fmax
#     write(fnum2,*)
#   end do
RHOULINES_OUT = """  0.000000000      -0.1000000000
  0.000000000       0.5000000000

  2.000000000      -0.1000000000
  2.000000000       0.5000000000

"""

# vendor/elk/src/plotu3d.f90 -- identical layout to plot3d.f90, which is why
# elkpy.parsers.volumetric.parse_plot3d is reused rather than duplicated:
#   write(fnum,'(3I6," : grid size")') np3d(:)
#   do ip=1,np
#     call r3mv(avec,vpl(:,ip),v1)
#     write(fnum,'(7G18.10)') v1(:),(fp(ip,jf),jf=1,nf)
#   end do
RHOU3D_OUT = """     2     1     1 : grid size
  0.000000000       0.000000000       0.000000000       0.1000000000
  2.565000000       2.565000000       0.000000000       0.2000000000
"""


def test_parse_ulr_dos(tmp_path):
    path = tmp_path / "TDOSULR.OUT"
    path.write_text(TDOSULR_OUT)
    energies, dos = ulr.parse_ulr_dos(path)
    assert energies == pytest.approx(np.array([-0.5, -0.25, 0.0, 0.25]))
    assert dos == pytest.approx(np.array([0.0, 1.25, 4.5, 2.1]))


def test_parse_ulr_bands_reads_three_columns(tmp_path):
    """BANDULR.OUT carries a third column (the kappa-point character), so
    elkpy.parsers.band.parse_bands -- which unpacks exactly two fields --
    cannot be reused here."""
    path = tmp_path / "BANDULR.OUT"
    path.write_text(BANDULR_OUT)
    distances, energies, characters = ulr.parse_ulr_bands(path)
    assert distances == pytest.approx(np.array([0.0, 0.5, 1.0]))
    assert energies.shape == (2, 3)
    assert energies[0] == pytest.approx(np.array([-0.3, -0.25, -0.2]))
    assert characters[0] == pytest.approx(np.array([0.95, 0.93, 0.91]))
    assert characters[1] == pytest.approx(np.array([0.40, 0.42, 0.44]))


def test_parse_ulr_bands_splits_appended_kappa_passes(tmp_path):
    """Task 725 opens BANDULR.OUT with position='APPEND' once per
    kappa-point, so the blocks come in nkpa consecutive groups."""
    path = tmp_path / "BANDULR.OUT"
    path.write_text(BANDULR_OUT + BANDULR_OUT)
    _distances, energies, characters = ulr.parse_ulr_bands(path, nkappa=2)
    assert energies.shape == (2, 2, 3)
    assert characters.shape == (2, 2, 3)
    assert energies[0] == pytest.approx(energies[1])


def test_parse_ulr_bands_rejects_an_inconsistent_nkappa(tmp_path):
    path = tmp_path / "BANDULR.OUT"
    path.write_text(BANDULR_OUT)
    with pytest.raises(ValueError, match="not a multiple"):
        ulr.parse_ulr_bands(path, nkappa=4)


def test_parse_ulr_band_spectral_reshapes_against_the_loop_order(tmp_path):
    """Header is (nkpt0, nwplot) = (3, 2) but the frequency loop is on the
    OUTSIDE, so the block is (nwplot, nkpt0) = (2, 3); the fixture is
    deliberately non-square so a transposed reshape would fail."""
    path = tmp_path / "BANDSFU.OUT"
    path.write_text(BANDSFU_OUT)
    distances, frequencies, sf = ulr.parse_ulr_band_spectral(path)
    assert distances == pytest.approx(np.array([0.0, 0.5, 1.0]))
    assert frequencies == pytest.approx(np.array([-0.5, 0.5]))
    assert sf.shape == (2, 3)
    assert sf[0] == pytest.approx(np.array([0.1, 0.2, 0.3]))
    assert sf[1] == pytest.approx(np.array([0.4, 0.5, 0.6]))


def test_parse_plotu1d_scalar_field(tmp_path):
    path = tmp_path / "RHOU1D.OUT"
    path.write_text(RHOU1D_OUT)
    distances, values = ulr.parse_plotu1d(path, nf=1)
    assert distances == pytest.approx(np.array([0.0, 1.0, 2.0]))
    assert values == pytest.approx(np.array([0.1234, 0.2345, 0.3456]))
    assert values.ndim == 1


def test_parse_plotu1d_vector_field(tmp_path):
    """Non-collinear magnetisation (ndmag = 3) puts three function columns
    after the distance."""
    path = tmp_path / "MAGU1D.OUT"
    path.write_text(MAGU1D_OUT)
    distances, values = ulr.parse_plotu1d(path, nf=3)
    assert distances == pytest.approx(np.array([0.0, 1.0, 2.0]))
    assert values.shape == (3, 3)
    assert values[1] == pytest.approx(np.array([0.1, -0.2, 0.4]))


def test_parse_plotu_lines(tmp_path):
    path = tmp_path / "RHOULINES.OUT"
    path.write_text(RHOULINES_OUT)
    assert ulr.parse_plotu_lines(path) == pytest.approx(np.array([0.0, 2.0]))


def test_ulr_3d_plot_reuses_the_ordinary_plot3d_reader(tmp_path):
    """plotu3d.f90 and plot3d.f90 emit the identical header and row layout,
    so the ULR 3D plots need no separate reader."""
    path = tmp_path / "RHOU3D.OUT"
    path.write_text(RHOU3D_OUT)
    points, values = volumetric.parse_plot3d(path, nf=1)
    assert points.shape == (2, 3)
    assert points[1] == pytest.approx(np.array([2.565, 2.565, 0.0]))
    assert values == pytest.approx(np.array([0.1, 0.2]))
