"""Unit tests for elkpy.parsers.gw.

Every fixture string is transcribed from the Fortran `write` statement that
produces it (file and statement cited above each fixture). Format-derived
verification: it pins the layout against the source, not the numbers against
a real GW run (which is hours to days).
"""

import numpy as np
import pytest

from elkpy.parsers import gw


# vendor/elk/src/writegwefm.f90:
#   open(50,file='GWEFERMI.OUT',form='FORMATTED',action='WRITE')
#   write(50,'(G18.10)') efermi
GWEFERMI_OUT = "  0.2035600000\n"

# vendor/elk/src/gwspecf.f90:
#   open(50,file='GWTSF.OUT',form='FORMATTED')
#   do iw=1,nwplot
#     write(50,'(2G18.10)') wr(iw),sft(iw)
#   end do
GWTSF_OUT = """ -0.8000000000       0.1000000000E-02
 -0.7000000000       0.5000000000E-02
 -0.6000000000       0.2500000000E-01
 -0.5000000000       0.1200000000
"""

# vendor/elk/src/writegwsf.f90:
#   write(fname,'("GWSF_K",I6.6,".OUT")') ik
#   do iw=1,nwplot
#     write(50,'(2G18.10)') wr(iw),sf(iw)
#   end do
#   write(50,*)
#   do ist=1,nstsv
#     e=evalsv(ist,ik)-efermi
#     write(50,'(2G18.10)') e,0.d0
#     write(50,'(2G18.10)') e,1.d0/swidth
#     write(50,*)
#   end do
GWSF_K_OUT = """ -0.8000000000       0.1000000000E-02
 -0.7000000000       0.5000000000E-02
 -0.6000000000       0.2500000000E-01

 -0.4500000000        0.000000000
 -0.4500000000        1000.000000

  0.2100000000        0.000000000
  0.2100000000        1000.000000

"""

# vendor/elk/src/gwbandstr.f90:
#   write(85,'(2I6," : grid size")') nwplot,npp1d
#   do ip=ip0gw,npp1d
#     do iw=1,nwplot
#       write(85,'(3G18.10)') dpp1d(ip),wr(iw),sf(iw)
#     end do
#   end do
# NOTE the header names nwplot FIRST but the loops put ip on the OUTSIDE.
GWBAND_OUT = """     3     2 : grid size
  0.000000000      -1.500000000       0.1000000000
  0.000000000       0.000000000       0.2000000000
  0.000000000       1.500000000       0.3000000000
  0.5000000000     -1.500000000       0.4000000000
  0.5000000000      0.000000000       0.5000000000
  0.5000000000      1.500000000       0.6000000000
"""


def test_parse_gw_fermi_energy(tmp_path):
    path = tmp_path / "GWEFERMI.OUT"
    path.write_text(GWEFERMI_OUT)
    assert gw.parse_gw_fermi_energy(path) == pytest.approx(0.20356)


def test_parse_gw_total_spectral_function(tmp_path):
    path = tmp_path / "GWTSF.OUT"
    path.write_text(GWTSF_OUT)
    w, sf = gw.parse_gw_total_spectral_function(path)
    assert w == pytest.approx(np.array([-0.8, -0.7, -0.6, -0.5]))
    assert sf == pytest.approx(np.array([1e-3, 5e-3, 2.5e-2, 0.12]))


def test_parse_gw_spectral_function_stops_at_the_blank_line(tmp_path):
    """writegwsf.f90 appends 2*nstsv Kohn-Sham marker points after a blank
    line; reading the whole file would silently splice them onto the
    spectrum."""
    path = tmp_path / "GWSF_K000001.OUT"
    path.write_text(GWSF_K_OUT)
    w, sf = gw.parse_gw_spectral_function(path)
    assert len(w) == 3
    assert w == pytest.approx(np.array([-0.8, -0.7, -0.6]))
    assert sf == pytest.approx(np.array([1e-3, 5e-3, 2.5e-2]))


def test_parse_gw_spectral_function_eigenvalues(tmp_path):
    """Each Kohn-Sham state is drawn as a vertical marker, i.e. the same
    abscissa twice; the eigenvalues are the distinct abscissae."""
    path = tmp_path / "GWSF_K000001.OUT"
    path.write_text(GWSF_K_OUT)
    energies = gw.parse_gw_spectral_function_eigenvalues(path)
    assert energies == pytest.approx(np.array([-0.45, 0.21]))


def test_parse_gw_band_reshapes_against_the_loop_order_not_the_header(tmp_path):
    """The header is (nwplot, npp1d) = (3, 2) but the data runs ip-outer,
    so the block is (npp1d, nwplot) = (2, 3). Getting this backwards would
    still reshape without error on a square grid, which is why the fixture
    is deliberately non-square."""
    path = tmp_path / "GWBAND.OUT"
    path.write_text(GWBAND_OUT)
    distances, frequencies, sf = gw.parse_gw_band(path)
    assert distances == pytest.approx(np.array([0.0, 0.5]))
    assert frequencies == pytest.approx(np.array([-1.5, 0.0, 1.5]))
    assert sf.shape == (2, 3)
    assert sf[0] == pytest.approx(np.array([0.1, 0.2, 0.3]))
    assert sf[1] == pytest.approx(np.array([0.4, 0.5, 0.6]))


def test_parse_gw_band_rejects_a_truncated_file(tmp_path):
    path = tmp_path / "GWBAND.OUT"
    path.write_text("\n".join(GWBAND_OUT.splitlines()[:5]) + "\n")
    with pytest.raises(ValueError, match="whole multiple"):
        gw.parse_gw_band(path)
