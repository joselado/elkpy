"""Unit tests for elkpy.parsers.wannier90.

Fixtures transcribed from the Fortran `write` statements in
vendor/elk/src/writew90eig.f90 and writew90win.f90 (cited inline).
Format-derived verification.
"""

import numpy as np
import pytest

from elkpy.parsers import wannier90


# vendor/elk/src/writew90eig.f90:
#   do ik=1,nkptnr
#     jk=ivkik(ivk(1,ik),ivk(2,ik),ivk(3,ik))
#     do i=1,num_bands
#       ist=idxw90(i)
#       t1=evalsv(ist,jk)-efermi
#       write(50,'(2I6,G18.10)') i,ik,t1*ha_ev
#     end do
#   end do
# band index fastest, k-point index slowest, energies in eV.
WANNIER_EIG = """     1     1 -5.900000000
     2     1  0.000000000
     3     1  0.000000000
     1     2 -3.400000000
     2     2 -1.100000000
     3     2 -1.100000000
"""

# vendor/elk/src/writew90win.f90 (abbreviated to the blocks the parser reads):
#   write(50,'("length_unit = bohr")')
#   write(50,'("num_wann =  ",I8)') num_wann
#   write(50,'("num_bands = ",I8)') num_bands
#   write(50,'("begin unit_cell_cart")') / '("bohr")' / '(3G18.10)' avec(:,i)
#   write(50,'("begin atoms_frac")') / '(" ",A,3G18.10)' spsymb, atposl
#   write(50,'("mp_grid = ",3I6)') ngridk
#   write(50,'("begin kpoints")') / '(3G18.10)' vkl(:,ik)
WANNIER_WIN = """length_unit = bohr
num_wann =         3
num_bands =        3
num_iter =      100
dis_num_iter =      100

trial_step =   2.000000000

begin unit_cell_cart
bohr
  5.130000000       5.130000000       0.000000000
  5.130000000       0.000000000       5.130000000
  0.000000000       5.130000000       5.130000000
end unit_cell_cart

begin atoms_frac
 Si  0.000000000       0.000000000       0.000000000
 Si  0.2500000000      0.2500000000      0.2500000000
end atoms_frac

mp_grid =      2     2     2

begin kpoints
  0.000000000       0.000000000       0.000000000
  0.5000000000      0.000000000       0.000000000
end kpoints

"""


def test_parse_w90_eig(tmp_path):
    path = tmp_path / "wannier.eig"
    path.write_text(WANNIER_EIG)
    energies, nkpt, num_bands = wannier90.parse_w90_eig(path)
    assert (nkpt, num_bands) == (2, 3)
    assert energies.shape == (2, 3)
    # band index runs fastest, so row 0 is k-point 1's three bands
    assert energies[0] == pytest.approx(np.array([-5.9, 0.0, 0.0]))
    assert energies[1] == pytest.approx(np.array([-3.4, -1.1, -1.1]))


def test_parse_w90_eig_rejects_a_truncated_file(tmp_path):
    path = tmp_path / "wannier.eig"
    path.write_text("\n".join(WANNIER_EIG.splitlines()[:5]) + "\n")
    with pytest.raises(ValueError):
        wannier90.parse_w90_eig(path)


def test_parse_w90_win(tmp_path):
    path = tmp_path / "wannier.win"
    path.write_text(WANNIER_WIN)
    win = wannier90.parse_w90_win(path)
    assert win["settings"]["num_wann"] == "3"
    assert win["settings"]["mp_grid"] == "2     2     2"
    # the literal "bohr" units line inside the block is not a lattice vector
    assert win["unit_cell_cart"].shape == (3, 3)
    assert win["unit_cell_cart"][0] == pytest.approx(np.array([5.13, 5.13, 0.0]))
    assert [s for s, _ in win["atoms_frac"]] == ["Si", "Si"]
    assert win["atoms_frac"][1][1] == pytest.approx(np.array([0.25, 0.25, 0.25]))
    assert win["kpoints"].shape == (2, 3)
    assert win["kpoints"][1] == pytest.approx(np.array([0.5, 0.0, 0.0]))
