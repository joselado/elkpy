
! Copyright (C) 2025 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine finddlefe
use modmain
implicit none
! arguments
integer ik,ist,n
real(8) e,de
! determine the first energy moment of eigenvalues below the Fermi energy
de=0.d0
n=0
do ik=1,nkpt
  do ist=1,nstsv
    e=evalsv(ist,ik)-efermi
    if ((e <= 0.d0).and.(e > esccut)) then
      de=de+e
      n=n+1
    end if
  end do
end do
! set difference between fixed linearisation and Fermi energies
if (n > 0) dlefe=de/dble(n)
end subroutine

