
! Copyright (C) 2025 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine writegwefm
use modmain
use modgw
implicit none
! local variables
integer ik
! initialise universal variables
call init0
call init1
call init3
! read Fermi energy from file
call readefm
! get the eigenvalues from file
do ik=1,nkpt
  call getevalsv(filext,ik,vkl(:,ik),evalsv(:,ik))
end do
! determine the GW Fermi energy
call gwefermi
! write the GW Fermi energy to file
open(50,file='GWEFERMI.OUT',form='FORMATTED',action='WRITE')
write(50,'(G18.10)') efermi
close(50)
end subroutine

