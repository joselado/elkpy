
! Copyright (C) 2024 Wenhan Chen, J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine putevalu(ik0)
use modmain
use modulr
implicit none
! arguments
integer, intent(in) :: ik0
! local variables
integer ik,recl
! central k-point
ik=(ik0-1)*nkpa+1
! find the record length
inquire(iolength=recl) vkl(1:3,ik),nstulr,evalu(1:nstulr,ik0)
!$OMP CRITICAL(u304)
open(304,file='EVALU.OUT',form='UNFORMATTED',access='DIRECT',recl=recl)
write(304,rec=ik0) vkl(1:3,ik),nstulr,evalu(1:nstulr,ik0)
close(304)
!$OMP END CRITICAL(u304)
end subroutine

