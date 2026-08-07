
! Copyright (C) 2024 Wenhan Chen, J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine getevalu(ik0)
use modmain
use modulr
implicit none
! arguments
integer, intent(in) :: ik0
! local variables
integer ik,recl,nstulr_
real(8) vkl_(3),t1
! central k-point
ik=(ik0-1)*nkpa+1
! find the record length
inquire(iolength=recl) vkl_,nstulr_,evalu(1:nstulr,ik0)
!$OMP CRITICAL(u304)
open(304,file='EVALU.OUT',form='UNFORMATTED',access='DIRECT',recl=recl)
read(304,rec=ik0) vkl_,nstulr_,evalu(1:nstulr,ik0)
close(304)
!$OMP END CRITICAL(u304)
t1=abs(vkl(1,ik)-vkl_(1))+abs(vkl(2,ik)-vkl_(2))+abs(vkl(3,ik)-vkl_(3))
if (t1 > epslat) then
  write(*,*)
  write(*,'("Error(getevalu): differing vectors for k-point ",I0)') ik
  write(*,'(" current   : ",3G18.10)') vkl(:,ik)
  write(*,'(" EVALU.OUT : ",3G18.10)') vkl_
  write(*,*)
  stop
end if
if (nstulr /= nstulr_) then
  write(*,*)
  write(*,'("Error(getevalu): differing nstulr for central k-point ",I0)') ik0
  write(*,'(" current   : ",I0)') nstulr
  write(*,'(" EVALU.OUT : ",I0)') nstulr_
  write(*,*)
  stop
end if
end subroutine

