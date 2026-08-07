
! Copyright (C) 2020 Peter Elliott, J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine genefieldt
use modmain
use modtddft
implicit none
! local variables
integer, parameter :: npm=8
integer np,it,i
real(8) t0
! automatic arrays
real(8) ya(npm)
! external functions
real(8), external :: polynm
! determine the electric field at the current time step
t0=-1.d0/solsc
np=min(npm,itimes)
it=itimes-np+1
do i=1,3
  ya(1:np)=afieldt(i,it:itimes)
  efieldt(i)=t0*polynm(1,np,times(it),ya,times(itimes))
end do
end subroutine

