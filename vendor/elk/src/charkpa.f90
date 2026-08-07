
! Copyright (C) 2025 Wenhan Chen, J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

pure subroutine charkpa(ikpa,evecu,chkpa)
use modmain
use modulr
implicit none
! arguments
integer, intent(in) :: ikpa
complex(8), intent(in) :: evecu(nstulr,nstulr)
real(8), intent(out) :: chkpa(nstulr)
! local arguments
integer ist,jst,i
real(8) sm
do jst=1,nstulr
  sm=0.d0
  do ist=1,nstsv
    i=(ikpa-1)*nstsv+ist
    sm=sm+evecu(i,jst)%re**2+evecu(i,jst)%im**2
  end do
  chkpa(jst)=sm
end do
end subroutine

