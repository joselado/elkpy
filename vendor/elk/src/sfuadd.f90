
! Copyright (C) 2025 Wenhan Chen, J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine sfuadd(ik0,w,chkpa,sfu)
use modmain
use modulr
implicit none
! arguments
integer, intent(in) :: ik0
real(8), intent(in) :: w(nwplot),chkpa(nstulr)
real(8), intent(inout) :: sfu(nwplot)
! local variables
integer ist,iw
real(8) e,x,t0,t1
! external functions
real(8), external :: sdelta_lr
t0=1.d0/swidth
do ist=1,nstulr
  e=evalu(ist,ik0)-efermi
  t1=t0*chkpa(ist)
  do iw=1,nwplot
    x=(e-w(iw))*t0
    sfu(iw)=sfu(iw)+t1*sdelta_lr(x)
  end do
end do
end subroutine

