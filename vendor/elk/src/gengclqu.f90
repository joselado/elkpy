
! Copyright (C) 2025 Wenhan Chen, J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine gengclqu
use modmain
use modulr
implicit none
! local variables
integer iq
real(8) t1
gclq(1)=0.d0
do iq=2,nqpt
  t1=vqc(1,iq)**2+vqc(2,iq)**2+vqc(3,iq)**2
  if (q0cut >= 0.d0) then
! hard cut-off
    if (t1 > q0cut**2) then
      gclq(iq)=fourpi/t1
    else
      gclq(iq)=0.d0
    end if
  else
! Yukawa-type screening
    gclq(iq)=fourpi/(t1+q0cut**2)
  end if
end do
end subroutine

