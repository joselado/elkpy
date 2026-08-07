
! Copyright (C) 2012 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

pure subroutine dmatch(ias,ip,ngp,vgpc,apwalm,dapwalm)
use modmain
implicit none
! arguments
integer, intent(in) :: ias,ip,ngp
real(8), intent(in) :: vgpc(3,ngkmax)
complex(8), intent(in) :: apwalm(ngkmax,apwordmax,lmmaxapw,natmtot)
complex(8), intent(out) :: dapwalm(ngkmax,apwordmax,lmmaxapw)
! local variables
integer is,l,lm,io
! take derivative with respect to atomic displacement
is=idxis(ias)
do l=0,lmaxapw
  do lm=l**2+1,(l+1)**2
    do io=1,apword(l,is)
      dapwalm(1:ngp,io,lm)=vgpc(ip,1:ngp)*zi*apwalm(1:ngp,io,lm,ias)
    end do
  end do
end do
end subroutine

