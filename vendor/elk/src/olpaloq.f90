
! Copyright (C) 2013 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine olpaloq(is,ias,ngp,ngpq,apwalm,apwalmq,ld,oq)
use modmain
implicit none
! arguments
integer, intent(in) :: is,ias,ngp,ngpq
complex(8), intent(in) :: apwalm(ngkmax,apwordmax,lmmaxapw)
complex(8), intent(in) :: apwalmq(ngkmax,apwordmax,lmmaxapw)
integer, intent(in) :: ld
complex(8), intent(inout) :: oq(ld,*)
! local variables
integer ilo,io,l,lm,i,j
real(8) t1
do ilo=1,nlorb(is)
  l=lorbl(ilo,is)
  do lm=l**2+1,(l+1)**2
    j=idxlo(lm,ilo,ias)
    i=ngpq+j
    j=ngp+j
    do io=1,apword(l,is)
      t1=oalo(io,ilo,ias)
      oq(1:ngpq,j)=oq(1:ngpq,j)+t1*conjg(apwalmq(1:ngpq,io,lm))
      oq(i,1:ngp)=oq(i,1:ngp)+t1*apwalm(1:ngp,io,lm)
    end do
  end do
end do
end subroutine

