
! Copyright (C) 2013 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine hmlaloq(is,ias,ngp,ngpq,apwalm,apwalmq,ld,hq)
use modmain
implicit none
! arguments
integer, intent(in) :: is,ias,ngp,ngpq
complex(8), intent(in) :: apwalm(ngkmax,apwordmax,lmmaxapw)
complex(8), intent(in) :: apwalmq(ngkmax,apwordmax,lmmaxapw)
integer, intent(in) :: ld
complex(8), intent(inout) :: hq(ld,*)
! local variables
integer io,ilo,i,j
integer l0,l1,l2,l3
integer lm1,lm3,lma,lmb
complex(8) z1
do ilo=1,nlorb(is)
  l1=lorbl(ilo,is)
  do lm1=l1**2+1,(l1+1)**2
    j=idxlo(lm1,ilo,ias)
    i=ngpq+j
    j=ngp+j
    do l3=0,lmaxapw
      l0=merge(0,1,mod(l1+l3,2) == 0)
      do lm3=l3**2+1,(l3+1)**2
        do io=1,apword(l3,is)
          z1=0.d0
          do l2=l0,lmaxo,2
            lma=l2**2+1; lmb=lma+2*l2
            z1=z1+sum(gntyry(lma:lmb,lm3,lm1)*hloa(lma:lmb,io,l3,ilo,ias))
          end do
          if (abs(z1%re)+abs(z1%im) > 1.d-12) then
            hq(1:ngpq,j)=hq(1:ngpq,j)+conjg(z1*apwalmq(1:ngpq,io,lm3))
            hq(i,1:ngp)=hq(i,1:ngp)+z1*apwalm(1:ngp,io,lm3)
          end if
        end do
      end do
    end do
  end do
end do
end subroutine

