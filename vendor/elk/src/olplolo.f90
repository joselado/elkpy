
! Copyright (C) 2002-2005 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

pure subroutine olplolo(is,ias,ngp,ld,o)
use modmain
implicit none
! arguments
integer, intent(in) :: is,ias,ngp,ld
complex(8), intent(out) :: o(ld,*)
! local variables
integer ilo,jlo,l,lm,i,j
do jlo=1,nlorb(is)
  l=lorbl(jlo,is)
  do ilo=1,jlo
    if (lorbl(ilo,is) /= l) cycle
    do lm=l**2+1,(l+1)**2
      i=ngp+idxlo(lm,ilo,ias)
      j=ngp+idxlo(lm,jlo,ias)
      o(i,j)=ololo(ilo,jlo,ias)
    end do
  end do
end do
end subroutine

