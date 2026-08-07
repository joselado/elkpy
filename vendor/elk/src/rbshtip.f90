
! Copyright (C) 2023 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine rbshtip(nr,nri,rfmt)
use modmain
implicit none
! arguments
integer, intent(in) :: nr,nri
real(8), intent(inout) :: rfmt(*)
! local variables
integer nro,npi,np,i
nro=nr-nri
npi=lmmaxi*nri
np=npi+lmmaxo*nro
i=npi+1
block
real(8) f(np)
f(1:np)=rfmt(1:np)
! transform the inner part of the muffin-tin function in-place
call dgemm('N','N',lmmaxi,nri,lmmaxi,1.d0,rbshti,lmmaxi,f,lmmaxi,0.d0,rfmt, &
 lmmaxi)
! transform the outer part of the muffin-tin function in-place
call dgemm('N','N',lmmaxo,nro,lmmaxo,1.d0,rbshto,lmmaxo,f(i),lmmaxo,0.d0, &
 rfmt(i),lmmaxo)
end block
end subroutine

