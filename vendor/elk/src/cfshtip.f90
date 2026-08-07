
! Copyright (C) 2023 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine cfshtip(nr,nri,cfmt)
use modmain
implicit none
! arguments
integer, intent(in) :: nr,nri
complex(4), intent(inout) :: cfmt(*)
! local variables
integer nro,npi,np,i
nro=nr-nri
npi=lmmaxi*nri
np=npi+lmmaxo*nro
i=npi+1
block
complex(4) f(np)
f(1:np)=cfmt(1:np)
! transform the inner part of the muffin-tin function in-place
call cgemm('N','N',lmmaxi,nri,lmmaxi,cone,cfshti,lmmaxi,f,lmmaxi,czero,cfmt, &
 lmmaxi)
! transform the outer part of the muffin-tin function in-place
call cgemm('N','N',lmmaxo,nro,lmmaxo,cone,cfshto,lmmaxo,f(i),lmmaxo,czero, &
 cfmt(i),lmmaxo)
end block
end subroutine

