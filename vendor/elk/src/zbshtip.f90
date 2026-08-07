
! Copyright (C) 2023 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine zbshtip(nr,nri,zfmt)
use modmain
implicit none
! arguments
integer, intent(in) :: nr,nri
complex(8), intent(inout) :: zfmt(*)
! local variables
integer nro,npi,np,i
nro=nr-nri
npi=lmmaxi*nri
np=npi+lmmaxo*nro
i=npi+1
block
complex(8) f(np)
f(1:np)=zfmt(1:np)
! transform the inner part of the muffin-tin function in-place
call zgemm('N','N',lmmaxi,nri,lmmaxi,zone,zbshti,lmmaxi,f,lmmaxi,zzero,zfmt, &
 lmmaxi)
! transform the outer part of the muffin-tin function in-place
call zgemm('N','N',lmmaxo,nro,lmmaxo,zone,zbshto,lmmaxo,f(i),lmmaxo,zzero, &
 zfmt(i),lmmaxo)
end block
end subroutine

