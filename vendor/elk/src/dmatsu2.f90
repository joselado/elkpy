
! Copyright (C) 2025 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: dmatsu2
! !INTERFACE:
subroutine dmatsu2(lmmax,su2,dmat)
! !USES:
use modmain
! !INPUT/OUTPUT PARAMETERS:
!   lmmax : number of (l,m) components: (lmax+1)^2 (in,integer)
!   su2   : SU(2) rotation matrix (in,complex(2,2))
!   dmat  : density matrices for all second-variational states
!           (inout,complex(lmmax,nspinor,lmmax,nspinor,nstsv))
! !DESCRIPTION:
!   Applies a $SU(2)$ rotation matrix $U$ to the spin degrees of freedom of a
!   state-resolved muffin-tin density matrix. In other words given a density
!   matrix $\gamma$, this subroutine performs the operation
!   $$ \gamma(l,m,\sigma,l,m,\sigma')\rightarrow \sum_{\sigma_1,\sigma_2}
!    U(\sigma,\sigma_1)\gamma(l,m,\sigma_1,l,m,\sigma_2)
!    U^{\dag}(\sigma_2,\sigma'). $$
!   Note that the operation is performed only on the $(l,m)$ diagonal part of
!   the matrix. See the routines {\tt bandstr} and {\tt dos}.
!
! !REVISION HISTORY:
!   Created November 2025 (JKD)
!EOP
!BOC
implicit none
! arguments
integer, intent(in) :: lmmax
complex(8), intent(in) :: su2(2,2)
complex(8), intent(inout) :: dmat(lmmax,nspinor,lmmax,nspinor,nstsv)
! local variables
integer ist,lm
complex(8) b(2,2),c(2,2)
do ist=1,nstsv
  do lm=1,lmmax
! apply the SU(2) matrix as U γ U†
    b(1:2,1:2)=dmat(lm,1:2,lm,1:2,ist)
    call z2mm(su2,b,c)
    call z2mmct(c,su2,b)
    dmat(lm,1:2,lm,1:2,ist)=b(1:2,1:2)
  end do
end do
end subroutine
!EOC

