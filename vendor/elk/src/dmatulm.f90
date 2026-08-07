
! Copyright (C) 2025 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: dmatulm
! !INTERFACE:
subroutine dmatulm(ulm,dmat)
! !USES:
use modmain
! !INPUT/OUTPUT PARAMETERS:
!   ulm  : unitary transformation in the (l,m) basis
!          (in,complex(lmmaxdb,lmmaxdb))
!   dmat : density matrix in the (l,m) and spin basis for each
!          second-variational state
!          (in,complex(lmmaxdb,nspinor,lmmaxdb,nspinor,nstsv))
! !DESCRIPTION:
!   Applies a unitary transformation to a density matrix corresponding to a
!   particular angular momentum $l$, atom and second-variational state. This is
!   done to transform the density matrix to irreducible representation form. See
!   also {\tt genlmirep}.
!
! !REVISION HISTORY:
!   Created October 2025 (JKD)
!EOP
!BOC
implicit none
! arguments
complex(8), intent(in) :: ulm(lmmaxdb,lmmaxdb)
complex(8), intent(inout) :: dmat(lmmaxdb,nspinor,lmmaxdb,nspinor,nstsv)
! local variables
integer ist,ispn,jspn,ld
! automatic arrays
complex(8) a(lmmaxdb,lmmaxdb)
ld=lmmaxdb*nspinor
do ist=1,nstsv
  do ispn=1,nspinor
    do jspn=1,nspinor
! apply the unitary matrix as U γ U†
      call zgemm('N','N',lmmaxdb,lmmaxdb,lmmaxdb,zone,ulm,lmmaxdb, &
       dmat(:,ispn,1,jspn,ist),ld,zzero,a,lmmaxdb)
      call zgemm('N','C',lmmaxdb,lmmaxdb,lmmaxdb,zone,a,lmmaxdb,ulm,lmmaxdb, &
       zzero,dmat(:,ispn,1,jspn,ist),ld)
    end do
  end do
end do
end subroutine
!EOC

