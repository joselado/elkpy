
! Copyright (C) 2023 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: rmtavrg
! !INTERFACE:
subroutine rmtavrg
! !USES:
use modmain
! !DESCRIPTION:
!   Performs a simple averaging procedure on all the muffin-tin radii. First
!   the average muffin-tin radius $\overline{R_{\rm MT}}$ is found and then
!   every radius is averaged with this and itself:
!   $R_{\rm MT}^{\alpha}\rightarrow(R_{\rm MT}^{\alpha}
!    +\overline{R_{\rm MT}})/2$. This is repeated {\tt mrmtav} times.
!
! !REVISION HISTORY:
!   Created May 2023 (JKD)
!EOP
!BOC
implicit none
! local variables
integer i
real(8) ra
if (nspecies <= 1) return
do i=1,mrmtav
! average muffin-tin radius
  ra=sum(rmt(1:nspecies))/nspecies
! replace each muffin-tin radius with half itself plus the average
  rmt(1:nspecies)=0.5d0*(rmt(1:nspecies)+ra)
end do
end subroutine
!EOC

