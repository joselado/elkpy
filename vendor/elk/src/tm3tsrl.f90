
! Copyright (C) 2026 S. Kalhoefer, L. Nordstrom and O. Granas.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: tm3tsrl
! !INTERFACE:
pure subroutine tm3tsrl(r,ld,wkpr_v,wkpr_t)
! !INPUT/OUTPUT PARAMETERS:
!   r      : r-index of tensor moment (in,integer)
!   ld     : leading dimension (in,integer)
!   wkpr_v : 3-index complex van der Laan components (in,complex(-ld:ld))
!   wkpr_t : real tesseral tensor moment components (out,real(-ld:ld))
! !DESCRIPTION:
!   Converts complex tensor moments in the van der Laan convention into real
!   tensor moments corresponding to tesseral spherical harmonics. See routines
!   {\tt tm3vdl} and {\tt tm3todm}.
!
! !REVISION HISTORY:
!   Created May 2026 (Sebastian Kalhoefer)
!EOP
!BOC
implicit none
! arguments
integer, intent(in) :: r,ld
complex(8), intent(in) :: wkpr_v(-ld:ld)
real(8), intent(out) :: wkpr_t(-ld:ld)
! local variables
integer t
real(8), parameter :: s2=sqrt(2.d0)
do t=-r,r
  if (t < 0) then
    wkpr_t(t)=s2*wkpr_v(t)%im
  else if (t == 0) then
    wkpr_t(t)=wkpr_v(t)%re
  else
    wkpr_t(t)=s2*wkpr_v(t)%re
  end if
end do
end subroutine
!EOC

