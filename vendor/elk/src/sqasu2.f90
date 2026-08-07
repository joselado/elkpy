
! Copyright (C) 2025 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine sqasu2(sqaxis,tsqaz,su2)
implicit none
! arguments
real(8), intent(in) :: sqaxis(3)
logical, intent(out) :: tsqaz
complex(8), intent(out) :: su2(2,2)
! local variables
real(8) v1(3),v2(3),v3(3),th,t1
v1(:)=sqaxis(:)
t1=sqrt(v1(1)**2+v1(2)**2+v1(3)**2)
if (t1 <= 1.d-8) then
  write(*,*)
  write(*,'("Error(sqasu2): spin-quantisation axis (sqaxis) has zero length")')
  write(*,*)
  stop
end if
v1(:)=v1(:)/t1
if (abs(v1(3)-1.d0) < 1.d-8) then
! spin-quantisation axis in +z direction
  tsqaz=.true.
  su2(1,1)=1.d0; su2(1,2)=0.d0
  su2(2,1)=0.d0; su2(2,2)=1.d0
else
! determine the SU(2) matrix corresponding to the rotation from +z to sqaxis
  tsqaz=.false.
  v2(1:2)=0.d0
  v2(3)=1.d0
  call r3cross(v1,v2,v3)
! note that the spin-quantisation axis is rotated, so the density matrix should
! be rotated in the opposite direction
  th=-acos(v1(3))
  call axangsu2(v3,th,su2)
end if
end subroutine

