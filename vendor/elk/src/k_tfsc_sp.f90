
! Copyright (C) 2025 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine k_tfsc_sp(n,rhoup,rhodn,tauup,taudn,dtdru,dtdrd)
implicit none
! arguments
integer, intent(in) :: n
real(8), intent(in) :: rhoup(n),rhodn(n),tauup(n),taudn(n)
real(8), intent(out) :: dtdru(n),dtdrd(n)
! local variables
integer i
do i=1,n
  call k_tfsc1(2.d0*rhoup(i),2.d0*tauup(i),dtdru(i))
end do
do i=1,n
  call k_tfsc1(2.d0*rhodn(i),2.d0*taudn(i),dtdrd(i))
end do
end subroutine

