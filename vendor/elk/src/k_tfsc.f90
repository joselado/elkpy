
! Copyright (C) 2025 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine k_tfsc(n,rho,tau,dtdr)
implicit none
! arguments
integer, intent(in) :: n
real(8), intent(in) :: rho(n),tau(n)
real(8), intent(out) :: dtdr(n)
! local variables
integer i
do i=1,n
  call k_tfsc1(rho(i),tau(i),dtdr(i))
end do
end subroutine

