
! Copyright (C) 2025 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

elemental subroutine k_tfsc1(rho,tau,dtdr)
implicit none
! arguments
real(8), intent(in) :: rho,tau
real(8), intent(out) :: dtdr
! local variables
real(8), parameter :: pi=3.1415926535897932385d0
! Thomas-Fermi coefficient
real(8), parameter :: ctf=(3.d0/10.d0)*(3.d0*pi**2)**(2.d0/3.d0)
real(8) ttf
if (rho < 1.d-20) then
  dtdr=0.d0
  return
end if
! Thomas-Fermi τ
ttf=ctf*rho**(5.d0/3.d0)
! δτ(r')/δρ(r) scaled by the ratio of the exact τ to the TF τ
dtdr=(tau/ttf)*ctf*(5.d0/3.d0)*rho**(2.d0/3.d0)
end subroutine

