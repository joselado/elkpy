
! Copyright (C) 2022 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine rzmctmu(l,n,a,b,ld,c)
use modomp
implicit none
! arguments
integer, intent(in) :: l,n
complex(8), intent(in) :: a(l,n),b(l,n)
integer, intent(in) :: ld
complex(8), intent(inout) :: c(ld,*)
! local variables
integer j,nthd
call holdthd(n,nthd)
!$OMP PARALLEL DO DEFAULT(SHARED) &
!$OMP SCHEDULE(DYNAMIC) &
!$OMP NUM_THREADS(nthd)
do j=1,n
  call dgemv('T',2*l,j,1.d0,a,2*l,b(1,j),1,1.d0,c(1,j),2)
end do
!$OMP END PARALLEL DO
call freethd(nthd)
end subroutine

