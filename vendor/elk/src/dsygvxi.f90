
! Copyright (C) 2026 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine dsygvxi(n,m,ld1,a,b,w,ld2,z,lwork,work)
use modomp
implicit none
! arguments
integer, intent(in) :: n,m,ld1
real(8), intent(in) :: a(ld1,*),b(ld1,*)
real(8), intent(out) :: w(m)
integer, intent(in) :: ld2
real(8), intent(out) :: z(ld2,m)
integer, intent(in) :: lwork
real(8), intent(out) :: work(lwork)
! local variables
integer nts,p,info,nthd
real(8) vl,vu
! enable MKL parallelism
call holdthd(maxthdmkl,nthd)
nts=mkl_set_num_threads_local(nthd)
block
integer iwork(5*n),ifail(n)
real(8) wn(n)
! find the first m eigenvalues and eigenvectors
call dsygvx(1,'V','I','U',n,a,ld1,b,ld1,vl,vu,1,m,-1.d0,p,wn,z,ld2,work,lwork, &
 iwork,ifail,info)
w(1:m)=wn(1:m)
end block
nts=mkl_set_num_threads_local(0)
call freethd(nthd)
if (info /= 0) then
  write(*,*)
  write(*,'("Error(dsygvxi): diagonalisation failed")')
  write(*,'(" DSYGVX returned INFO = ",I0)') info
  write(*,*)
  stop
end if
end subroutine

