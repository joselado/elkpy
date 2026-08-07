
! Copyright (C) 2011 J. K. Dewhurst, S. Sharma and E. K. U. Gross
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine zhegvxi(n,m,ld1,a,b,w,ld2,z)
use modomp
implicit none
! arguments
integer, intent(in) :: n,m,ld1
complex(8), intent(in) :: a(ld1,*),b(ld1,*)
real(8), intent(out) :: w(m)
integer, intent(in) :: ld2
complex(8), intent(out) :: z(ld2,m)
! local variables
integer nb,lwork,nts
integer p,info,nthd
real(8) vl,vu
! external functions
integer, external :: ilaenv
! find the optimal blocksize for allocating the work array
nb=ilaenv(1,'ZHETRD','U',n,-1,-1,-1)
nb=max(nb,1)
lwork=(nb+1)*n
! enable MKL parallelism
call holdthd(maxthdmkl,nthd)
nts=mkl_set_num_threads_local(nthd)
block
integer iwork(5*n),ifail(n)
real(8) rwork(7*n),wn(n)
complex(8) work(lwork)
! find the first m eigenvalues and eigenvectors
call zhegvx(1,'V','I','U',n,a,ld1,b,ld1,vl,vu,1,m,-1.d0,p,wn,z,ld2,work,lwork, &
 rwork,iwork,ifail,info)
w(1:m)=wn(1:m)
end block
nts=mkl_set_num_threads_local(0)
call freethd(nthd)
if (info /= 0) then
  write(*,*)
  write(*,'("Error(zhegvxi): diagonalisation failed")')
  write(*,'(" ZHEGVX returned INFO = ",I0)') info
  write(*,*)
  stop
end if
end subroutine

