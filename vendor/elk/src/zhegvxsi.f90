
! Copyright (C) 2026 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine zhegvxsi(n,m,ms,ld1,a,b,w,ld2,z)
use modomp
use, intrinsic :: iso_c_binding
implicit none
! arguments
integer, intent(in) :: n,m,ms,ld1
complex(8), target :: a(ld1,*),b(ld1,*)
real(8), intent(out) :: w(m)
integer, intent(in) :: ld2
complex(8), intent(out) :: z(ld2,m)
! local variables
integer ns,i,j,k,l,nthd
! automatic arrays
integer idx(n)
real(8) d(n)
! allocatable arrays and pointers
complex(8), allocatable :: as(:,:)
complex(8), pointer, contiguous :: bs(:,:),zs(:,:)
! determine the size of the subspace
ns=merge(n/abs(ms),ms,ms < 0)
if (ns < m) ns=m
if (ns > n) ns=n
! choose the subspace based on the lowest ns diagonal elements of A
do i=1,n
  d(i)=a(i,i)%re
end do
call sortidx(n,d,idx)
! reuse already allocated memory
call c_f_pointer(c_loc(a),bs,shape=[ns,ns])
call c_f_pointer(c_loc(b),zs,shape=[ns,m])
allocate(as(ns,ns))
call holdthd(ns,nthd)
!$OMP PARALLEL DEFAULT(SHARED) &
!$OMP PRIVATE(i,j,k,l) &
!$OMP NUM_THREADS(nthd)
! reconstruct A and B in the subspace
!$OMP DO SCHEDULE(DYNAMIC)
do j=1,ns
  l=idx(j)
  do i=1,j
    k=idx(i)
    as(i,j)=merge(a(k,l),conjg(a(l,k)),k <= l)
  end do
end do
!$OMP END DO NOWAIT
!$OMP DO SCHEDULE(DYNAMIC)
do j=1,ns
  l=idx(j)
  do i=1,j
    k=idx(i)
    bs(i,j)=merge(b(k,l),conjg(b(l,k)),k <= l)
  end do
end do
!$OMP END DO
!$OMP END PARALLEL
call freethd(nthd)
! solve the generalised eigenvalue problem in the subspace
call zhegvxi(ns,m,ns,as,bs,w,ns,zs)
call holdthd(m,nthd)
!$OMP PARALLEL DO DEFAULT(SHARED) &
!$OMP NUM_THREADS(nthd)
do i=1,m
  z(idx(1:ns),i)=zs(1:ns,i)
  z(idx(ns+1:n),i)=0.d0
end do
!$OMP END PARALLEL DO
call freethd(nthd)
deallocate(as)
end subroutine

