
! Copyright (C) 2016 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine zheevdi(n,ld,a,w)
use modmain
use modomp
implicit none
! arguments
integer, intent(in) :: n,ld
complex(8), intent(inout) :: a(ld,n)
real(8), intent(out) :: w(n)
! local variables
integer lwork,lrwork,liwork
integer nts,info,nthd
! allocatable arrays
real(8), allocatable :: rwork(:)
complex(8), allocatable :: work(:)
lwork=2*n+n**2
lrwork=1+5*n+2*n**2
liwork=3+5*n
allocate(work(lwork),rwork(lrwork))
! enable MKL parallelism
call holdthd(maxthdmkl,nthd)
nts=mkl_set_num_threads_local(nthd)
block
integer iwork(liwork)
call zheevd('V','U',n,a,ld,w,work,lwork,rwork,lrwork,iwork,liwork,info)
end block
nts=mkl_set_num_threads_local(0)
call freethd(nthd)
if (info /= 0) then
  write(*,*)
  write(*,'("Error(zheevdi): diagonalisation failed")')
  write(*,'(" ZHEEVD returned INFO = ",I0)') info
  write(*,*)
  stop
end if
deallocate(rwork,work)
end subroutine

