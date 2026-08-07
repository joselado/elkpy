
! Copyright (C) 2025 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine zftft(w,wt,ld,ft,fw)
use modmain
use modtddft
use modomp
implicit none
! arguments
real(8), intent(in) :: w(nwplot),wt(ntimes)
integer, intent(in) :: ld
real(8), intent(in) :: ft(ld,ntimes)
complex(8), intent(out) :: fw(nwplot)
! local variables
integer iw,its,nthd
real(8) t1,t2
! automatic arrays
real(8) f1(ntimes),f2(ntimes)
call holdthd(nwplot,nthd)
!$OMP PARALLEL DO DEFAULT(SHARED) &
!$OMP PRIVATE(f1,f2,its,t1,t2) &
!$OMP SCHEDULE(DYNAMIC) &
!$OMP NUM_THREADS(nthd)
do iw=1,nwplot
  do its=1,ntimes
    t1=ft(1,its)
    t2=w(iw)*times(its)
    f1(its)=t1*cos(t2)
    f2(its)=t1*sin(t2)
  end do
  t1=dot_product(wt(:),f1(:))
  t2=dot_product(wt(:),f2(:))
  fw(iw)=cmplx(t1,t2,8)
end do
!$OMP END PARALLEL DO
call freethd(nthd)
end subroutine

