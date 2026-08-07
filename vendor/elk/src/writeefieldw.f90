
! Copyright (C) 2024 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine writeefieldw
use modmain
use modtddft
implicit none
! local variables
integer iw,i
real(8) w1,w2,t0,t1
complex(8) z1
! allocatable arrays
real(8), allocatable :: w(:),wt(:)
complex(8), allocatable :: ew(:,:)
! read time-dependent A-field from file
call readafieldt
! generate energy grid (always non-negative)
allocate(w(nwplot))
w1=max(wplot(1),0.d0)
w2=max(wplot(2),w1)
t1=(w2-w1)/dble(nwplot)
do iw=1,nwplot
  w(iw)=w1+t1*dble(iw-1)
end do
! determine the weights for the spline integration
allocate(wt(ntimes))
call wsplint(ntimes,times,wt)
! compute the electric field from E = -1/c dA/dt and Fourier transform
allocate(ew(nwplot,3))
t0=-1.d0/solsc
do i=1,3
! Fourier transform A(t) numerically to obtain A(ω)
  call zftft(w,wt,3,afieldt(i,1),ew(:,i))
! take the time derivative E(t) = -1/c dA(t)/dt analytically to get E(ω)
  do iw=1,nwplot
    z1=ew(iw,i)
    ew(iw,i)=t0*w(iw)*cmplx(z1%im,-z1%re,8)
  end do
! filter the high-frequency components from E(ω) with a Lorentzian convolution
  call zlrzncnv(nwplot,swidth,w,ew(:,i))
end do
! write Fourier transform of electric field to file
open(50,file='EFIELDW.OUT',form='FORMATTED')
do i=1,3
  do iw=1,nwplot
    write(50,'(3G18.10)') w(iw),ew(iw,i)
  end do
  write(50,*)
end do
close(50)
write(*,*)
write(*,'("Info(writeefieldw):")')
write(*,'(" Fourier transform of electric field E(ω) written to EFIELDW.OUT")')
deallocate(w,wt,ew)
end subroutine

