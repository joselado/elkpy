
! Copyright (C) 2017 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine dielectric_tdrt
use modmain
use modtddft
implicit none
! local variables
integer iw,i,j
real(8) w1,w2,t0,t1
complex(8) eta,z1,z2
character(256) fname
! allocatable arrays
real(8), allocatable :: w(:),wt(:),jt(:,:)
complex(8), allocatable :: ew(:,:),jw(:,:),eps(:)
! initialise global variables
call init0
call init1
! generate energy grid (always non-negative)
allocate(w(nwplot))
w1=max(wplot(1),0.d0)
w2=max(wplot(2),w1)
t1=(w2-w1)/dble(nwplot)
do iw=1,nwplot
  w(iw)=w1+t1*dble(iw-1)
end do
! i divided by the complex relaxation time
eta=cmplx(0.d0,swidth,8)
! determine the weights for the spline integration
allocate(wt(ntimes))
call wsplint(ntimes,times,wt)
! compute the electric field from E = -1/c dA/dt and Fourier transform
allocate(ew(nwplot,3))
t0=-1.d0/solsc
do i=1,3
  if (task == 480) then
! Fourier transform A(t) numerically to obtain A(ω)
    call zftft(w,wt,3,afieldt(i,1),ew(:,i))
! take the time derivative E(t) = -1/c dA(t)/dt analytically to get E(ω)
    do iw=1,nwplot
      z1=ew(iw,i)
      ew(iw,i)=t0*w(iw)*cmplx(z1%im,-z1%re,8)
    end do
! filter the high-frequency components from E(ω) with a Lorentzian convolution
    call zlrzncnv(nwplot,swidth,w,ew(:,i))
  else
! analytic Fourier transform of E(t) assumed to be a delta function at t = 0
    t1=t0*afieldt(i,1)
    ew(:,i)=t1
  end if
end do
! read in the total current from file
allocate(jt(3,ntimes))
call readjtot(jt)
! divide by the unit cell volume
jt(:,:)=jt(:,:)/omega
! set the constant part of J(t) to zero if required; this effectively removes
! the Drude term
if (jtconst0) then
  do i=1,3
    t1=dot_product(wt(:),jt(i,:))
    t1=t1/times(ntimes)
    jt(i,1:ntimes)=jt(i,1:ntimes)-t1
  end do
end if
! Fourier transform the current
allocate(jw(nwplot,3))
do i=1,3
  call zftft(w,wt,3,jt(i,1),jw(:,i))
! filter the high-frequency components from J(ω) with a Lorentzian convolution
  call zlrzncnv(nwplot,swidth,w,jw(:,i))
end do
deallocate(wt,jt)
! compute the dielectric function and write to file
allocate(eps(nwplot))
do i=1,3
  do j=1,3
    do iw=1,nwplot
      z1=jw(iw,i)
      z2=ew(iw,j)
      t1=abs(z2%re)+abs(z2%im)
      if (t1 > 1.d-8) then
        z1=z1/z2
      else
        z1=0.d0
      end if
      z1=fourpi*cmplx(-z1%im,z1%re,8)/(w(iw)+eta)
      if (i == j) z1=z1+1.d0
      eps(iw)=z1
    end do
! filter the high-frequency components of ϵ
    call zlrzncnv(nwplot,2.d0*swidth,w,eps)
    write(fname,'("EPSILON_TDRT_",2I1,".OUT")') i,j
    open(50,file=trim(fname),form='FORMATTED')
    do iw=1,nwplot
      write(50,'(2G18.10)') w(iw),dble(eps(iw))
    end do
    write(50,*)
    do iw=1,nwplot
      write(50,'(2G18.10)') w(iw),aimag(eps(iw))
    end do
    close(50)
  end do
end do
! write Fourier transform of total current to file
open(50,file='JTOTW.OUT',form='FORMATTED')
do i=1,3
  do iw=1,nwplot
    write(50,'(3G18.10)') w(iw),jw(iw,i)
  end do
  write(50,*)
end do
close(50)
write(*,*)
write(*,'("Info(dielectric_tdrt):")')
write(*,'(" dielectric tensor determined from real-time evolution")')
write(*,'(" written to EPSILON_TDRT_ij.OUT for components i,j = 1,2,3")')
write(*,*)
write(*,'(" (Note that only those components which are not orthogonal to the")')
write(*,'(" applied A-field will be calculated correctly)")')
write(*,*)
write(*,'(" Fourier transform of total current J(ω) written to JTOTW.OUT")')
deallocate(w,ew,jw,eps)
end subroutine

