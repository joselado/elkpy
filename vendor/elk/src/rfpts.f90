
! Copyright (C) 2002-2005 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine rfpts(np,vpl,rfmt,rfir,fp)
use modmain
use modomp
implicit none
! arguments
integer, intent(in) :: np
real(8), intent(in) :: vpl(3,np),rfmt(npmtmax,natmtot),rfir(ngtot)
real(8), intent(out) :: fp(np)
! local variables
integer ip,nthd
! allocatable arrays
complex(8), allocatable :: zfft(:)
! Fourier transform rfir to G-space
allocate(zfft(ngtot))
zfft(1:ngtot)=rfir(1:ngtot)
call zfftifc(3,ngridg,-1,zfft)
! begin loop over all points
call holdthd(np,nthd)
!$OMP PARALLEL DO DEFAULT(SHARED) &
!$OMP SCHEDULE(DYNAMIC) &
!$OMP NUM_THREADS(nthd)
do ip=1,np
  call rfpt(ip)
end do
!$OMP END PARALLEL DO
call freethd(nthd)
deallocate(zfft)

contains

subroutine rfpt(ip)
implicit none
! arguments
integer, intent(in) :: ip
! local variables
integer is,ias,nr,nri
integer ir,ir0,i0,ig,j
integer lmax,lmmax,lm
real(8) v(3),r,sm,t1
! automatic arrays
real(8) ya(4,lmmaxo),rlm(lmmaxo)
! check if point is in a muffin-tin
call findmtpt(vpl(:,ip),ias,ir,v,r)
if (ir > 0) then
! point is inside a muffin-tin
  is=idxis(ias)
  nr=nrmt(is)
  nri=nrmti(is)
  if (ir <= 3) then
    ir0=1
  else if (ir > nr-2) then
    ir0=nr-3
  else
    ir0=ir-2
  end if
  r=max(r,rsp(1,is))
  if (ir0 <= nri) then
    lmax=lmaxi
    lmmax=lmmaxi
  else
    lmax=lmaxo
    lmmax=lmmaxo
  end if
  do j=1,4
    ir=ir0+j-1
    if (ir <= nri) then
      i0=lmmaxi*(ir-1)
    else
      i0=lmmaxi*nri+lmmaxo*(ir-nri-1)
    end if
    ya(j,1:lmmax)=rfmt(i0+1:i0+lmmax,ias)
  end do
! generate real spherical harmonics
  call genrlmv(lmax,v,rlm)
  sm=0.d0
  do lm=1,lmmax
    t1=poly4(rsp(ir0,is),ya(:,lm),r)
    sm=sm+t1*rlm(lm)
  end do
else
! otherwise use direct Fourier transform of the interstitial function
  v(1:3)=vpl(1,ip)*avec(1:3,1)+vpl(2,ip)*avec(1:3,2)+vpl(3,ip)*avec(1:3,3)
  sm=0.d0
!$OMP SIMD PRIVATE(t1) REDUCTION(+:sm)
  do ig=1,ngvec
    t1=vgc(1,ig)*v(1)+vgc(2,ig)*v(2)+vgc(3,ig)*v(3)
    sm=sm+dble(zfft(igfft(ig))*cmplx(cos(t1),sin(t1),8))
  end do
end if
fp(ip)=sm
end subroutine

pure real(8) function poly4(xa,ya,x)
implicit none
! arguments
real(8), intent(in) :: xa(4),ya(4),x
! local variables
real(8) x0,x1,x2,x3,y0,y1,y2,y3
real(8) c1,c2,c3,t0,t1,t2,t3,t4,t5,t6
! evaluate the polynomial coefficients
x0=xa(1)
x1=xa(2)-x0; x2=xa(3)-x0; x3=xa(4)-x0
y0=ya(1)
y1=ya(2)-y0; y2=ya(3)-y0; y3=ya(4)-y0
t4=x1-x2; t5=x1-x3; t6=x2-x3
t1=x1*x2*y3; t2=x2*x3*y1; t3=x1*x3
t0=1.d0/(x2*t3*t4*t5*t6)
t3=t3*y2
c3=t1*t4+t2*t6-t3*t5
t4=x1**2; t5=x2**2; t6=x3**2
c2=t1*(t5-t4)+t2*(t6-t5)+t3*(t4-t6)
c1=t1*(x2*t4-x1*t5)+t2*(x3*t5-x2*t6)+t3*(x1*t6-x3*t4)
t1=x-x0
! evaluate the polynomial
poly4=y0+t0*t1*(c1+t1*(c2+c3*t1))
end function

end subroutine

