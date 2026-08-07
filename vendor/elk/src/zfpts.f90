
! Copyright (C) 2018 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine zfpts(np,vrl,zfmt,zfir,fp)
use modmain
use modomp
implicit none
! arguments
integer, intent(in) :: np
real(8), intent(in) :: vrl(3,np)
complex(8), intent(in) :: zfmt(npcmtmax,natmtot),zfir(ngtot)
complex(8), intent(out) :: fp(np)
! local variables
integer ip,nthd
! allocatable arrays
complex(8), allocatable :: zfft(:)
! Fourier transform zfir to G-space
allocate(zfft(ngtot))
zfft(1:ngtot)=zfir(1:ngtot)
call zfftifc(3,ngridg,-1,zfft)
! begin loop over all points
call holdthd(np,nthd)
!$OMP PARALLEL DO DEFAULT(SHARED) &
!$OMP SCHEDULE(DYNAMIC) &
!$OMP NUM_THREADS(nthd)
do ip=1,np
  call zfip(ip)
end do
!$OMP END PARALLEL DO
call freethd(nthd)
deallocate(zfft)

contains

subroutine zfip(ip)
implicit none
! arguments
integer, intent(in) :: ip
! local variables
integer is,ias,nrc,nrci
integer ir,irc,irc0,i0,j
integer lmax,lmmax,lm,ig
real(8) v(3),r,t1
complex(8) zsm,z1
! automatic arrays
complex(8) ya(4,lmmaxo),ylm(lmmaxo)
! check if point is in a muffin-tin
call findmtpt(vrl(:,ip),ias,ir,v,r)
if (ir > 0) then
! point is inside a muffin-tin
  is=idxis(ias)
  nrc=nrcmt(is)
  nrci=nrcmti(is)
  irc=(ir-1)/lradstp+1
  if (irc <= 3) then
    irc0=1
  else if (irc > nrc-2) then
    irc0=nrc-3
  else
    irc0=irc-2
  end if
  r=max(r,rcmt(1,is))
  if (irc0 <= nrci) then
    lmax=lmaxi
    lmmax=lmmaxi
  else
    lmax=lmaxo
    lmmax=lmmaxo
  end if
  do j=1,4
    irc=irc0+j-1
    if (irc <= nrci) then
      i0=lmmaxi*(irc-1)
    else
      i0=lmmaxi*nrci+lmmaxo*(irc-nrci-1)
    end if
    ya(j,1:lmmax)=zfmt(i0+1:i0+lmmax,ias)
  end do
! generate complex spherical harmonics
  call genylmv(.false.,lmaxo,v,ylm)
  zsm=0.d0
  do lm=1,lmmax
    z1=poly4(rcmt(irc0,is),ya(:,lm),r)
    zsm=zsm+z1*ylm(lm)
  end do
else
! otherwise use direct Fourier transform of interstitial function
  v(1:3)=vrl(1,ip)*avec(1:3,1)+vrl(2,ip)*avec(1:3,2)+vrl(3,ip)*avec(1:3,3)
  zsm=0.d0
!$OMP SIMD PRIVATE(t1) REDUCTION(+:zsm)
  do ig=1,ngvec
    t1=vgc(1,ig)*v(1)+vgc(2,ig)*v(2)+vgc(3,ig)*v(3)
    zsm=zsm+zfft(igfft(ig))*cmplx(cos(t1),sin(t1),8)
  end do
end if
fp(ip)=zsm
end subroutine

pure complex(8) function poly4(xa,ya,x)
implicit none
! arguments
real(8), intent(in) :: xa(4),x
complex(8), intent(in) :: ya(4)
! local variables
real(8) x0,x1,x2,x3,t0,t1,t2,t3,t4
complex(8) y0,y1,y2,y3,z1,z2,z3,c1,c2,c3
! evaluate the polynomial coefficients
x0=xa(1)
x1=xa(2)-x0; x2=xa(3)-x0; x3=xa(4)-x0
t1=x1-x2; t2=x1-x3; t3=x2-x3
y0=ya(1)
y1=ya(2)-y0; y2=ya(3)-y0; y3=ya(4)-y0
z1=x1*x2*y3; z2=x2*x3*y1; t4=x1*x3
t0=1.d0/(x2*t1*t2*t3*t4)
z3=t4*y2
c3=z1*t1+z2*t3-z3*t2
t1=x1**2; t2=x2**2; t3=x3**2
c2=z1*(t2-t1)+z2*(t3-t2)+z3*(t1-t3)
c1=z1*(x2*t1-x1*t2)+z2*(x3*t2-x2*t3)+z3*(x1*t3-x3*t1)
t1=x-x0
! evaluate the polynomial
poly4=y0+t0*t1*(c1+t1*(c2+c3*t1))
end function

end subroutine

