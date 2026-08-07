
! Copyright (C) 2010 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine potefield
use modmain
implicit none
! local variables
integer is,ia,ias
integer nr,nri,ir
integer i,i1,i2,i3
real(8) ex,ey,ez,e,v0,r
real(8) f01,f11,f12,f13
real(8) v1(3),v2(3),t1,t2
! external E-field in Cartesian coordinates
ex=efieldc(1); ey=efieldc(2); ez=efieldc(3)
e=sqrt(ex**2+ey**2+ez**2)
if (e > 1.d-8) then
  ex=ex/e; ey=ey/e; ez=ez/e
else
  return
end if
! constant added to potential so that it is zero at the unit cell center
v1(:)=0.5d0*(avec(:,1)+avec(:,2)+avec(:,3))
v0=dot_product(efieldc(:),v1(:))
! coefficients for real spherical harmonics R₁₋₁, R₁₀ and R₁₁
t1=e*sqrt(fourpi/3.d0)
f11=t1*ey
f12=-t1*ez
f13=t1*ex
! muffin-tin potential
do ias=1,natmtot
  is=idxis(ias)
  ia=idxia(ias)
  nr=nrmt(is)
  nri=nrmti(is)
! coefficient for R_00
  f01=v0-dot_product(efieldc(:),atposc(:,ia,is))
  f01=f01*y00i
  i=1
  do ir=1,nr
    r=rlmt(ir,1,is)
    vclmt(i,ias)=vclmt(i,ias)+f01
    vclmt(i+1,ias)=vclmt(i+1,ias)+f11*r
    vclmt(i+2,ias)=vclmt(i+2,ias)+f12*r
    vclmt(i+3,ias)=vclmt(i+3,ias)+f13*r
    if (ir <= nri) then
      i=i+lmmaxi
    else
      i=i+lmmaxo
    end if
  end do
end do
! interstitial potential
t2=0.5d0*vmaxefc
ir=0
do i3=0,ngridg(3)-1
  v1(3)=dble(i3)/dble(ngridg(3))
  do i2=0,ngridg(2)-1
    v1(2)=dble(i2)/dble(ngridg(2))
    do i1=0,ngridg(1)-1
      v1(1)=dble(i1)/dble(ngridg(1))
      ir=ir+1
      call r3mv(avec,v1,v2)
      t1=efieldc(1)*v2(1)+efieldc(2)*v2(2)+efieldc(3)*v2(3)
      t1=v0-t1
      if (abs(t1) < vmaxefc) then
        vclir(ir)=vclir(ir)+t1
      else
        vclir(ir)=vclir(ir)+sign(t2,t1)
      end if
    end do
  end do
end do
end subroutine

