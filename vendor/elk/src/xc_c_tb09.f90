
! Copyright (C) 2011 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine xc_c_tb09
use modmain
use libxcifc
implicit none
! local variables
integer is,ias,i
integer nr,nri,ir
real(8), parameter :: alpha=-0.012d0, beta=1.023d0
real(8) t1
! allocatable arrays
real(8), allocatable :: grfmt(:,:,:),grfir(:,:)
real(8), allocatable :: rfmt(:,:),rfir(:)
! external functions
real(8), external :: rfint
! compute the gradient of the density
allocate(grfmt(npmtmax,natmtot,3),grfir(ngtot,3))
call gradrf(rhomt,rhoir,grfmt,grfir)
allocate(rfmt(npmtmax,natmtot))
do ias=1,natmtot
  is=idxis(ias)
  nr=nrmt(is)
  nri=nrmti(is)
! convert muffin-tin density to spherical coordinates
  call rbsht(nr,nri,rhomt(:,ias),rfmt(:,ias))
! convert muffin-tin gradient to spherical coordinates
  do i=1,3
    call rbshtip(nr,nri,grfmt(:,ias,i))
  end do
! integrand in muffin-tin
  do i=1,npmt(is)
    t1=sqrt(grfmt(i,ias,1)**2+grfmt(i,ias,2)**2+grfmt(i,ias,3)**2)
    rfmt(i,ias)=t1/rfmt(i,ias)
  end do
! convert to spherical harmonics
  call rfshtip(nr,nri,rfmt(:,ias))
end do
deallocate(grfmt)
! integrand in interstitial
allocate(rfir(ngtot))
do ir=1,ngtot
  t1=sqrt(grfir(ir,1)**2+grfir(ir,2)**2+grfir(ir,3)**2)
  rfir(ir)=t1/rhoir(ir)
end do
! integrate over the unit cell
t1=rfint(rfmt,rfir)
! set the constant
c_tb09=alpha+beta*sqrt(abs(t1)/omega)
deallocate(grfir,rfmt,rfir)
end subroutine

