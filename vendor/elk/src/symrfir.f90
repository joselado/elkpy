
! Copyright (C) 2007 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: symrfir
! !INTERFACE:
subroutine symrfir(ngridg_,ngtot_,ngvec_,nfgrz_,igfft_,igrzf_,rfir)
! !USES:
use modmain
! !INPUT/OUTPUT PARAMETERS:
!   ngridg_ : G-vector grid sizes (in,integer(3))
!   ngtot_  : total number of G-vectors (in,integer)
!   ngvec_  : number of G-vectors within cut-off (in,integer)
!   nfgrz_  : number of FFT elements for real-complex transforms (in,integer)
!   igfft_  : map from G-vector index to FFT array (in,integer(ngvec_))
!   igrzf_  : map from real-complex FFT index to G-point index
!             (in,integer(nfgrz_))
!   rfir    : real intersitial function (inout,real(ngtot_))
! !DESCRIPTION:
!   Symmetrises a real scalar interstitial function. The function is first
!   Fourier transformed to $G$-space, and then averaged over each symmetry by
!   rotating the Fourier coefficients and multiplying them by a phase factor
!   corresponding to the symmetry translation.
!
! !REVISION HISTORY:
!   Created July 2007 (JKD)
!EOP
!BOC
implicit none
! arguments
integer, intent(in) :: ngridg_(3),ngtot_,ngvec_,nfgrz_
integer, intent(in) :: igfft_(ngvec_),igrzf_(nfgrz_)
real(8), intent(inout) :: rfir(ngtot_)
! local variables
logical tv0
integer isym,lspl,sym(3,3)
integer ig,jg,ifg,jfg
integer i1,i2,i3,j1,j2,j3
real(8) v1,v2,v3,t1
! allocatable arrays
complex(8), allocatable :: zfft1(:),zfft2(:)
allocate(zfft1(ngtot_),zfft2(nfgrz_))
! Fourier transform function to G-space
zfft1(1:ngtot_)=rfir(1:ngtot_)
call zfftifc(3,ngridg_,-1,zfft1)
zfft2(1:nfgrz_)=0.d0
! loop over crystal symmetries
do isym=1,nsymcrys
! zero translation vector flag
  tv0=tv0symc(isym)
! translation vector in Cartesian coordinates
  if (.not.tv0) then
    v1=vtcsymc(1,isym)
    v2=vtcsymc(2,isym)
    v3=vtcsymc(3,isym)
  end if
! index to spatial rotation lattice symmetry
  lspl=lsplsymc(isym)
  sym(1:3,1:3)=symlat(1:3,1:3,lspl)
  do ifg=1,nfgrz_
    ig=igrzf_(ifg)
    if (ig > ngvec_) cycle
! multiply the transpose of the inverse symmetry matrix with the G-vector
    if (lspl == 1) then
      jg=ig
    else
      i1=ivg(1,ig); i2=ivg(2,ig); i3=ivg(3,ig)
      j1=sym(1,1)*i1+sym(2,1)*i2+sym(3,1)*i3
      j2=sym(1,2)*i1+sym(2,2)*i2+sym(3,2)*i3
      j3=sym(1,3)*i1+sym(2,3)*i2+sym(3,3)*i3
      jg=ivgig(j1,j2,j3)
    end if
    jfg=igfft_(jg)
! translation and spatial rotation
    if (tv0) then
! zero translation vector
      zfft2(ifg)=zfft2(ifg)+zfft1(jfg)
    else
! complex phase factor for translation
      t1=vgc(1,jg)*v1+vgc(2,jg)*v2+vgc(3,jg)*v3
      zfft2(ifg)=zfft2(ifg)+zfft1(jfg)*cmplx(cos(t1),-sin(t1),8)
    end if
  end do
end do
! Fourier transform to real-space and normalise
call rzfftifc(3,ngridg_,1,rfir,zfft2)
t1=1.d0/dble(nsymcrys)
rfir(1:ngtot_)=t1*rfir(1:ngtot_)
deallocate(zfft1,zfft2)
end subroutine
!EOC

