
! Copyright (C) 2007 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: symrvfir
! !INTERFACE:
subroutine symrvfir(tspin,tnc,ngridg_,ngtot_,ngvec_,nfgrz_,igfft_,igrzf_,ld, &
 rvfir)
! !USES:
use modmain
! !INPUT/OUTPUT PARAMETERS:
!   tspin   : .true. if spin rotations should be used (in,logical)
!   tnc     : .true. if the vector field is non-collinear, otherwise it is
!             collinear along the z-axis (in,logical)
!   ngridg_ : G-vector grid sizes (in,integer(3))
!   ngtot_  : total number of G-vectors (in,integer)
!   ngvec_  : number of G-vectors within cut-off (in,integer)
!   nfgrz_  : number of FFT elements for real-complex transforms (in,integer)
!   igfft_  : map from G-vector index to FFT array (in,integer(ngvec_))
!   igrzf_  : map from real-complex FFT index to G-point index
!             (in,integer(nfgrz_))
!   ld      : leading dimension (in,integer)
!   rvfir   : real interstitial vector function (inout,real(ld,*))
! !DESCRIPTION:
!   Symmetrises a real interstitial vector function. See routines {\tt symrvf}
!   and {\tt symrfir} for details.
!
! !REVISION HISTORY:
!   Created July 2007 (JKD)
!EOP
!BOC
implicit none
! arguments
logical, intent(in) :: tspin,tnc
integer, intent(in) :: ngridg_(3),ngtot_,ngvec_,nfgrz_
integer, intent(in) :: igfft_(ngvec_),igrzf_(nfgrz_),ld
real(8), intent(inout) :: rvfir(ld,*)
! local variables
logical tv0
integer isym,lspl,lspn,sym(3,3)
integer nd,ig,jg,ifg,jfg
integer i1,i2,i3,j1,j2,j3,i
real(8) sc(3,3),v1,v2,v3,t1
complex(8) z0,z1,z2,z3
! allocatable arrays
complex(8), allocatable :: zfft1(:,:),zfft2(:,:)
! dimension of the vector field
if (tnc) then
  nd=3
else
  nd=1
end if
allocate(zfft1(ngtot_,nd),zfft2(nfgrz_,nd))
! Fourier transform vector function to G-space
do i=1,nd
  zfft1(1:ngtot_,i)=rvfir(1:ngtot_,i)
  call zfftifc(3,ngridg_,-1,zfft1(:,i))
end do
zfft2(1:nfgrz_,1:nd)=0.d0
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
  if (tspin) then
! global spin proper rotation in Cartesian coordinates
    lspn=lspnsymc(isym)
    sc(1:3,1:3)=symlatd(lspn)*symlatc(1:3,1:3,lspn)
  else
! set spin rotation equal to spatial rotation
    lspn=lspl
    sc(1:3,1:3)=symlatc(1:3,1:3,lspl)
  end if
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
! translation, spatial rotation and global spin rotation
    if (tv0) then
! zero translation vector
      if (lspn == 1) then
! global spin symmetry is the identity
        zfft2(ifg,1:nd)=zfft2(ifg,1:nd)+zfft1(jfg,1:nd)
      else
        if (tnc) then
! non-collinear case
          z1=zfft1(jfg,1); z2=zfft1(jfg,2); z3=zfft1(jfg,3)
          zfft2(ifg,1)=zfft2(ifg,1)+sc(1,1)*z1+sc(1,2)*z2+sc(1,3)*z3
          zfft2(ifg,2)=zfft2(ifg,2)+sc(2,1)*z1+sc(2,2)*z2+sc(2,3)*z3
          zfft2(ifg,3)=zfft2(ifg,3)+sc(3,1)*z1+sc(3,2)*z2+sc(3,3)*z3
        else
! collinear case
          zfft2(ifg,1)=zfft2(ifg,1)+sc(3,3)*zfft1(jfg,1)
        end if
      end if
    else
! complex phase factor for translation
      t1=vgc(1,jg)*v1+vgc(2,jg)*v2+vgc(3,jg)*v3
      z0=cmplx(cos(t1),-sin(t1),8)
      if (lspn == 1) then
        zfft2(ifg,1:nd)=zfft2(ifg,1:nd)+z0*zfft1(jfg,1:nd)
      else
        if (tnc) then
          z1=zfft1(jfg,1); z2=zfft1(jfg,2); z3=zfft1(jfg,3)
          zfft2(ifg,1)=zfft2(ifg,1)+z0*(sc(1,1)*z1+sc(1,2)*z2+sc(1,3)*z3)
          zfft2(ifg,2)=zfft2(ifg,2)+z0*(sc(2,1)*z1+sc(2,2)*z2+sc(2,3)*z3)
          zfft2(ifg,3)=zfft2(ifg,3)+z0*(sc(3,1)*z1+sc(3,2)*z2+sc(3,3)*z3)
        else
          zfft2(ifg,1)=zfft2(ifg,1)+sc(3,3)*z0*zfft1(jfg,1)
        end if
      end if
    end if
  end do
end do
! Fourier transform to real-space and normalise
t1=1.d0/dble(nsymcrys)
do i=1,nd
  call rzfftifc(3,ngridg_,1,rvfir(:,i),zfft2(:,i))
  rvfir(1:ngtot_,i)=t1*rvfir(1:ngtot_,i)
end do
deallocate(zfft1,zfft2)
end subroutine
!EOC

