
! Copyright (C) 2007 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: symrf
! !INTERFACE:
subroutine symrf(nrmt_,nrmti_,npmt_,ngridg_,ngtot_,ngvec_,nfgrz_,igfft_,igrzf_,&
 ld,rfmt,rfir)
! !USES:
use modmain
use modomp
! !INPUT/OUTPUT PARAMETERS:
!   nrmt_   : number of radial points for each species (in,integer(nspecies))
!   nrmti_  : number of radial points on the inner part (in,integer(nspecies))
!   npmt_   : total number of points in each muffin-tin (in,integer(nspecies))
!   ngridg_ : G-vector grid sizes (in,integer(3))
!   ngtot_  : total number of G-vectors (in,integer)
!   ngvec_  : number of G-vectors within cut-off (in,integer)
!   nfgrz_  : number of FFT elements for real-complex transforms (in,integer)
!   igfft_  : map from G-vector index to FFT array (in,integer(ngvec_))
!   igrzf_  : map from real-complex FFT index to G-point index
!             (in,integer(nfgrz_)
!   ld      : leading dimension (in,integer)
!   rfmt    : real muffin-tin function (inout,real(ld,natmtot))
!   rfir    : real intersitial function (inout,real(ngtot_))
! !DESCRIPTION:
!   Symmetrises a real scalar function defined over the entire unit cell using
!   the full set of crystal symmetries. In the muffin-tin of a particular atom
!   the spherical harmonic coefficients of every equivlent atom are rotated and
!   averaged. The interstitial part of the function is first Fourier transformed
!   to $G$-space, and then averaged over each symmetry by rotating the Fourier
!   coefficients and multiplying them by a phase factor corresponding to the
!   symmetry translation.
!
! !REVISION HISTORY:
!   Created May 2007 (JKD)
!EOP
!BOC
implicit none
! arguments
integer, intent(in) :: nrmt_(nspecies),nrmti_(nspecies),npmt_(nspecies)
integer, intent(in) :: ngridg_(3),ngtot_,ngvec_,nfgrz_
integer, intent(in) :: igfft_(ngvec_),igrzf_(nfgrz_),ld
real(8), intent(inout) :: rfmt(ld,natmtot),rfir(ngtot_)
! local variables
integer nthd
call holdthd(2,nthd)
!$OMP PARALLEL SECTIONS DEFAULT(SHARED) &
!$OMP NUM_THREADS(nthd)
!$OMP SECTION
call symrfmt(nrmt_,nrmti_,npmt_,ld,rfmt)
!$OMP SECTION
call symrfir(ngridg_,ngtot_,ngvec_,nfgrz_,igfft_,igrzf_,rfir)
!$OMP END PARALLEL SECTIONS
call freethd(nthd)
end subroutine
!EOC

