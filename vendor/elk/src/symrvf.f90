
! Copyright (C) 2002-2005 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: symrvf
! !INTERFACE:
subroutine symrvf(tspin,tnc,nrmt_,nrmti_,npmt_,ngridg_,ngtot_,ngvec_,nfgrz_, &
 igfft_,igrzf_,ld1,rvfmt,ld2,rvfir)
! !USES:
use modmain
use modomp
! !INPUT/OUTPUT PARAMETERS:
!   tspin   : .true. if spin rotations should be used (in,logical)
!   tnc     : .true. if the vector field is non-collinear, otherwise it is
!             collinear along the z-axis (in,logical)
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
!   ld1     : leading dimension (in,integer)
!   rvfmt   : real muffin-tin vector field (in,real(ld1,natmtot,*))
!   ld2     : leading dimension (in,integer)
!   rvfir   : real interstitial vector field (in,real(ld2,*))
! !DESCRIPTION:
!   Symmetrises a vector field defined over the entire unit cell using the full
!   set of crystal symmetries. If a particular symmetry involves rotating atom
!   1 into atom 2, then the spatial and spin rotations of that symmetry are
!   applied to the vector field in atom 2 (expressed in spherical harmonic
!   coefficients), which is then added to the field in atom 1. This is repeated
!   for all symmetry operations. The fully symmetrised field in atom 1 is then
!   rotated and copied to atom 2. Symmetrisation of the interstitial part of the
!   field is performed by {\tt symrvfir}. See also {\tt symrfmt} and
!   {\tt findsym}.
!
! !REVISION HISTORY:
!   Created May 2007 (JKD)
!   Fixed problem with improper rotations, February 2008 (L. Nordstrom,
!    F. Bultmark and F. Cricchio)
!EOP
!BOC
implicit none
! arguments
logical, intent(in) :: tspin,tnc
integer, intent(in) :: nrmt_(nspecies),nrmti_(nspecies),npmt_(nspecies)
integer, intent(in) :: ngridg_(3),ngtot_,ngvec_,nfgrz_
integer, intent(in) :: igfft_(ngvec_),igrzf_(nfgrz_)
integer, intent(in) :: ld1
real(8), intent(inout) :: rvfmt(ld1,natmtot,*)
integer, intent(in) :: ld2
real(8), intent(inout) :: rvfir(ld2,*)
! local variables
integer nthd
call holdthd(2,nthd)
!$OMP PARALLEL SECTIONS DEFAULT(SHARED) &
!$OMP NUM_THREADS(nthd)
!$OMP SECTION
call symrvfmt(tspin,tnc,nrmt_,nrmti_,npmt_,ld1,rvfmt)
!$OMP SECTION
call symrvfir(tspin,tnc,ngridg_,ngtot_,ngvec_,nfgrz_,igfft_,igrzf_,ld2,rvfir)
!$OMP END PARALLEL SECTIONS
call freethd(nthd)
end subroutine
!EOC

