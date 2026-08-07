
! Copyright (C) 2002-2005 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: genwfsv
! !INTERFACE:
subroutine genwfsv(tsh,tgp,nst,idx,ngridg_,igfft_,ngp,igpig,apwalm,evecfv, &
 evecsv,wfmt,ld,wfir)
! !USES:
use modmain
use modomp
! !INPUT/OUTPUT PARAMETERS:
!   tsh     : .true. if wfmt should be in spherical harmonic basis, otherwise
!             in spherical coordinates (in,logical)
!   tgp     : .true. if wfir should be in G+p-space, otherwise in real-space
!             (in,logical)
!   nst     : number of states to be calculated (in,integer)
!   idx     : index to states which are to be calculated; if idx(1)=0 then
!             all states are calculated in the usual order (in,integer(*))
!   ngridg_ : G-vector grid sizes (in,integer(3))
!   igfft_  : map from G-vector index to FFT array (in,integer(*))
!   ngp     : number of G+p-vectors (in,integer(nspnfv))
!   igpig   : index from G+p-vectors to G-vectors (in,integer(ngkmax,nspnfv))
!   apwalm  : APW matching coefficients
!             (in,complex(ngkmax,apwordmax,lmmaxapw,natmtot,nspnfv))
!   evecfv  : first-variational eigenvectors (in,complex(nmatmax,nstfv,nspnfv))
!   evecsv  : second-variational eigenvectors (in,complex(nstsv,nstsv))
!   wfmt    : muffin-tin part of the wavefunctions for every state in spherical
!             coordinates (out,complex(npcmtmax,natmtot,nspinor,nst))
!   ld      : leading dimension of wfir (in,integer)
!   wfir    : interstitial part of the wavefunctions for every state
!             (out,complex(ld,nspinor,nst))
! !DESCRIPTION:
!   Calculates the second-variational spinor wavefunctions in both the
!   muffin-tin and interstitial regions for every state of a particular
!   $k$-point. A coarse radial mesh is assumed in the muffin-tins with angular
!   momentum cut-off of {\tt lmaxo}.
!
! !REVISION HISTORY:
!   Created November 2004 (Sharma)
!   Updated for spin-spirals, June 2010 (JKD)
!   Packed muffin-tins, April 2016 (JKD)
!EOP
!BOC
implicit none
! arguments
logical, intent(in) :: tsh,tgp
integer, intent(in) :: nst,idx(*),ngridg_(3),igfft_(*)
integer, intent(in) :: ngp(nspnfv),igpig(ngkmax,nspnfv)
complex(8), intent(in) :: apwalm(ngkmax,apwordmax,lmmaxapw,natmtot,nspnfv)
complex(8), intent(in) :: evecfv(nmatmax,nstfv,nspnfv),evecsv(nstsv,nstsv)
complex(8), intent(out) :: wfmt(npcmtmax,natmtot,nspinor,nst)
integer, intent(in) :: ld
complex(8), intent(out) :: wfir(ld,nspinor,nst)
! local variables
integer is,ias,ldmt,nthd
if (nst < 1) return
! muffin-tin wavefunction
ldmt=npcmtmax*natmtot
call holdthd(natmtot+1,nthd)
!$OMP PARALLEL DEFAULT(SHARED) &
!$OMP PRIVATE(ias,is) &
!$OMP NUM_THREADS(nthd)
!$OMP DO SCHEDULE(DYNAMIC)
do ias=1,natmtot
  is=idxis(ias)
  call wfmtsv(tsh,lradstp,is,ias,nst,idx,ngp,apwalm,evecfv,evecsv,ldmt, &
   wfmt(1,ias,1,1))
end do
!$OMP END DO NOWAIT
! interstitial wavefunction
!$OMP SINGLE
call wfirsv(tgp,nst,idx,ngridg_,igfft_,ngp,igpig,evecfv,evecsv,ld,wfir)
!$OMP END SINGLE
!$OMP END PARALLEL
call freethd(nthd)
end subroutine
!EOC

