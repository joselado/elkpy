
! Copyright (C) 2002-2005 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: potcoul
! !INTERFACE:
subroutine potcoul
! !USES:
use modmain
use modomp
! !DESCRIPTION:
!   Calculates the Coulomb potential of the real charge density stored in the
!   global variables {\tt rhomt} and {\tt rhoir} by solving Poisson's equation.
!   These variables are coverted to complex representations and passed to the
!   routine {\tt zpotcoul}.
!
! !REVISION HISTORY:
!   Created April 2003 (JKD)
!EOP
!BOC
implicit none
! local variables
integer is,ias,nthd
integer nr,nri,iro,i0,i1
! allocatable arrays
complex(8), allocatable :: zvclmt(:,:),zvclir(:)
allocate(zvclmt(npmtmax,natmtot))
! convert real muffin-tin charge density to complex spherical harmonic expansion
call holdthd(natmtot,nthd)
!$OMP PARALLEL DO DEFAULT(SHARED) &
!$OMP PRIVATE(is) &
!$OMP SCHEDULE(DYNAMIC) &
!$OMP NUM_THREADS(nthd)
do ias=1,natmtot
  is=idxis(ias)
  call rtozfmt(nrmt(is),nrmti(is),rhomt(:,ias),zvclmt(:,ias))
end do
!$OMP END PARALLEL DO
call freethd(nthd)
! solve the complex Poisson's equation in the muffin-tins
call genzvclmt(nrmt,nrmti,nrmtmax,rlmt,wprmt,npmtmax,zvclmt)
! add the nuclear monopole potentials
do ias=1,natmtot
  is=idxis(ias)
  nr=nrmt(is)
  nri=nrmti(is)
  iro=nri+1
  i1=lmmaxi*(nri-1)+1
  zvclmt(1:i1:lmmaxi,ias)=zvclmt(1:i1:lmmaxi,ias)+vcln(1:nri,is)
  i0=i1+lmmaxi
  i1=lmmaxo*(nr-iro)+i0
  zvclmt(i0:i1:lmmaxo,ias)=zvclmt(i0:i1:lmmaxo,ias)+vcln(iro:nr,is)
end do
! apply atomic displacement potential if required
if (tatdisp) call zvcldisp(zvclmt)
allocate(zvclir(ngtot))
! store real interstitial charge density in complex array
zvclir(1:ngtot)=rhoir(1:ngtot)
! solve Poisson's equation in the entire unit cell
call zpotcoul(0,nrmt,nrmti,npmt,nrmtmax,rlmt,ngridg,igfft,ngvec,gc,gclg,ngvec, &
 jlgrmt,ylmg,sfacg,npmtmax,zvclmt,zvclir)
! convert complex muffin-tin potential to real spherical harmonic expansion
call holdthd(natmtot+1,nthd)
!$OMP PARALLEL DEFAULT(SHARED) &
!$OMP PRIVATE(ias,is) &
!$OMP NUM_THREADS(nthd)
!$OMP DO SCHEDULE(DYNAMIC)
do ias=1,natmtot
  is=idxis(ias)
  call ztorfmt(nrmt(is),nrmti(is),zvclmt(:,ias),vclmt(:,ias))
end do
!$OMP END DO NOWAIT
! store complex interstitial potential in real array
!$OMP SINGLE
vclir(1:ngtot)=dble(zvclir(1:ngtot))
!$OMP END SINGLE
!$OMP END PARALLEL
call freethd(nthd)
deallocate(zvclmt,zvclir)
! apply constant electric field if required
if (tefield) call potefield
end subroutine
!EOC

