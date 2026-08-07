
! Copyright (C) 2021 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine genwfsv_sp(tsh,tgp,nst,idx,ngridg_,igfft_,ngp,igpig,apwalm,evecfv, &
 evecsv,wfmt,ld,wfir)
use modmain
use modomp
implicit none
! arguments
logical, intent(in) :: tsh,tgp
integer, intent(in) :: nst,idx(*),ngridg_(3),igfft_(*)
integer, intent(in) :: ngp(nspnfv),igpig(ngkmax,nspnfv)
complex(8), intent(in) :: apwalm(ngkmax,apwordmax,lmmaxapw,natmtot,nspnfv)
complex(8), intent(in) :: evecfv(nmatmax,nstfv,nspnfv),evecsv(nstsv,nstsv)
complex(4), intent(out) :: wfmt(npcmtmax,natmtot,nspinor,nst)
integer, intent(in) :: ld
complex(4), intent(out) :: wfir(ld,nspinor,nst)
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
  call wfmtsv_sp(tsh,is,ias,nst,idx,ngp,apwalm,evecfv,evecsv,ldmt, &
   wfmt(1,ias,1,1))
end do
!$OMP END DO NOWAIT
! interstitial wavefunction
!$OMP SINGLE
call wfirsv_sp(tgp,nst,idx,ngridg_,igfft_,ngp,igpig,evecfv,evecsv,ld,wfir)
!$OMP END SINGLE
!$OMP END PARALLEL
call freethd(nthd)
end subroutine

