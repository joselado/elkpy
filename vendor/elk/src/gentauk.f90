
! Copyright (C) 2011 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine gentauk(ik,taumt_,tauir_)
use modmain
use modomp
implicit none
! arguments
integer, intent(in) :: ik
real(8), intent(inout) :: taumt_(npcmtmax,natmtot,nspinor),tauir_(ngtc,nspinor)
! local variables
integer ispn,jspn,nst,ist,jst
integer is,ias,nrc,nrci
integer npc,igk,ifg,i,nthd
real(8) wo
! automatic arrays
integer idx(nstsv)
! automatic arrays
complex(8) gzfmt(npcmtmax,3),zfmt(npcmtmax),zfft(ngtc)
! allocatable arrays
complex(8), allocatable :: apwalm(:,:,:,:,:),evecfv(:,:),evecsv(:,:)
complex(8), allocatable :: wfmt(:,:,:,:),wfgp(:,:,:)
allocate(apwalm(ngkmax,apwordmax,lmmaxapw,natmtot,nspnfv))
allocate(evecfv(nmatmax,nstfv),evecsv(nstsv,nstsv))
! find the matching coefficients
do ispn=1,nspnfv
  call match(ngk(ispn,ik),vgkc(:,:,ispn,ik),gkc(:,ispn,ik), &
   sfacgk(:,:,ispn,ik),apwalm(:,:,:,:,ispn))
end do
! get the eigenvectors from file
call getevecfv(filext,ik,vkl(:,ik),vgkl(:,:,:,ik),evecfv)
call getevecsv(filext,ik,vkl(:,ik),evecsv)
! count and index the occupied states
nst=0
do ist=1,nstsv
  if (abs(occsv(ist,ik)) < epsocc) cycle
  nst=nst+1
  idx(nst)=ist
end do
! calculate the second-variational wavefunctions for occupied states
allocate(wfmt(npcmtmax,natmtot,nspinor,nst),wfgp(ngkmax,nspinor,nst))
call genwfsv(.true.,.true.,nst,idx,ngdgc,igfc,ngk(:,ik),igkig(:,:,ik),apwalm, &
 evecfv,evecsv,wfmt,ngkmax,wfgp)
deallocate(apwalm,evecfv,evecsv)
call holdthd(nspinor*natmtot+1,nthd)
!$OMP PARALLEL DEFAULT(SHARED) &
!$OMP PRIVATE(gzfmt,zfmt,zfft) &
!$OMP PRIVATE(ispn,jspn,ias,is) &
!$OMP PRIVATE(nrc,nrci,npc,ist,jst) &
!$OMP PRIVATE(wo,i,igk,ifg) &
!$OMP NUM_THREADS(nthd)
!-------------------------!
!     muffin-tin part     !
!-------------------------!
!$OMP DO COLLAPSE(2) SCHEDULE(DYNAMIC)
do ispn=1,nspinor
  do ias=1,natmtot
    is=idxis(ias)
    nrc=nrcmt(is)
    nrci=nrcmti(is)
    npc=npcmt(is)
    do ist=1,nst
      jst=idx(ist)
      wo=0.5d0*wkpt(ik)*occsv(jst,ik)
! compute the gradient of the wavefunction
      call gradzfmt(nrc,nrci,rlcmt(:,-1,is),wcrcmt(:,:,is), &
       wfmt(:,ias,ispn,ist),npcmtmax,gzfmt)
      do i=1,3
! convert gradient to spherical coordinates
        call zbsht(nrc,nrci,gzfmt(:,i),zfmt)
! add to total in muffin-tin
        taumt_(1:npc,ias,ispn)=taumt_(1:npc,ias,ispn) &
         +wo*(dble(zfmt(1:npc))**2+aimag(zfmt(1:npc))**2)
      end do
    end do
  end do
end do
!$OMP END DO NOWAIT
!---------------------------!
!     interstitial part     !
!---------------------------!
!$OMP SINGLE
do ist=1,nst
  jst=idx(ist)
  wo=0.5d0*wkpt(ik)*occsv(jst,ik)/omega
  do ispn=1,nspinor
    jspn=jspnfv(ispn)
    do i=1,3
      zfft(1:ngtc)=0.d0
      do igk=1,ngk(jspn,ik)
        ifg=igfc(igkig(igk,jspn,ik))
        zfft(ifg)=vgkc(i,igk,jspn,ik)*zi*wfgp(igk,ispn,ist)
      end do
      call zfftifc(3,ngdgc,1,zfft)
      tauir_(1:ngtc,ispn)=tauir_(1:ngtc,ispn) &
       +wo*(dble(zfft(1:ngtc))**2+aimag(zfft(1:ngtc))**2)
    end do
  end do
end do
!$OMP END SINGLE
!$OMP END PARALLEL
call freethd(nthd)
deallocate(wfmt,wfgp)
end subroutine

