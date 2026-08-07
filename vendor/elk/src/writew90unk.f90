
! Copyright (C) 2024 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine writew90unk
use modmain
use modw90
use modomp
implicit none
! local variables
integer ispn,ik,ist,nthd
integer is,ias,nrc,nrci,npc
integer np,ngp(nspnfv),iu,i
real(8) vc(3),kc
character(256) fname
! automatic arrays
complex(8) ylmk(lmmaxo),sfack(natmtot)
! allocatable arrays
integer, allocatable :: igpig(:,:)
real(8), allocatable :: vpl(:,:),jlkr(:,:)
complex(8), allocatable :: wfmt(:,:,:,:),wfir(:,:,:)
complex(8), allocatable :: expmt(:,:),wf(:,:,:)
! total number of plot points
np=np3d(1)*np3d(2)*np3d(3)
! generate the 3D plotting points
allocate(vpl(3,np))
call plotpt3d(vpl)
! parallel loop over non-reduced k-points
call holdthd(nkptnr,nthd)
!$OMP PARALLEL DEFAULT(SHARED) &
!$OMP PRIVATE(igpig,jlkr,wfmt,wfir) &
!$OMP PRIVATE(wf,expmt,ylmk,sfack) &
!$OMP PRIVATE(ngp,vc,kc,ist,ispn,ias,is) &
!$OMP PRIVATE(nrc,nrci,npc,fname,iu,i) &
!$OMP NUM_THREADS(nthd)
allocate(igpig(ngkmax,nspnfv))
allocate(jlkr(njcmax,nspecies))
allocate(wfmt(npcmtmax,natmtot,nspinor,num_bands))
allocate(wfir(ngtot,nspinor,num_bands))
allocate(wf(np,nspinor,num_bands))
allocate(expmt(npcmtmax,natmtot))
!$OMP DO SCHEDULE(DYNAMIC)
do ik=1,nkptnr
!$OMP CRITICAL(writew90unk_)
  write(*,'("Info(writew90unk): ",I0," of ",I0," k-points")') ik,nkptnr
!$OMP END CRITICAL(writew90unk_)
! generate the second-variational wavefunctions
  call genwfsvp(.false.,.false.,num_bands,idxw90,ngridg,igfft,vkl(:,ik),ngp, &
   igpig,wfmt,ngtot,wfir)
! generate the phase factor function exp(-ik.r) in the muffin-tins
  vc(:)=-vkc(:,ik)
  kc=sqrt(vc(1)**2+vc(2)**2+vc(3)**2)
  call genjlgpr(1,kc,jlkr)
  call genylmv(.true.,lmaxo,vc,ylmk)
  call gensfacgp(1,vc,1,sfack)
  call genexpmt(1,jlkr,ylmk,1,sfack,expmt)
! split the wavefunctions into real and imaginary parts
  do ist=1,num_bands
    do ispn=1,nspinor
      do ias=1,natmtot
        is=idxis(ias)
        nrc=nrcmt(is)
        nrci=nrcmti(is)
        npc=npcmt(is)
! remove the explicit phase exp(ik.r) from the muffin-tin wavefunction
        wfmt(1:npc,ias,ispn,ist)=wfmt(1:npc,ias,ispn,ist)*expmt(1:npc,ias)
        call zfshtip(nrc,nrci,wfmt(:,ias,ispn,ist))
      end do
! generate the wavefunctions on a regular grid
      call zfpts(np,vpl,wfmt(:,:,ispn,ist),wfir(:,ispn,ist),wf(:,ispn,ist))
    end do
  end do
  if (spinpol) then
    write(fname,'("UNK",I5.5,".NC")') ik
  else
    write(fname,'("UNK",I5.5,".1")') ik
  end if
  open(newunit=iu,file=trim(fname),form='UNFORMATTED',action='WRITE')
  write(iu) np3d(1),np3d(2),np3d(3),ik,num_bands
  do ist=1,num_bands
    write(iu) (wf(i,1,ist),i=1,np)
    if (spinpol) then
      write(iu) (wf(i,2,ist),i=1,np)
    end if
  end do
  close(iu)
end do
!$OMP END DO
deallocate(igpig,jlkr,wfmt,wfir,wf,expmt)
!$OMP END PARALLEL
call freethd(nthd)
write(*,*)
write(*,'("Info(writew90unk): created the UNKkkkkk.s files")')
deallocate(vpl)
end subroutine

