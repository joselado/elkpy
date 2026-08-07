
! Copyright (C) 2018 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine genjprk(ik,jrmt_,jrir_)
use modmain
implicit none
! arguments
integer, intent(in) :: ik
real(8), intent(inout) :: jrmt_(npcmtmax,natmtot,3),jrir_(ngtc,3)
! local variables
integer ispn,jspn,nst,ist,jst
integer is,ia,ias,nrc,nrci,npc
integer igk,ifg,i
real(8) wo
! automatic arrays
integer idx(nstsv)
real(8) rfmt(npcmtmax)
complex(8) gwfmt(npcmtmax,3),zfmt1(npcmtmax),zfmt2(npcmtmax)
! allocatable arrays
complex(8), allocatable :: apwalm(:,:,:,:,:),evecfv(:,:,:),evecsv(:,:)
complex(8), allocatable :: wfmt(:,:,:,:),wfgk(:,:,:),zfft1(:),zfft2(:)
allocate(apwalm(ngkmax,apwordmax,lmmaxapw,natmtot,nspnfv))
allocate(evecfv(nmatmax,nstfv,nspnfv),evecsv(nstsv,nstsv))
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
allocate(wfmt(npcmtmax,natmtot,nspinor,nst),wfgk(ngkmax,nspinor,nst))
call genwfsv(.true.,.true.,nst,idx,ngdgc,igfc,ngk(:,ik),igkig(:,:,ik),apwalm, &
 evecfv,evecsv,wfmt,ngkmax,wfgk)
deallocate(apwalm,evecfv,evecsv)
!-------------------------------------------------!
!     muffin-tin paramagnetic current density     !
!-------------------------------------------------!
do ist=1,nst
  jst=idx(ist)
  wo=wkpt(ik)*occsv(jst,ik)
  do ispn=1,nspinor
    do is=1,nspecies
      nrc=nrcmt(is)
      nrci=nrcmti(is)
      npc=npcmt(is)
      do ia=1,natoms(is)
        ias=idxas(ia,is)
! compute the gradient of the wavefunction
        call gradzfmt(nrc,nrci,rlcmt(:,-1,is),wcrcmt(:,:,is), &
         wfmt(:,ias,ispn,ist),npcmtmax,gwfmt)
! convert wavefunction to spherical coordinates and conjugate
        call zbsht(nrc,nrci,wfmt(:,ias,ispn,ist),zfmt1)
        zfmt1(1:npc)=conjg(zfmt1(1:npc))
        do i=1,3
! convert wavefunction gradient to spherical coordinates
          call zbsht(nrc,nrci,gwfmt(:,i),zfmt2)
! compute the partial current density
          rfmt(1:npc)=aimag(zfmt1(1:npc)*zfmt2(1:npc))
          jrmt_(1:npc,ias,i)=jrmt_(1:npc,ias,i)+wo*rfmt(1:npc)
        end do
      end do
    end do
  end do
end do
deallocate(wfmt)
!---------------------------------------------------!
!     interstitial paramagnetic current density     !
!---------------------------------------------------!
allocate(zfft1(ngtc),zfft2(ngtc))
do ist=1,nst
  jst=idx(ist)
  wo=wkpt(ik)*occsv(jst,ik)/omega
  do ispn=1,nspinor
    jspn=jspnfv(ispn)
! Fourier transform to real-space and conjugate
    zfft1(1:ngtc)=0.d0
    do igk=1,ngk(jspn,ik)
      ifg=igfc(igkig(igk,jspn,ik))
      zfft1(ifg)=wfgk(igk,ispn,ist)
    end do
    call zfftifc(3,ngdgc,1,zfft1)
    zfft1(1:ngtc)=conjg(zfft1(1:ngtc))
    do i=1,3
! compute the gradient of the wavefunction
      zfft2(1:ngtc)=0.d0
      do igk=1,ngk(jspn,ik)
        ifg=igfc(igkig(igk,jspn,ik))
        zfft2(ifg)=vgkc(i,igk,jspn,ik)*zi*wfgk(igk,ispn,ist)
      end do
      call zfftifc(3,ngdgc,1,zfft2)
      jrir_(1:ngtc,i)=jrir_(1:ngtc,i)+wo*aimag(zfft1(1:ngtc)*zfft2(1:ngtc))
    end do
  end do
end do
deallocate(wfgk,zfft1,zfft2)
end subroutine

