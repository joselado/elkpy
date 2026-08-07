
! Copyright (C) 2007 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine genvmatk(vmt,vir,ngp,igpig,wfmt,ld,wfgp,vmat)
use modmain
use moddftu
use modomp
implicit none
! arguments
! the potential is multiplied by the radial integration weights in the
! muffin-tin and by the characteristic function in the interstitial region
real(8), intent(in) :: vmt(npcmtmax,natmtot),vir(ngtc)
integer, intent(in) :: ngp(nspnfv),igpig(ngkmax,nspnfv)
complex(4), intent(in) :: wfmt(npcmtmax,natmtot,nspinor,nstsv)
integer, intent(in) :: ld
complex(4), intent(in) :: wfgp(ld,nspinor,nstsv)
complex(8), intent(out) :: vmat(nstsv,nstsv)
! local variables
integer ist,jst,nj,ispn,jspn
integer is,ias,nrc,nrci,nrco
integer npc,npc2,ipco
integer ld1,ld2,n,igp,nthd
! automatic arrays
complex(4) wfmt1(npcmtmax),wfmt2(npcmtmax)
complex(4) wfir(ngtc),c(ngkmax),y(nstsv)
! external functions
real(4), external :: sdot
ld1=npcmtmax*natmtot*nspinor
ld2=ld*nspinor
! zero the upper triangular matrix elements
do jst=1,nstsv
  vmat(1:jst,jst)=0.d0
end do
call holdthd(nstsv,nthd)
!$OMP PARALLEL DEFAULT(SHARED) &
!$OMP PRIVATE(wfmt1,wfmt2,wfir,c,y) &
!$OMP PRIVATE(jst,nj,ispn,jspn) &
!$OMP PRIVATE(ias,is,npc,npc2,nrc) &
!$OMP PRIVATE(nrci,nrco,ipco,n,igp) &
!$OMP NUM_THREADS(nthd)
!-------------------------!
!     muffin-tin part     !
!-------------------------!
!$OMP DO SCHEDULE(DYNAMIC)
do jst=1,nstsv
  nj=jst-1
  do ispn=1,nspinor
    do ias=1,natmtot
      is=idxis(ias)
      npc=npcmt(is)
      npc2=npc*2
! apply local potential to wavefunction
      wfmt1(1:npc)=vmt(1:npc,ias)*wfmt(1:npc,ias,ispn,jst)
! apply muffin-tin DFT+U potential matrix if required (note that this should be
! used only in the spin-unpolarised case)
      if (dftu /= 0) then
        if (any(tvmmt(0:lmaxdm,ias))) then
          nrc=nrcmt(is)
          nrci=nrcmti(is)
          nrco=nrc-nrci
          ipco=npcmti(is)+1
          call cgemm('N','N',lmmaxi,nrci,lmmaxi,cone,vmatmti(1,1,1,1,ias), &
           lmmaxi,wfmt(1,ias,ispn,jst),lmmaxi,czero,wfmt2,lmmaxi)
          call cgemm('N','N',lmmaxo,nrco,lmmaxo,cone,vmatmto(1,1,1,1,ias), &
           lmmaxo,wfmt(ipco,ias,ispn,jst),lmmaxo,czero,wfmt2(ipco),lmmaxo)
          call cfcmtwr(nrc,nrci,wr2cmt(:,is),wfmt2)
          wfmt1(1:npc)=wfmt1(1:npc)+wfmt2(1:npc)
        end if
      end if
! compute the inner products
      call cgemv('C',npc,nj,cone,wfmt(1,ias,ispn,1),ld1,wfmt1,1,czero,y,1)
      vmat(1:nj,jst)=vmat(1:nj,jst)+y(1:nj)
      vmat(jst,jst)=vmat(jst,jst)+sdot(npc2,wfmt(1,ias,ispn,jst),1,wfmt1,1)
    end do
  end do
end do
!$OMP END DO
!---------------------------!
!     interstitial part     !
!---------------------------!
!$OMP DO SCHEDULE(DYNAMIC)
do jst=1,nstsv
  nj=jst-1
  do ispn=1,nspinor
    jspn=jspnfv(ispn)
    n=ngp(jspn)
! Fourier transform wavefunction to real-space
    wfir(1:ngtc)=0.e0
    do igp=1,n
      wfir(igfc(igpig(igp,jspn)))=wfgp(igp,ispn,jst)
    end do
    call cfftifc(3,ngdgc,1,wfir)
! apply potential to wavefunction
    wfir(1:ngtc)=vir(1:ngtc)*wfir(1:ngtc)
! Fourier transform to G+p-space
    call cfftifc(3,ngdgc,-1,wfir)
    do igp=1,n
      c(igp)=wfir(igfc(igpig(igp,jspn)))
    end do
! compute the inner products
    call cgemv('C',n,nj,cone,wfgp(1,ispn,1),ld2,c,1,czero,y,1)
    vmat(1:nj,jst)=vmat(1:nj,jst)+y(1:nj)
    vmat(jst,jst)=vmat(jst,jst)+sdot(n*2,wfgp(1,ispn,jst),1,c,1)
  end do
end do
!$OMP END DO
!$OMP END PARALLEL
call freethd(nthd)
! lower triangular part
do ist=1,nstsv
  do jst=1,ist-1
    vmat(ist,jst)=conjg(vmat(jst,ist))
  end do
end do
end subroutine

