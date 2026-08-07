
! Copyright (C) 2017 T. Mueller, J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine genzvmatk(zvmt,zvir,ngp,igpig,wfmt,wfir,wfgp,vmat)
use modmain
use modomp
implicit none
! arguments
! the potential is multiplied by the radial integration weights in the
! muffin-tin and by the characteristic function in the interstitial region
complex(8), intent(in) :: zvmt(npcmtmax,natmtot),zvir(ngtc)
integer, intent(in) :: ngp,igpig(ngp)
complex(4), intent(in) :: wfmt(npcmtmax,natmtot,nspinor,nstsv)
! note that wfir does not have a 1/sqrt(omega) prefactor
complex(4), intent(in) :: wfir(ngtc,nspinor,nstsv),wfgp(ngp,nspinor,nstsv)
complex(8), intent(out) :: vmat(nstsv,nstsv)
! local variables
integer jst,ispn,nthd
integer is,ias,npc
integer ld1,ld2,igp
! automatic arrays
complex(4) wfmt1(npcmtmax),wfir1(ngtc),y(nstsv),c(ngp)
ld1=npcmtmax*natmtot*nspinor
ld2=ngp*nspinor
! zero the matrix elements
vmat(1:nstsv,1:nstsv)=0.d0
call holdthd(nstsv,nthd)
!$OMP PARALLEL DEFAULT(SHARED) &
!$OMP PRIVATE(wfmt1,wfir1,y,c) &
!$OMP PRIVATE(jst,ispn,ias) &
!$OMP PRIVATE(is,npc,igp) &
!$OMP NUM_THREADS(nthd)
!-------------------------!
!     muffin-tin part     !
!-------------------------!
!$OMP DO SCHEDULE(DYNAMIC)
do jst=1,nstsv
  do ispn=1,nspinor
    do ias=1,natmtot
      is=idxis(ias)
      npc=npcmt(is)
! apply complex potential to wavefunction
      wfmt1(1:npc)=zvmt(1:npc,ias)*wfmt(1:npc,ias,ispn,jst)
! compute the inner products
      call cgemv('C',npc,nstsv,cone,wfmt(1,ias,ispn,1),ld1,wfmt1,1,czero,y,1)
      vmat(1:nstsv,jst)=vmat(1:nstsv,jst)+y(1:nstsv)
    end do
  end do
end do
!$OMP END DO
!---------------------------!
!     interstitial part     !
!---------------------------!
!$OMP DO SCHEDULE(DYNAMIC)
do jst=1,nstsv
  do ispn=1,nspinor
! apply potential to wavefunction
    wfir1(1:ngtc)=zvir(1:ngtc)*wfir(1:ngtc,ispn,jst)
! Fourier transform to G+p-space
    call cfftifc(3,ngdgc,-1,wfir1)
    do igp=1,ngp
      c(igp)=wfir1(igfc(igpig(igp)))
    end do
    call cgemv('C',ngp,nstsv,cone,wfgp(1,ispn,1),ld2,c,1,czero,y,1)
    vmat(1:nstsv,jst)=vmat(1:nstsv,jst)+y(1:nstsv)
  end do
end do
!$OMP END DO
!$OMP END PARALLEL
call freethd(nthd)
end subroutine

