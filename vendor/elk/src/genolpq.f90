
! Copyright (C) 2019 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine genolpq(nst,expqmt,ngpq,igpqig,wfmt,wfir,wfmtq,wfgpq,oq)
use modmain
use modomp
implicit none
! arguments
integer, intent(in) :: nst
complex(8), intent(in) :: expqmt(npcmtmax,natmtot)
integer, intent(in) :: ngpq(nspnfv),igpqig(ngkmax,nspnfv)
complex(8), intent(in) :: wfmt(npcmtmax,natmtot,nspinor,nst)
complex(8), intent(in) :: wfir(ngtc,nspinor,nst)
complex(8), intent(in) :: wfmtq(npcmtmax,natmtot,nspinor,nst)
complex(8), intent(in) :: wfgpq(ngkmax,nspinor,nst)
complex(8), intent(out) :: oq(nst,nst)
! local variables
integer jst,ispn,jspn
integer is,ias,nrc,nrci,npc
integer ld1,ld2,igpq,nthd
real(8) t0
! automatic arrays
complex(8) wfmt1(npcmtmax),wfir1(ngtc),z(ngkmax)
ld1=npcmtmax*natmtot*nspinor
ld2=ngkmax*nspinor
t0=sqrt(omega)
! zero the matrix elements
oq(1:nst,1:nst)=0.d0
call holdthd(nst,nthd)
!$OMP PARALLEL DEFAULT(SHARED) &
!$OMP PRIVATE(wfmt1,wfir1,z) &
!$OMP PRIVATE(jst,ispn,jspn,igpq) &
!$OMP PRIVATE(ias,is,nrc,nrci,npc) &
!$OMP NUM_THREADS(nthd)
!---------------------------!
!     interstitial part     !
!---------------------------!
!$OMP DO SCHEDULE(DYNAMIC)
do jst=1,nst
  do ispn=1,nspinor
    jspn=jspnfv(ispn)
! multiply wavefunction by characteristic function
    wfir1(1:ngtc)=wfir(1:ngtc,ispn,jst)*cfrc(1:ngtc)
! Fourier transform to G+p+q-space
    call zfftifc(3,ngdgc,-1,wfir1)
    do igpq=1,ngpq(jspn)
      z(igpq)=wfir1(igfc(igpqig(igpq,jspn)))
    end do
    call zgemv('C',ngpq(jspn),nst,zone,wfgpq(1,ispn,1),ld2,z,1,zone,oq(1,jst),1)
  end do
end do
!$OMP END DO
! scale the matrix elements because of the real-space wavefunction normalisation
!$OMP SINGLE
oq(1:nst,1:nst)=t0*oq(1:nst,1:nst)
!$OMP END SINGLE
!-------------------------!
!     muffin-tin part     !
!-------------------------!
!$OMP DO SCHEDULE(DYNAMIC)
do jst=1,nst
  do ispn=1,nspinor
    do ias=1,natmtot
      is=idxis(ias)
      nrc=nrcmt(is)
      nrci=nrcmti(is)
      npc=npcmt(is)
! multiply by local phase factor function exp(iq⋅r)
      wfmt1(1:npc)=expqmt(1:npc,ias)*wfmt(1:npc,ias,ispn,jst)
! apply the radial integral weights
      call zfcmtwr(nrc,nrci,wr2cmt(:,is),wfmt1)
! compute the inner products
      call zgemv('C',npc,nst,zone,wfmtq(1,ias,ispn,1),ld1,wfmt1,1,zone, &
       oq(1,jst),1)
    end do
  end do
end do
!$OMP END DO
!$OMP END PARALLEL
call freethd(nthd)
end subroutine

