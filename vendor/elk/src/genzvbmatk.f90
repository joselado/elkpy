
! Copyright (C) 2018 T. Mueller, J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine genzvbmatk(zvmt,zvir,zbmt,zbir,ngp,igpig,wfmt,wfir,wfgp,vbmat)
use modmain
use modomp
implicit none
! arguments
! the potential and field are multiplied by the radial integration weights in
! the muffin-tin and by the characteristic function in the interstitial region
complex(8), intent(in) :: zvmt(npcmtmax,natmtot),zvir(ngtc)
complex(8), intent(in) :: zbmt(npcmtmax,natmtot,ndmag),zbir(ngtc,ndmag)
integer, intent(in) :: ngp,igpig(ngp)
complex(4), intent(in) :: wfmt(npcmtmax,natmtot,nspinor,nstsv)
! note that wfir does not have a 1/sqrt(omega) prefactor
complex(4), intent(in) :: wfir(ngtc,nspinor,nstsv)
complex(4), intent(in) :: wfgp(ngp,nspinor,nstsv)
complex(8), intent(out) :: vbmat(nstsv,nstsv)
! local variables
integer jst,ispn,nthd
integer is,ias,npc
integer ld1,ld2,igp
! automatic arrays
complex(4) wfmt1(npcmtmax,nspinor),wfir1(ngtc,nspinor),y(nstsv),c(ngp)
ld1=npcmtmax*natmtot*nspinor
ld2=ngp*nspinor
! zero the matrix elements
vbmat(1:nstsv,1:nstsv)=0.d0
call holdthd(nstsv,nthd)
!$OMP PARALLEL DEFAULT(SHARED) &
!$OMP PRIVATE(wfmt1,wfir1,y,c) &
!$OMP PRIVATE(jst,ias,is) &
!$OMP PRIVATE(npc,ispn,igp) &
!$OMP NUM_THREADS(nthd)
!-------------------------!
!     muffin-tin part     !
!-------------------------!
!$OMP DO SCHEDULE(DYNAMIC)
do jst=1,nstsv
  do ias=1,natmtot
    is=idxis(ias)
    npc=npcmt(is)
! apply local potential and magnetic field to spinor wavefunction
    if (ncmag) then
! non-collinear case
      call zvbmk1(npc,zvmt(:,ias),zbmt(:,ias,1),zbmt(:,ias,2),zbmt(:,ias,3), &
       wfmt(:,ias,1,jst),wfmt(:,ias,2,jst),wfmt1,wfmt1(:,2))
    else
! collinear case
      call zvbmk2(npc,zvmt(:,ias),zbmt(:,ias,1),wfmt(:,ias,1,jst), &
       wfmt(:,ias,2,jst),wfmt1,wfmt1(:,2))
    end if
! compute the inner products
    call cgemv('C',npc,nstsv,cone,wfmt(1,ias,1,1),ld1,wfmt1(1,1),1,czero,y,1)
    call cgemv('C',npc,nstsv,cone,wfmt(1,ias,2,1),ld1,wfmt1(1,2),1,cone,y,1)
    vbmat(1:nstsv,jst)=vbmat(1:nstsv,jst)+y(1:nstsv)
  end do
end do
!$OMP END DO
!---------------------------!
!     interstitial part     !
!---------------------------!
!$OMP DO SCHEDULE(DYNAMIC)
do jst=1,nstsv
! apply local potential and magnetic field to spinor wavefunction
  if (ncmag) then
! non-collinear case
    call zvbmk1(ngtc,zvir,zbir,zbir(:,2),zbir(:,3),wfir(:,1,jst), &
     wfir(:,2,jst),wfir1,wfir1(:,2))
  else
! collinear case
    call zvbmk2(ngtc,zvir,zbir,wfir(:,1,jst),wfir(:,2,jst),wfir1,wfir1(:,2))
  end if
  do ispn=1,nspinor
! Fourier transform to G+p-space
    call cfftifc(3,ngdgc,-1,wfir1(:,ispn))
    do igp=1,ngp
      c(igp)=wfir1(igfc(igpig(igp)),ispn)
    end do
    call cgemv('C',ngp,nstsv,cone,wfgp(1,ispn,1),ld2,c,1,czero,y,1)
    vbmat(1:nstsv,jst)=vbmat(1:nstsv,jst)+y(1:nstsv)
  end do
end do
!$OMP END DO
!$OMP END PARALLEL
call freethd(nthd)

contains

pure subroutine zvbmk1(n,zv,zb1,zb2,zb3,wf11,wf12,wf21,wf22)
implicit none
! arguments
integer, intent(in) :: n
complex(8), intent(in) :: zv(n),zb1(n),zb2(n),zb3(n)
complex(4), intent(in) :: wf11(n),wf12(n)
complex(4), intent(out) :: wf21(n),wf22(n)
! local variables
integer i
complex(8) z1
do i=1,n
  z1=cmplx(-zb2(i)%im,zb2(i)%re,8)
  wf21(i)=(zv(i)+zb3(i))*wf11(i)+(zb1(i)-z1)*wf12(i)
  wf22(i)=(zv(i)-zb3(i))*wf12(i)+(zb1(i)+z1)*wf11(i)
end do
end subroutine

pure subroutine zvbmk2(n,zv,zb,wf11,wf12,wf21,wf22)
implicit none
! arguments
integer, intent(in) :: n
complex(8), intent(in) :: zv(n),zb(n)
complex(4), intent(in) :: wf11(n),wf12(n)
complex(4), intent(out) :: wf21(n),wf22(n)
! local variables
integer i
do i=1,n
  wf21(i)=(zv(i)+zb(i))*wf11(i)
  wf22(i)=(zv(i)-zb(i))*wf12(i)
end do
end subroutine

end subroutine

