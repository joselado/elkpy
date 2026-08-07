
! Copyright (C) 2006 F. Bultmark, F. Cricchio, L. Nordstrom and J. K. Dewhurst.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine eveqnss(ngp,igpig,apwalm,evalfv,evecfv,evalsv_,evecsv)
use modmain
use moddftu
use modomp
implicit none
! arguments
integer, intent(in) :: ngp(nspnfv),igpig(ngkmax,nspnfv)
complex(8), intent(in) :: apwalm(ngkmax,apwordmax,lmmaxapw,natmtot,nspnfv)
real(8), intent(in) :: evalfv(nstfv,nspnfv)
complex(8), intent(in) :: evecfv(nmatmax,nstfv,nspnfv)
real(8), intent(out) :: evalsv_(nstsv)
complex(8), intent(out) :: evecsv(nstsv,nstsv)
! local variables
integer ld,ist,jst,ispn,is,ias
integer nrc,nrci,nrco,nj,i,j
integer l,lm,nm,npc,npc2,npci
integer n1,n12,n2,n22,igp,nthd
real(8) ts0,ts1
complex(8) zq
! automatic arrays
complex(8) wfmt2(npcmtmax,nspnfv),wfmt4(npcmtmax)
complex(8) wfmt31(npcmtmax),wfmt32(npcmtmax),wfmt33(npcmtmax)
complex(4) wfmt5(npcmtmax),wfgp1(ngkmax),wfgp2(ngkmax),wfgp3(ngkmax)
complex(4) wfir1(ngtc,nspnfv),wfir2(ngtc),y(nstfv)
! allocatable arrays
complex(4), allocatable :: wfmt0(:,:,:),wfgp0(:,:,:)
complex(8), allocatable :: wfmt1(:,:,:)
! external functions
real(4), external :: sdot
if (.not.spinpol) then
  write(*,*)
  write(*,'("Error(eveqnss): spin-unpolarised calculation")')
  write(*,*)
  stop
end if
call timesec(ts0)
ld=lmmaxdm*nspinor
! zero the second-variational Hamiltonian (stored in the eigenvector array)
evecsv(1:nstsv,1:nstsv)=0.d0
! set the diagonal elements equal to the first-variational eigenvalues
do ispn=1,nspinor
  do ist=1,nstfv
    i=nstfv*(ispn-1)+ist
    evecsv(i,i)=evalfv(ist,ispn)
  end do
end do
call holdthd(nstfv,nthd)
!$OMP PARALLEL DEFAULT(SHARED) &
!$OMP PRIVATE(wfmt2,wfmt31,wfmt32,wfmt33,wfmt4,wfmt5) &
!$OMP PRIVATE(wfir1,wfir2,wfgp1,wfgp2,wfgp3,y) &
!$OMP PRIVATE(ias,is,nrc,nrci,nrco,npc,npc2,npci) &
!$OMP PRIVATE(zq,ispn,ist,jst,l,nm,lm,i,j,nj,igp) &
!$OMP NUM_THREADS(nthd)
!-------------------------!
!     muffin-tin part     !
!-------------------------!
!$OMP SINGLE
allocate(wfmt0(npcmtmax,nstfv,nspnfv),wfmt1(npcmtmax,nstfv,nspnfv))
!$OMP END SINGLE
do ias=1,natmtot
  is=idxis(ias)
  nrc=nrcmt(is)
  nrci=nrcmti(is)
  nrco=nrc-nrci
  npc=npcmt(is)
  npc2=npc*2
  npci=npcmti(is)
! de-phasing factor (FC, FB & LN)
  if (ssdph) zq=zqss(ias)
! compute the first-variational wavefunctions
  do ispn=1,nspnfv
    if (ssdph.and.(ispn == 2)) zq=conjg(zq)
!$OMP DO SCHEDULE(DYNAMIC)
    do ist=1,nstfv
      call wfmtfv(ias,ngp(ispn),apwalm(:,:,:,ias,ispn),evecfv(:,ist,ispn), &
       wfmt1(:,ist,ispn))
! de-phase if required
      if (ssdph) wfmt1(1:npc,ist,ispn)=zq*wfmt1(1:npc,ist,ispn)
! multiply wavefunction by integration weights and store as single-precision
      call zcfmtwr(nrc,nrci,wr2cmt(:,is),wfmt1(:,ist,ispn),wfmt0(:,ist,ispn))
    end do
!$OMP END DO
  end do
!$OMP DO SCHEDULE(DYNAMIC)
  do jst=1,nstfv
! convert wavefunction to spherical coordinates
    do ispn=1,nspnfv
      call zbsht(nrc,nrci,wfmt1(:,jst,ispn),wfmt2(:,ispn))
    end do
! apply effective magnetic field and convert to spherical harmonics
    wfmt4(1:npc)=bsmt(1:npc,ias,3)*wfmt2(1:npc,1)
    call zfsht(nrc,nrci,wfmt4,wfmt31)
    wfmt4(1:npc)=-bsmt(1:npc,ias,3)*wfmt2(1:npc,2)
    call zfsht(nrc,nrci,wfmt4,wfmt32)
    wfmt4(1:npc)=cmplx(bsmt(1:npc,ias,1),-bsmt(1:npc,ias,2),8)*wfmt2(1:npc,2)
    call zfsht(nrc,nrci,wfmt4,wfmt33)
! apply muffin-tin DFT+U potential matrix if required
    if (tvmatmt) then
      do l=0,lmaxdm
        if (tvmmt(l,ias)) then
          nm=2*l+1
          lm=l**2+1
          i=npci+lm
          if (l <= lmaxi) &
           call zgemm('N','N',nm,nrci,nm,zone,vmatmt(lm,1,lm,1,ias),ld, &
            wfmt1(lm,jst,1),lmmaxi,zone,wfmt31(lm),lmmaxi)
          call zgemm('N','N',nm,nrco,nm,zone,vmatmt(lm,1,lm,1,ias),ld, &
           wfmt1(i,jst,1),lmmaxo,zone,wfmt31(i),lmmaxo)
          if (l <= lmaxi) &
           call zgemm('N','N',nm,nrci,nm,zone,vmatmt(lm,2,lm,2,ias),ld, &
            wfmt1(lm,jst,2),lmmaxi,zone,wfmt32(lm),lmmaxi)
          call zgemm('N','N',nm,nrco,nm,zone,vmatmt(lm,2,lm,2,ias),ld, &
           wfmt1(i,jst,2),lmmaxo,zone,wfmt32(i),lmmaxo)
          if (l <= lmaxi) &
           call zgemm('N','N',nm,nrci,nm,zone,vmatmt(lm,1,lm,2,ias),ld, &
            wfmt1(lm,jst,2),lmmaxi,zone,wfmt33(lm),lmmaxi)
          call zgemm('N','N',nm,nrco,nm,zone,vmatmt(lm,1,lm,2,ias),ld, &
           wfmt1(i,jst,2),lmmaxo,zone,wfmt33(i),lmmaxo)
        end if
      end do
    end if
! add to second-variational Hamiltonian matrix
    nj=jst-1
! upper diagonal block
    wfmt5(1:npc)=wfmt31(1:npc)
    call cgemv('C',npc,nj,cone,wfmt0,npcmtmax,wfmt5,1,czero,y,1)
    evecsv(1:nj,jst)=evecsv(1:nj,jst)+y(1:nj)
    evecsv(jst,jst)=evecsv(jst,jst)+sdot(npc2,wfmt0(:,jst,1),1,wfmt5,1)
    j=jst+nstfv
! lower diagonal block
    wfmt5(1:npc)=wfmt32(1:npc)
    call cgemv('C',npc,nj,cone,wfmt0(:,:,2),npcmtmax,wfmt5,1,czero,y,1)
    evecsv(nstfv+1:nstfv+nj,j)=evecsv(nstfv+1:nstfv+nj,j)+y(1:nj)
    evecsv(j,j)=evecsv(j,j)+sdot(npc2,wfmt0(:,jst,2),1,wfmt5,1)
! off-diagonal block
    wfmt5(1:npc)=wfmt33(1:npc)
    call cgemv('C',npc,nstfv,cone,wfmt0,npcmtmax,wfmt5,1,czero,y,1)
    evecsv(1:nstfv,j)=evecsv(1:nstfv,j)+y(1:nstfv)
  end do
!$OMP END DO
! end loop over atoms
end do
!$OMP SINGLE
deallocate(wfmt0,wfmt1)
!$OMP END SINGLE
!---------------------------!
!     interstitial part     !
!---------------------------!
!$OMP SINGLE
n1=ngp(1); n12=n1*2
n2=ngp(2); n22=n2*2
allocate(wfgp0(ngkmax,nstfv,nspnfv))
!$OMP END SINGLE
! make single-precision copy of wavefunction
!$OMP DO SCHEDULE(DYNAMIC)
  do ist=1,nstfv
    do ispn=1,nspnfv
      wfgp0(1:ngp(ispn),ist,ispn)=evecfv(1:ngp(ispn),ist,ispn)
    end do
  end do
!$OMP END DO
! begin loop over states
!$OMP DO SCHEDULE(DYNAMIC)
do jst=1,nstfv
  do ispn=1,nspnfv
    wfir1(:,ispn)=0.e0
    do igp=1,ngp(ispn)
      wfir1(igfc(igpig(igp,ispn)),ispn)=wfgp0(igp,jst,ispn)
    end do
! Fourier transform wavefunction to real-space
    call cfftifc(3,ngdgc,1,wfir1(:,ispn))
  end do
! multiply with magnetic field and transform to G-space
  wfir2(1:ngtc)=bsirc(1:ngtc,3)*wfir1(1:ngtc,1)
  call cfftifc(3,ngdgc,-1,wfir2)
  do igp=1,n1
    wfgp1(igp)=wfir2(igfc(igpig(igp,1)))
  end do
  wfir2(1:ngtc)=-bsirc(1:ngtc,3)*wfir1(1:ngtc,2)
  call cfftifc(3,ngdgc,-1,wfir2)
  do igp=1,n2
    wfgp2(igp)=wfir2(igfc(igpig(igp,2)))
  end do
  wfir2(1:ngtc)=cmplx(bsirc(1:ngtc,1),-bsirc(1:ngtc,2),8)*wfir1(1:ngtc,2)
  call cfftifc(3,ngdgc,-1,wfir2)
  do igp=1,n1
    wfgp3(igp)=wfir2(igfc(igpig(igp,1)))
  end do
! add to second-variational Hamiltonian matrix
  nj=jst-1
! upper diagonal block
  call cgemv('C',n1,nj,cone,wfgp0,ngkmax,wfgp1,1,czero,y,1)
  evecsv(1:nj,jst)=evecsv(1:nj,jst)+y(1:nj)
  evecsv(jst,jst)=evecsv(jst,jst)+sdot(n12,wfgp0(:,jst,1),1,wfgp1,1)
! lower diagonal block
  j=jst+nstfv
  call cgemv('C',n2,nj,cone,wfgp0(:,:,2),ngkmax,wfgp2,1,czero,y,1)
  evecsv(nstfv+1:nstfv+nj,j)=evecsv(nstfv+1:nstfv+nj,j)+y(1:nj)
  evecsv(j,j)=evecsv(j,j)+sdot(n22,wfgp0(:,jst,2),1,wfgp2,1)
! off-diagonal block
  call cgemv('C',n1,nstfv,cone,wfgp0,ngkmax,wfgp3,1,czero,y,1)
  evecsv(1:nstfv,j)=evecsv(1:nstfv,j)+y(1:nstfv)
end do
!$OMP END DO
!$OMP SINGLE
deallocate(wfgp0)
!$OMP END SINGLE
!$OMP END PARALLEL
call freethd(nthd)
! diagonalise the second-variational Hamiltonian
call zheevdi(nstsv,nstsv,evecsv,evalsv_)
call timesec(ts1)
!$OMP ATOMIC
timesv=timesv+ts1-ts0
end subroutine

