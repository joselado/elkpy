
! Copyright (C) 2021 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine wfmtsv_sp(tsh,is,ias,nst,idx,ngp,apwalm,evecfv,evecsv,ld,wfmt)
use modmain
use modomp
implicit none
! arguments
logical, intent(in) :: tsh
integer, intent(in) :: is,ias,nst,idx(*),ngp(nspnfv)
complex(8), intent(in) :: apwalm(ngkmax,apwordmax,lmmaxapw,natmtot,nspnfv)
complex(8), intent(in) :: evecfv(nmatmax,nstfv,nspnfv),evecsv(nstsv,nstsv)
integer, intent(in) :: ld
complex(4), intent(out) :: wfmt(ld,nspinor,nst)
! local variables
logical tasv
integer io,ilo,ispn,jspn
integer nrc,nrci,nrco,irco
integer npc,npci,ipco
integer l,lm,lma,lmb,nm
integer n,p,i,j,k,nthd
complex(8) zq(2)
! automatic arrays
complex(8) x(nstfv,nspnfv),y(nlmwf(is),nspinor,nst)
complex(4) cfmt(npcmtmax),c(2*lmaxo+1)
! external functions
complex(8), external :: zdotu
nrc=nrcmt(is)
nrci=nrcmti(is)
nrco=nrc-nrci
irco=nrci+1
npc=npcmt(is)
npci=npcmti(is)
ipco=npci+1
! de-phasing factor for spin-spirals
if (ssdph) then
  zq(1)=zqss(ias)
  zq(2)=conjg(zq(1))
end if
! check if all the second-variational wavefunctions should be calculated
tasv=(idx(1) == 0)
call holdthd(nst,nthd)
!-----------------------!
!     APW functions     !
!-----------------------!
!$OMP PARALLEL DEFAULT(SHARED) &
!$OMP PRIVATE(cfmt,c,p,l,lma,lmb,io,lm) &
!$OMP PRIVATE(ispn,jspn,n,i,j,k,nm,ilo) &
!$OMP NUM_THREADS(nthd)
p=0
do l=0,lmaxo
  lma=l**2+1; lmb=lma+2*l
  do io=1,apword(l,is)
    do lm=lma,lmb
      p=p+1
      if (tevecsv) then
        do jspn=1,nspnfv
          n=ngp(jspn)
!$OMP DO SCHEDULE(DYNAMIC)
          do j=1,nstfv
            x(j,jspn)=zdotu(n,evecfv(:,j,jspn),1,apwalm(:,io,lm,ias,jspn),1)
          end do
!$OMP END DO
        end do
! loop only over required states
!$OMP DO SCHEDULE(DYNAMIC)
        do j=1,nst
! index to state in evecsv
          k=merge(j,idx(j),tasv)
          y(p,1,j)=zdotu(nstfv,evecsv(1,k),1,x,1)
          if (spinpol) then
            jspn=jspnfv(2)
            y(p,2,j)=zdotu(nstfv,evecsv(nstfv+1,k),1,x(1,jspn),1)
          end if
        end do
!$OMP END DO
      else
!$OMP DO SCHEDULE(DYNAMIC)
        do j=1,nst
          k=merge(j,idx(j),tasv)
          y(p,1,j)=zdotu(ngp(1),evecfv(:,k,1),1,apwalm(:,io,lm,ias,1),1)
        end do
!$OMP END DO
      end if
    end do
  end do
end do
!$OMP DO SCHEDULE(DYNAMIC)
do j=1,nst
  wfmt(1:npc,1:nspinor,j)=0.e0
  do ispn=1,nspinor
    p=0
    do l=0,lmaxo
      nm=2*l+1
      lma=l**2+1
      do io=1,apword(l,is)
        do lm=1,nm
          p=p+1
          c(lm)=y(p,ispn,j)
        end do
        if (ssdph) c(1:nm)=c(1:nm)*zq(ispn)
        if (l <= lmaxi) &
         call cfcrf(nm,nrci,apwfr_sp(1,io,l,ias),c,lmmaxi,wfmt(lma,ispn,j))
        call cfcrf(nm,nrco,apwfr_sp(irco,io,l,ias),c,lmmaxo, &
         wfmt(npci+lma,ispn,j))
      end do
    end do
  end do
end do
!$OMP END DO
!---------------------------------!
!     local-orbital functions     !
!---------------------------------!
p=0
do ilo=1,nlorb(is)
  l=lorbl(ilo,is)
  do lm=l**2+1,(l+1)**2
    p=p+1
    i=idxlo(lm,ilo,ias)
    if (tevecsv) then
      do jspn=1,nspnfv
        n=ngp(jspn)
        x(1:nstfv,jspn)=evecfv(n+i,1:nstfv,jspn)
      end do
!$OMP DO SCHEDULE(DYNAMIC)
      do j=1,nst
        k=merge(j,idx(j),tasv)
        y(p,1,j)=zdotu(nstfv,evecsv(1,k),1,x,1)
        if (spinpol) then
          jspn=jspnfv(2)
          y(p,2,j)=zdotu(nstfv,evecsv(nstfv+1,k),1,x(1,jspn),1)
        end if
      end do
!$OMP END DO
    else
      do j=1,nst
        k=merge(j,idx(j),tasv)
        y(p,1,j)=evecfv(ngp(1)+i,k,1)
      end do
    end if
  end do
end do
!$OMP DO SCHEDULE(DYNAMIC)
do j=1,nst
  do ispn=1,nspinor
    p=0
    do ilo=1,nlorb(is)
      l=lorbl(ilo,is)
      nm=2*l+1
      lma=l**2+1
      do lm=1,nm
        p=p+1
        c(lm)=y(p,ispn,j)
      end do
      if (ssdph) c(1:nm)=c(1:nm)*zq(ispn)
      if (l <= lmaxi) &
       call cfcrf(nm,nrci,lofr_sp(1,ilo,ias),c,lmmaxi,wfmt(lma,ispn,j))
      call cfcrf(nm,nrco,lofr_sp(irco,ilo,ias),c,lmmaxo,wfmt(npci+lma,ispn,j))
    end do
! convert to spherical coordinates if required
    if (.not.tsh) then
      cfmt(1:npc)=wfmt(1:npc,ispn,j)
      call cgemm('N','N',lmmaxi,nrci,lmmaxi,cone,cbshti,lmmaxi,cfmt,lmmaxi, &
       czero,wfmt(:,ispn,j),lmmaxi)
      call cgemm('N','N',lmmaxo,nrco,lmmaxo,cone,cbshto,lmmaxo,cfmt(ipco), &
       lmmaxo,czero,wfmt(ipco,ispn,j),lmmaxo)
    end if
  end do
end do
!$OMP END DO
!$OMP END PARALLEL
call freethd(nthd)

contains

pure subroutine cfcrf(m,n,rf,c,ld,cf)
implicit none
! arguments
integer, intent(in) :: m,n
real(4), intent(in) :: rf(n)
complex(4), intent(in) :: c(m)
integer, intent(in) :: ld
complex(4), intent(inout) :: cf(ld,n)
! local variables
integer i
do i=1,m
  cf(i,1:n)=cf(i,1:n)+c(i)*rf(1:n)
end do
end subroutine

end subroutine

