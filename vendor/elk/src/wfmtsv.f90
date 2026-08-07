
! Copyright (C) 2021 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine wfmtsv(tsh,lrstp,is,ias,nst,idx,ngp,apwalm,evecfv,evecsv,ld,wfmt)
use modmain
use modomp
implicit none
! arguments
logical, intent(in) :: tsh
integer, intent(in) :: lrstp,is,ias,nst,idx(*),ngp(nspnfv)
complex(8), intent(in) :: apwalm(ngkmax,apwordmax,lmmaxapw,natmtot,nspnfv)
complex(8), intent(in) :: evecfv(nmatmax,nstfv,nspnfv),evecsv(nstsv,nstsv)
integer, intent(in) :: ld
complex(8), intent(out) :: wfmt(ld,nspinor,nst)
! local variables
logical tasv
integer io,ilo,ispn,jspn
integer nr,nri,nro,iro
integer np,npi,ipo
integer l,lm,lma,lmb,nm
integer n,p,i,j,k,nthd
complex(8) zq(2)
! automatic arrays
complex(8) x(nstfv,nspnfv),y(nlmwf(is),nspinor,nst)
complex(8) zfmt(npmtmax),z(2*lmaxo+1)
! external functions
complex(8), external :: zdotu
iro=nrmti(is)+lrstp
if (lrstp == 1) then
  nr=nrmt(is)
  nri=nrmti(is)
  np=npmt(is)
  npi=npmti(is)
else
  nr=nrcmt(is)
  nri=nrcmti(is)
  np=npcmt(is)
  npi=npcmti(is)
end if
nro=nr-nri
ipo=npi+1
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
!$OMP PRIVATE(zfmt,z,p,l,lma,lmb,io,lm) &
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
  wfmt(1:np,1:nspinor,j)=0.d0
  do ispn=1,nspinor
    p=0
    do l=0,lmaxo
      nm=2*l+1
      lma=l**2+1
      do io=1,apword(l,is)
        do lm=1,nm
          p=p+1
          z(lm)=y(p,ispn,j)
        end do
        if (ssdph) z(1:nm)=z(1:nm)*zq(ispn)
        if (l <= lmaxi) &
         call zfzrf(nm,nri,apwfr(1,1,io,l,ias),z,lmmaxi,wfmt(lma,ispn,j))
        call zfzrf(nm,nro,apwfr(iro,1,io,l,ias),z,lmmaxo,wfmt(npi+lma,ispn,j))
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
        z(lm)=y(p,ispn,j)
      end do
      if (ssdph) z(1:nm)=z(1:nm)*zq(ispn)
      if (l <= lmaxi) &
       call zfzrf(nm,nri,lofr(1,1,ilo,ias),z,lmmaxi,wfmt(lma,ispn,j))
      call zfzrf(nm,nro,lofr(iro,1,ilo,ias),z,lmmaxo,wfmt(npi+lma,ispn,j))
    end do
! convert to spherical coordinates if required
    if (.not.tsh) then
      zfmt(1:np)=wfmt(1:np,ispn,j)
      call zgemm('N','N',lmmaxi,nri,lmmaxi,zone,zbshti,lmmaxi,zfmt,lmmaxi, &
       zzero,wfmt(:,ispn,j),lmmaxi)
      call zgemm('N','N',lmmaxo,nro,lmmaxo,zone,zbshto,lmmaxo,zfmt(ipo),lmmaxo,&
       zzero,wfmt(ipo,ispn,j),lmmaxo)
    end if
  end do
end do
!$OMP END DO
!$OMP END PARALLEL
call freethd(nthd)

contains

pure subroutine zfzrf(m,n,rf,z,ld,zf)
implicit none
! arguments
integer, intent(in) :: m,n
real(8), intent(in) :: rf(lrstp,n)
complex(8), intent(in) :: z(m)
integer, intent(in) :: ld
complex(8), intent(inout) :: zf(ld,n)
! local variables
integer i
do i=1,m
  zf(i,1:n)=zf(i,1:n)+z(i)*rf(1,1:n)
end do
end subroutine

end subroutine

