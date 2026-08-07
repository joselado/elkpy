
! Copyright (C) 2002-2005 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: linengy
! !INTERFACE:
subroutine linengy
! !USES:
use modmain
use modmpi
use modomp
! !DESCRIPTION:
!   Calculates the new linearisation energies for both the APW and local-orbital
!   radial functions. See the routine {\tt findband}.
!
! !REVISION HISTORY:
!   Created May 2003 (JKD)
!EOP
!BOC
implicit none
! local variables
logical fnd
integer is,ia,ja,ias,jas
integer nr,nri,iro,i0,i1
integer l,io,jo,ilo,nnf,nthd
! automatic arrays
real(8) vr(nrmtmax)
nnf=0
! begin loops over atoms and species
call holdthd(natmtot,nthd)
!$OMP PARALLEL DO DEFAULT(SHARED) &
!$OMP PRIVATE(vr,is,ia,nr,nri,iro,i0,i1) &
!$OMP PRIVATE(l,io,jo,fnd,ilo,ja,jas) &
!$OMP REDUCTION(+:nnf) &
!$OMP SCHEDULE(DYNAMIC) &
!$OMP NUM_THREADS(nthd)
do ias=1,natmtot
  is=idxis(ias)
  ia=idxia(ias)
! perform calculation for only the first equivalent atom
  if (.not.tfeqat(ia,is)) cycle
  nr=nrmt(is)
  nri=nrmti(is)
  iro=nri+1
  i1=lmmaxi*(nri-1)+1
  vr(1:nri)=vsmt(1:i1:lmmaxi,ias)*y00
  i0=i1+lmmaxi
  i1=lmmaxo*(nr-iro)+i0
  vr(iro:nr)=vsmt(i0:i1:lmmaxo,ias)*y00
!-----------------------!
!     APW functions     !
!-----------------------!
  do l=0,lmaxapw
    do io=1,apword(l,is)
      if (apwve(io,l,is)) then
! check if previous radial functions have same default energies
        do jo=1,io-1
          if (apwve(jo,l,is)) then
            if (abs(apwe0(io,l,is)-apwe0(jo,l,is)) < 1.d-4) then
              apwe(io,l,ias)=apwe(jo,l,ias)
              goto 10
            end if
          end if
        end do
! find the band energy starting from default
        apwe(io,l,ias)=apwe0(io,l,is)
        call findband(solsc,l,nr,rlmt(:,1,is),vr,epsband,demaxbnd, &
         apwe(io,l,ias),fnd)
        if (.not.fnd) nnf=nnf+1
      else
! set linearisation energy automatically
        if (autolinengy) apwe(io,l,ias)=efermi+dlefe
      end if
10 continue
    end do
  end do
!---------------------------------!
!     local-orbital functions     !
!---------------------------------!
  do ilo=1,nlorb(is)
    do io=1,lorbord(ilo,is)
      if (lorbve(io,ilo,is)) then
! check if previous radial functions have same default energies
        do jo=1,io-1
          if (lorbve(jo,ilo,is)) then
            if (abs(lorbe0(io,ilo,is)-lorbe0(jo,ilo,is)) < 1.d-4) then
              lorbe(io,ilo,ias)=lorbe(jo,ilo,ias)
              goto 20
            end if
          end if
        end do
        l=lorbl(ilo,is)
! find the band energy starting from default
        lorbe(io,ilo,ias)=lorbe0(io,ilo,is)
        call findband(solsc,l,nr,rlmt(:,1,is),vr,epsband,demaxbnd, &
         lorbe(io,ilo,ias),fnd)
        if (.not.fnd) nnf=nnf+1
      else
! set linearisation energy automatically
        if (autolinengy) lorbe(io,ilo,ias)=efermi+dlefe
      end if
20 continue
    end do
  end do
! copy to equivalent atoms
  do ja=1,natoms(is)
    if (eqatoms(ia,ja,is).and.(ia /= ja)) then
      jas=idxas(ja,is)
      do l=0,lmaxapw
        do io=1,apword(l,is)
          apwe(io,l,jas)=apwe(io,l,ias)
        end do
      end do
      do ilo=1,nlorb(is)
        do io=1,lorbord(ilo,is)
          lorbe(io,ilo,jas)=lorbe(io,ilo,ias)
        end do
      end do
    end if
  end do
! end loop over species and atoms
end do
!$OMP END PARALLEL DO
call freethd(nthd)
if (mp_mpi.and.(nnf > 0)) then
  write(*,*)
  write(*,'("Warning(linengy): could not find ",I0," linearisation energies &
   &in s.c. loop ",I0)') nnf,iscl
end if
end subroutine
!EOC

