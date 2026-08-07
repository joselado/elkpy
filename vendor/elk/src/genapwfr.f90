
! Copyright (C) 2002-2005 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: genapwfr
! !INTERFACE:
subroutine genapwfr
! !USES:
use modmain
use modomp
! !DESCRIPTION:
!   Generates the APW radial functions. This is done by integrating the scalar
!   relativistic Schr\"{o}dinger equation (or its energy deriatives) at the
!   current linearisation energies using the spherical part of the Kohn-Sham
!   potential. The number of radial functions at each $l$-value is given by the
!   variable {\tt apword} (at the muffin-tin boundary, the APW functions have
!   continuous derivatives up to order ${\tt apword}-1$). Within each $l$, these
!   functions are orthonormalised with the Gram-Schmidt method. The radial
!   Hamiltonian is applied to the orthonormalised functions and the results are
!   stored in the global array {\tt apwfr}.
!
! !REVISION HISTORY:
!   Created March 2003 (JKD)
!   Copied to equivalent atoms, February 2010 (A. Kozhevnikov and JKD)
!EOP
!BOC
implicit none
! local variables
integer is,ia,ja,ias,jas
integer nr,nri,iro,nthd
integer i0,i1,nn,l,io,jo
real(8) e,t1
! automatic arrays
real(8) vr(nrmtmax),p0(nrmtmax,apwordmax),p1(nrmtmax),p1s(apwordmax)
real(8) q0(nrmtmax),q1(nrmtmax),ep0(nrmtmax,apwordmax)
! loop over all species and atoms
call holdthd(natmtot,nthd)
!$OMP PARALLEL DO DEFAULT(SHARED) &
!$OMP PRIVATE(vr,p0,p1,p1s,q0,q1,ep0) &
!$OMP PRIVATE(is,ia,nr,nri,iro,i0,i1) &
!$OMP PRIVATE(l,io,jo,e,nn,t1,ja,jas) &
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
! use spherical part of potential
  i1=lmmaxi*(nri-1)+1
  vr(1:nri)=vsmt(1:i1:lmmaxi,ias)*y00
  i0=i1+lmmaxi
  i1=lmmaxo*(nr-iro)+i0
  vr(iro:nr)=vsmt(i0:i1:lmmaxo,ias)*y00
  do l=0,lmaxapw
    do io=1,apword(l,is)
! linearisation energy accounting for energy derivative
      e=apwe(io,l,ias)+apwdm(io,l,is)*deapw
! integrate the radial Schrodinger equation
      call rschrodint(solsc,l,e,nr,rlmt(:,1,is),vr,nn,p0(:,io),p1,q0,q1)
! divide the radial wavefunction by r
      p0(1:nr,io)=p0(1:nr,io)*rlmt(1:nr,-1,is)
! multiply by the linearisation energy
      ep0(1:nr,io)=e*p0(1:nr,io)
! normalise radial functions
      t1=sum(wr2mt(1:nr,is)*p0(1:nr,io)**2)
      t1=1.d0/sqrt(abs(t1))
      p0(1:nr,io)=t1*p0(1:nr,io)
      p1s(io)=t1*p1(nr)
      ep0(1:nr,io)=t1*ep0(1:nr,io)
! subtract linear combination of previous vectors
      do jo=1,io-1
        t1=-sum(wr2mt(1:nr,is)*p0(1:nr,io)*p0(1:nr,jo))
        p0(1:nr,io)=p0(1:nr,io)+t1*p0(1:nr,jo)
        p1s(io)=p1s(io)+t1*p1s(jo)
        ep0(1:nr,io)=ep0(1:nr,io)+t1*ep0(1:nr,jo)
      end do
! normalise radial functions again
      if (io > 1) then
        t1=sum(wr2mt(1:nr,is)*p0(1:nr,io)**2)
        if (t1 < 1.d-25) then
          write(*,*)
          write(*,'("Error(genapwfr): degenerate APW radial functions")')
          write(*,'(" for species ",I0)') is
          write(*,'(" atom ",I0)') ia
          write(*,'(" angular momentum ",I0)') l
          write(*,'(" and order ",I0)') io
          write(*,*)
          stop
        end if
        t1=1.d0/sqrt(t1)
        p0(1:nr,io)=t1*p0(1:nr,io)
        p1s(io)=t1*p1s(io)
        ep0(1:nr,io)=t1*ep0(1:nr,io)
      end if
! store in global array
      apwfr(1:nr,1,io,l,ias)=p0(1:nr,io)
      apwfr(1:nr,2,io,l,ias)=ep0(1:nr,io)
! derivative at the muffin-tin surface multiplied by Rₘₜ²/2
      apwdfr(io,l,ias)=(p1s(io)-p0(nr,io))*rmt(is)/2.d0
    end do
  end do
! copy to equivalent atoms
  do ja=1,natoms(is)
    if (eqatoms(ia,ja,is).and.(ia /= ja)) then
      jas=idxas(ja,is)
      do l=0,lmaxapw
        do io=1,apword(l,is)
          apwfr(1:nr,1:2,io,l,jas)=apwfr(1:nr,1:2,io,l,ias)
          apwdfr(io,l,jas)=apwdfr(io,l,ias)
        end do
      end do
    end if
  end do
! end loop over atoms and species
end do
!$OMP END PARALLEL DO
call freethd(nthd)
! make single-precision copy if required
if (tfr_sp) then
  do ias=1,natmtot
    is=idxis(ias)
    do l=0,lmaxapw
      do io=1,apword(l,is)
        apwfr_sp(1:nrcmt(is),io,l,ias)=apwfr(1:nrmt(is):lradstp,1,io,l,ias)
      end do
    end do
  end do
end if
end subroutine
!EOC

