
! Copyright (C) 2002-2005 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: genlofr
! !INTERFACE:
subroutine genlofr
! !USES:
use modmain
use modomp
! !DESCRIPTION:
!   Generates the local-orbital radial functions. This is done by integrating
!   the scalar relativistic Schr\"{o}dinger equation (or its energy deriatives)
!   at the current linearisation energies using the spherical part of the
!   Kohn-Sham potential. For each local-orbital, a linear combination of
!   {\tt lorbord} radial functions is constructed such that its radial
!   derivatives up to order ${\tt lorbord}-1$ are zero at the muffin-tin radius.
!   This function is normalised and the radial Hamiltonian applied to it. The
!   results are stored in the global array {\tt lofr}.
!
! !REVISION HISTORY:
!   Created March 2003 (JKD)
!   Copied to equivalent atoms, February 2010 (A. Kozhevnikov and JKD)
!EOP
!BOC
implicit none
! local variables
integer is,ia,ja,ias,jas
integer nr,nri,iro,ir,i,j
integer i0,i1,nn,l,info
integer ilo,jlo,io,jo,nthd
real(8) e,t1
! automatic arrays
integer ipiv(nplorb)
real(8) vr(nrmtmax),p0(nrmtmax,lorbordmax),p1(nrmtmax)
real(8) q0(nrmtmax),q1(nrmtmax),ep0(nrmtmax,lorbordmax)
real(8) p0c(nrmtmax,nlomax),ep0c(nrmtmax,nlomax)
real(8) a(nplorb,nplorb),b(nplorb)
! external functions
real(8), external :: polynm
! loop over all species and atoms
call holdthd(natmtot,nthd)
!$OMP PARALLEL DO DEFAULT(SHARED) &
!$OMP PRIVATE(ipiv,vr,p0,p1,q0,q1) &
!$OMP PRIVATE(ep0,p0c,ep0c,a,b) &
!$OMP PRIVATE(is,ia,nr,nri,iro,i0,i1) &
!$OMP PRIVATE(i,j,ilo,jlo,l,io,jo) &
!$OMP PRIVATE(e,nn,t1,ir,info,ja,jas) &
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
! loop over local-orbitals in ascending energy order
  do i=1,nlorb(is)
    ilo=idxelo(i,is)
    l=lorbl(ilo,is)
    do jo=1,lorbord(ilo,is)
! linearisation energy accounting for energy derivative
      e=lorbe(jo,ilo,ias)+lorbdm(jo,ilo,is)*delorb
! integrate the radial Schrodinger equation
      call rschrodint(solsc,l,e,nr,rlmt(:,1,is),vr,nn,p0(:,jo),p1,q0,q1)
! divide the radial wavefunction by r
      p0(1:nr,jo)=p0(1:nr,jo)*rlmt(1:nr,-1,is)
! multiply by the linearisation energy
      ep0(1:nr,jo)=e*p0(1:nr,jo)
! set up the matrix of radial derivatives
      a(1,jo)=p0(nr,jo)
      ir=nr-nplorb+1
      do io=2,lorbord(ilo,is)
        a(io,jo)=polynm(io-1,nplorb,rlmt(ir,1,is),p0(ir,jo),rmt(is))
      end do
    end do
! set up the target vector
    b(1:nplorb)=0.d0
    b(lorbord(ilo,is))=1.d0
! solve the linear system
    call dgesv(lorbord(ilo,is),1,a,nplorb,ipiv,b,nplorb,info)
    if (info /= 0) call errstp
! generate linear combination of radial functions
    p0c(1:nr,ilo)=0.d0
    ep0c(1:nr,ilo)=0.d0
    do io=1,lorbord(ilo,is)
      t1=b(io)
      p0c(1:nr,ilo)=p0c(1:nr,ilo)+t1*p0(1:nr,io)
      ep0c(1:nr,ilo)=ep0c(1:nr,ilo)+t1*ep0(1:nr,io)
    end do
! normalise radial functions
    t1=sum(wr2mt(1:nr,is)*p0c(1:nr,ilo)**2)
    t1=1.d0/sqrt(abs(t1))
    p0c(1:nr,ilo)=t1*p0c(1:nr,ilo)
    ep0c(1:nr,ilo)=t1*ep0c(1:nr,ilo)
! subtract linear combination of previous local-orbitals with same l
    do j=1,i-1
      jlo=idxelo(j,is)
      if (lorbl(jlo,is) == l) then
        t1=-sum(wr2mt(1:nr,is)*p0c(1:nr,ilo)*p0c(1:nr,jlo))
        p0c(1:nr,ilo)=p0c(1:nr,ilo)+t1*p0c(1:nr,jlo)
        ep0c(1:nr,ilo)=ep0c(1:nr,ilo)+t1*ep0c(1:nr,jlo)
      end if
    end do
! normalise radial functions again
    if (i > 1) then
      t1=sum(wr2mt(1:nr,is)*p0c(1:nr,ilo)**2)
      if (t1 < 1.d-25) call errstp
      t1=1.d0/sqrt(t1)
      p0c(1:nr,ilo)=t1*p0c(1:nr,ilo)
      ep0c(1:nr,ilo)=t1*ep0c(1:nr,ilo)
    end if
! store in global array
    lofr(1:nr,1,ilo,ias)=p0c(1:nr,ilo)
    lofr(1:nr,2,ilo,ias)=ep0c(1:nr,ilo)
  end do
! copy to equivalent atoms
  do ja=1,natoms(is)
    if (eqatoms(ia,ja,is).and.(ia /= ja)) then
      jas=idxas(ja,is)
      do ilo=1,nlorb(is)
        lofr(1:nr,1:2,ilo,jas)=lofr(1:nr,1:2,ilo,ias)
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
    do ilo=1,nlorb(is)
      lofr_sp(1:nrcmt(is),ilo,ias)=lofr(1:nrmt(is):lradstp,1,ilo,ias)
    end do
  end do
end if

contains

subroutine errstp
implicit none
write(*,*)
write(*,'("Error(genlofr): degenerate local-orbital radial functions")')
write(*,'(" for species ",I0)') is
write(*,'(" atom ",I0)') ia
write(*,'(" and local-orbital ",I0)') ilo
write(*,*)
stop
end subroutine

end subroutine
!EOC

