
! Copyright (C) 2002-2005 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: mossbauer
! !INTERFACE:
subroutine mossbauer
! !USES:
use modmain
use modmpi
use modtest
! !DESCRIPTION:
!   Computes the contact charge density and magnetic hyperfine field for each
!   atom and outputs the data to the file {\tt MOSSBAUER.OUT}.
!   See S. Bl\"{u}gel, H. Akai, R. Zeller, and P. H. Dederichs,
!   {\it Phys. Rev. B} {\bf 35}, 3271 (1987).
!
! !REVISION HISTORY:
!   Created May 2004 (JKD)
!   Contact hyperfine field evaluated at the nuclear radius rather than averaged
!   over the Thomson sphere, June 2019 (JKD)
!   Added spin and orbital dipole terms, July 2019 (JKD)
!EOP
!BOC
implicit none
! local variables
integer idm,is,ia,ias
integer nr,nri,nrn,nrt
real(8) r0,rn,rna,rt,rta
real(8) mn(3),mt(3),bn(3),bt(3)
real(8) cb,t1
! allocatable arrays
real(8), allocatable :: fr(:)
! spin dipole field prefactor
cb=gfacte/(4.d0*solsc)
! initialise universal variables
call init0
call init1
! read density and potentials from file
call readstate
! generate the core wavefunctions and densities
call gencore
! find the new linearisation energies
call linengy
! generate the APW and local-orbital radial functions and integrals
call genapwlofr
! read the eigenvalues and occupation numbers from file
call readevalsv
call readoccsv
! calculate the density
call rhomag
! calculate the dipole magnetic field if required
if (tbdip) call bdipole
! allocate local arrays
allocate(fr(nrmtmax))
if (mp_mpi) then
  open(50,file='MOSSBAUER.OUT',form='FORMATTED')
end if
! loop over species
do is=1,nspecies
  nr=nrmt(is)
  nri=nrmti(is)
  nrn=nrnucl(is)
  nrt=nrtmsn(is)
! loop over atoms
  do ia=1,natoms(is)
    ias=idxas(ia,is)
    if (mp_mpi) then
      write(50,*)
      write(50,*)
      write(50,'("Species : ",I4," (",A,"), atom : ",I4)') is,trim(spsymb(is)),&
       ia
      write(50,*)
      write(50,'(" approximate nuclear radius : ",G18.10)') rnucl(is)
      write(50,'(" number of mesh points to nuclear radius : ",I6)') nrn
      write(50,'(" Thomson radius : ",G18.10)') rtmsn(is)
      write(50,'(" number of mesh points to Thomson radius : ",I6)') nrt
    end if
!--------------------------------!
!     contact charge density     !
!--------------------------------!
! extract the l = m = 0 component of the muffin-tin density
    call rfmtlm(1,nr,nri,rhomt(:,ias),fr)
! density at nuclear center
    r0=fr(1)*y00
! density at nuclear surface
    rn=fr(nrn)*y00
! average density in nuclear volume
    t1=dot_product(wr2mt(1:nrn,is),fr(1:nrn))
    rna=fourpi*y00*t1/volnucl(is)
! density at Thomson radius
    rt=fr(nrt)*y00
! average density in Thomson volume
    t1=dot_product(wr2mt(1:nrt,is),fr(1:nrt))
    rta=fourpi*y00*t1/voltmsn(is)
    if (mp_mpi) then
      write(50,*)
      write(50,'(" Contact density :")')
      write(50,'("  at nuclear center         : ",G18.10)') r0
      write(50,'("  at nuclear surface        : ",G18.10)') rn
      write(50,'("  average in nuclear volume : ",G18.10)') rna
      write(50,'("  at Thomson radius         : ",G18.10)') rt
      write(50,'("  average in Thomson volume : ",G18.10)') rta
    end if
!----------------------------------!
!     magnetic hyperfine field     !
!----------------------------------!
    if (spinpol) then
! contact term
      do idm=1,ndmag
! extract the l = m = 0 component of the muffin-tin magnetisation
        call rfmtlm(1,nr,nri,magmt(:,ias,idm),fr)
! average magnetisation in nuclear volume
        t1=dot_product(wr2mt(1:nrn,is),fr(1:nrn))
        mn(idm)=fourpi*y00*t1/volnucl(is)
! average in Thomson volume
        t1=dot_product(wr2mt(1:nrt,is),fr(1:nrt))
        mt(idm)=fourpi*y00*t1/voltmsn(is)
      end do
      t1=8.d0*pi*cb/(3.d0*solsc)
      bn(1:ndmag)=t1*mn(1:ndmag)
      bt(1:ndmag)=t1*mt(1:ndmag)
      if (mp_mpi) then
        write(50,*)
        write(50,'(" Contact average in nuclear volume :")')
        write(50,'("  moment (mu_B) : ",3G18.10)') mn(1:ndmag)
        write(50,'("  magnetic field : ",3G18.10)') bn(1:ndmag)
        write(50,'("   tesla : ",3G18.10)') b_si*bn(1:ndmag)
        write(50,*)
        write(50,'(" Contact average in Thomson volume :")')
        write(50,'("  moment (mu_B) : ",3G18.10)') mt(1:ndmag)
        write(50,'("  magnetic field : ",3G18.10)') bt(1:ndmag)
        write(50,'("   tesla : ",3G18.10)') b_si*bt(1:ndmag)
      end if
! spin and orbital dipole term
      if (tbdip) then
! extract the l = m = 0 component of the dipole field
        do idm=1,3
          call rfmtlm(1,nr,nri,bdmt(:,ias,idm),fr)
! average dipole field in nuclear volume
          t1=dot_product(wr2mt(1:nrn,is),fr(1:nrn))
          bn(idm)=fourpi*y00*t1/(volnucl(is)*solsc)
! average in Thomson volume
          t1=dot_product(wr2mt(1:nrt,is),fr(1:nrt))
          bt(idm)=fourpi*y00*t1/(voltmsn(is)*solsc)
        end do
        if (mp_mpi) then
          write(50,*)
          write(50,'(" Average dipole field in nuclear volume :")')
          if (tjr) then
            write(50,'("  spin and orbital : ",3G18.10)') bn
          else
            write(50,'("  spin : ",3G18.10)') bn
          end if
          write(50,'("   tesla : ",3G18.10)') b_si*bn
          write(50,*)
          write(50,'(" Average dipole field in Thomson volume :")')
          if (tjr) then
            write(50,'("  spin and orbital : ",3G18.10)') bt
          else
            write(50,'("  spin : ",3G18.10)') bt
          end if
          write(50,'("   tesla : ",3G18.10)') b_si*bt
        end if
! write to test file if required
        call writetest(110,'hyperfine field',nv=3,tol=1.d-4,rva=bn)
      end if
    end if
  end do
end do
if (mp_mpi) then
  if (spinpol.and.tbdip) then
    write(50,*)
    write(50,'("Note that the contact term is implicitly included in the &
     &spin dipole field")')
    write(50,'(" but may not match exactly with the directly &
     &calculated value.")')
  end if
  close(50)
  write(*,*)
  write(*,'("Info(mossbauer):")')
  write(*,'(" Mossbauer parameters written to MOSSBAUER.OUT")')
end if
deallocate(fr)
end subroutine
!EOC

