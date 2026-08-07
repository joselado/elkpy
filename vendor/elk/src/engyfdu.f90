
! Copyright (C) 2008 F. Bultmark, F. Cricchio, L. Nordstrom and J. K. Dewhurst.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: engyfdu
! !INTERFACE:
subroutine engyfdu(idu)
! !USES:
use modmain
use moddftu
use modmpi
! !INPUT/OUTPUT PARAMETERS:
!   idu : DFT+U entry (in,integer)
! !DESCRIPTION:
!   Calculates the energies of radial functions to be used to calculate the
!   Slater integrals. By convention those energies are chosen to be the ones at
!   the center of the band.
!
! !REVISION HISTORY:
!   Created April 2008 (F. Cricchio)
!EOP
!BOC
implicit none
integer, intent(in) :: idu
! local variables
integer is,ia,ja,ias,jas
integer nr,nri,nnf,l
logical fnd
! automatic arrays
real(8) vr(nrmtmax)
nnf=0
is=isldu(1,idu)
l=isldu(2,idu)
nr=nrmt(is)
nri=nrmti(is)
do ia=1,natoms(is)
! perform calculation for only the first equivalent atom
  if (.not.tfeqat(ia,is)) cycle
  ias=idxas(ia,is)
  call rfmtlm(1,nr,nri,vsmt(:,ias),vr)
  vr(1:nr)=vr(1:nr)*y00
! find the center of the band starting from -0.5 Ha
  efdu(l,ias)=-0.5d0
  call findband(solsc,l,nrmt(is),rsp(1,is),vr,epsband,demaxbnd,efdu(l,ias),fnd)
  if (.not.fnd) nnf=nnf+1
! copy to equivalent atoms
  do ja=1,natoms(is)
    if (eqatoms(ia,ja,is).and.(ia /= ja)) then
      jas=idxas(ja,is)
      efdu(l,jas)=efdu(l,ias)
    end if
  end do
! end loops over atoms and species
end do
if (mp_mpi.and.(nnf > 0)) then
  write(*,*)
  write(*,'("Warning(engyfdu): could not find ",I0," energies")') nnf
end if
end subroutine
!EOC

