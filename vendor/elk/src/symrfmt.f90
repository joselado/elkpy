
! Copyright (C) 2018 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine symrfmt(nrmt_,nrmti_,npmt_,ld,rfmt)
use modmain
implicit none
! arguments
integer, intent(in) :: nrmt_(nspecies),nrmti_(nspecies),npmt_(nspecies),ld
real(8), intent(inout) :: rfmt(ld,natmtot)
! local variables
integer is,ia,ja,ias,jas
integer nr,nri,np,isym,lspl
real(8) t0
! allocatable arrays
real(8), allocatable :: rfmt1(:,:),rfmt2(:)
allocate(rfmt1(ld,natmmax),rfmt2(ld))
t0=1.d0/dble(nsymcrys)
do is=1,nspecies
  nr=nrmt_(is)
  nri=nrmti_(is)
  np=npmt_(is)
! make a copy of the input function
  do ia=1,natoms(is)
    ias=idxas(ia,is)
    rfmt1(1:np,ia)=rfmt(1:np,ias)
  end do
! loop over atoms
  do ia=1,natoms(is)
! only symmetrise first equivalent atom (rotate into others)
    if (.not.tfeqat(ia,is)) cycle
    ias=idxas(ia,is)
    rfmt(1:np,ias)=0.d0
! loop over crystal symmetries
    do isym=1,nsymcrys
! index to spatial rotation lattice symmetry
      lspl=lsplsymc(isym)
! equivalent atom index (symmetry rotates atom ja into atom ia)
      ja=ieqatom(ia,is,isym)
! apply the rotation to the muffin-tin function
      call rotrfmt(symlatc(:,:,lspl),nr,nri,rfmt1(:,ja),rfmt2)
! accumulate in original function array
      rfmt(1:np,ias)=rfmt(1:np,ias)+rfmt2(1:np)
    end do
! normalise
    rfmt(1:np,ias)=t0*rfmt(1:np,ias)
! rotate into equivalent atoms
    do ja=1,natoms(is)
      if (eqatoms(ia,ja,is).and.(ia /= ja)) then
        isym=findloc(ieqatom(ia,is,1:nsymcrys),ja,1)
        jas=idxas(ja,is)
! inverse symmetry (which rotates atom ia into atom ja)
        lspl=isymlat(lsplsymc(isym))
! rotate symmetrised function into equivalent muffin-tin
        call rotrfmt(symlatc(:,:,lspl),nr,nri,rfmt(:,ias),rfmt(:,jas))
      end if
    end do
  end do
end do
deallocate(rfmt1,rfmt2)
end subroutine

