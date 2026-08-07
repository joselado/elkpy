
! Copyright (C) 2007 F. Bultmark, F. Cricchio, L. Nordstrom and J. K. Dewhurst.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine symdmat(lmax,ld,dmat)
use modmain
implicit none
! arguments
integer, intent(in) :: lmax,ld
complex(8), intent(inout) :: dmat(ld,nspinor,ld,nspinor,natmtot)
! local variables
integer is,ia,ja,ias,jas
integer isym,lspl,lspn,lmmax
real(8) t1
! allocatable arrays
complex(8), allocatable :: dm(:,:,:,:,:)
lmmax=(lmax+1)**2
! allocate local arrays
allocate(dm(ld,nspinor,ld,nspinor,natmmax))
t1=1.d0/dble(nsymcrys)
do is=1,nspecies
! make copy of the density matrices
  do ia=1,natoms(is)
    ias=idxas(ia,is)
    dm(1:lmmax,:,1:lmmax,:,ia)=dmat(1:lmmax,:,1:lmmax,:,ias)
  end do
! loop over atoms
  do ia=1,natoms(is)
! only symmetrise first equivalent atom (rotate into others)
    if (.not.tfeqat(ia,is)) cycle
    ias=idxas(ia,is)
    dmat(:,:,:,:,ias)=0.d0
    do isym=1,nsymcrys
      lspl=lsplsymc(isym)
      lspn=lspnsymc(isym)
! equivalent atom index (symmetry rotates atom ja into atom ia)
      ja=ieqatom(ia,is,isym)
      call rotdmat(symlatc(:,:,lspl),symlatc(:,:,lspn),lmax,nspinor,ld, &
       dm(:,:,:,:,ja),dmat(:,:,:,:,ias))
! end loop over crystal symmetries
    end do
! normalise
    dmat(:,:,:,:,ias)=t1*dmat(:,:,:,:,ias)
! rotate into equivalent atoms
    do ja=1,natoms(is)
      if (eqatoms(ia,ja,is).and.(ia /= ja)) then
        isym=findloc(ieqatom(ia,is,1:nsymcrys),ja,1)
        jas=idxas(ja,is)
! inverse symmetry (which rotates atom ia into atom ja)
        lspl=isymlat(lsplsymc(isym))
        lspn=isymlat(lspnsymc(isym))
        dmat(:,:,:,:,jas)=0.d0
        call rotdmat(symlatc(:,:,lspl),symlatc(:,:,lspn),lmax,nspinor,ld, &
         dmat(:,:,:,:,ias),dmat(:,:,:,:,jas))
      end if
    end do
! end loop over atoms and species
  end do
end do
deallocate(dm)
end subroutine

