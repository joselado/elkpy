
! Copyright (C) 2002-2008 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: genlmirep
! !INTERFACE:
subroutine genlmirep(elm,ulm)
! !USES:
use modmain
! !INPUT/OUTPUT PARAMETERS:
!   elm : eigenvalues of the symmetrised pseudorandom matrix H
!         (out,real(lmmaxdb,natmtot))
!   ulm : unitary matrix which will transform a density matrix from the (l,m)
!         basis to the irreducible basis (in,complex(lmmaxdb,lmmaxdb,natmtot))
! !DESCRIPTION:
!   First generates a symmetric, pseudorandom matrix $H$ which is then
!   symmetrised by applying all the group symmetries $\{S_i^{\alpha}\}$ at
!   atomic site $\alpha$ as
!   $$ \tilde{H}=\sum_i S_i^{\alpha} H {S_i^{\alpha}}^{\dag}. $$
!   This matrix is then diagonalised. By Schur's second lemma the eigenvalues
!   have degeneracies of the same dimension as an irreducible representation
!   (IR) of the symmetry group, and can be used to identify the IR. These
!   eigenvalues are returned in the matrix {\tt elm}. The conjugate transpose of
!   the eigenvector array forms a unitary matrix $U$ which can be applied
!   directly to a density matrix $\gamma$ in the $Y_{lm}$ basis as
!   $U\gamma U^{\dag}$. This will transform $\gamma$ into into the IR basis. The
!   matrix $U$ is returned in the array {\tt ulm}. See the routines
!   {\tt bandstr} and {\tt dos}.
!
! !REVISION HISTORY:
!   Created August 2002 (JKD)
!EOP
!BOC
implicit none
! arguments
real(8), intent(out) :: elm(lmmaxdb,natmtot)
complex(8), intent(out) :: ulm(lmmaxdb,lmmaxdb,natmtot)
! local variables
integer isym,lspl,ias
integer i,j,l,lm,nm,p,info
! automatic arrays
real(8) rwork(3*lmmaxdb)
complex(8) dlat(lmmaxdb,lmmaxdb,nsymlat)
complex(8) a(lmmaxdb,lmmaxdb),b(lmmaxdb,lmmaxdb)
complex(8) h(lmmaxdb,lmmaxdb),work(2*lmmaxdb)
! construct Yₗₘ rotation matrix for each lattice symmetry
a(:,:)=0.d0
do i=1,lmmaxdb
  a(i,i)=1.d0
end do
do isym=1,nsymlat
  call rotzflm(symlatc(:,:,isym),0,lmaxdb,lmmaxdb,lmmaxdb,lmmaxdb,a, &
   dlat(:,:,isym))
end do
! set up pseudorandom symmetric matrix H
h(:,:)=0.d0
p=0
do l=0,lmaxdb
  nm=2*l
  lm=l**2+1
  do i=lm,lm+nm
    do j=i,lm+nm
      p=p+1
      h(i,j)=p
      h(j,i)=p
    end do
  end do
end do
! loop over species and atoms
do ias=1,natmtot
! symmetrise H with site symmetries
  b(:,:)=0.d0
  do isym=1,nsymsite(ias)
! spatial rotation element in lattice point group
    lspl=lsplsyms(isym,ias)
! apply lattice symmetry as S H S†
    call zgemm('N','N',lmmaxdb,lmmaxdb,lmmaxdb,zone,dlat(:,:,lspl),lmmaxdb,h, &
     lmmaxdb,zzero,a,lmmaxdb)
    call zgemm('N','C',lmmaxdb,lmmaxdb,lmmaxdb,zone,a,lmmaxdb,dlat(:,:,lspl), &
     lmmaxdb,zone,b,lmmaxdb)
  end do
! block diagonalise symmetrised H
  do l=0,lmaxdb
    nm=2*l+1
    lm=l**2+1
    call zheev('V','U',nm,b(lm,lm),lmmaxdb,elm(lm,ias),work,2*lmmaxdb,rwork, &
     info)
  end do
! the unitary matrix U is the conjugate transpose of the eigenvector array since
! this will act on the density matrix as U γ U†
  do i=1,lmmaxdb
    do j=1,lmmaxdb
      ulm(i,j,ias)=conjg(b(j,i))
    end do
  end do
end do
end subroutine
!EOC

