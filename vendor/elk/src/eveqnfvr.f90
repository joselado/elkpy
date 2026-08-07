
! Copyright (C) 2011 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: eveqnfvr
! !INTERFACE:
subroutine eveqnfvr(nmatp,ngp,vpc,h_,o_,evalfv,evecfv)
! !USES:
use modmain
use, intrinsic :: iso_c_binding
! !INPUT/OUTPUT PARAMETERS:
!   nmatp  : order of overlap and Hamiltonian matrices (in,integer)
!   ngp    : number of G+p-vectors (in,integer)
!   vpc    : p-vector in Cartesian coordinates (in,real(3))
!   h,o    : Hamiltonian and overlap matrices in upper triangular form
!            (in,complex(*))
!   evalfv : first-variational eigenvalues (out,real(nstfv))
!   evecfv : first-variational eigenvectors (out,complex(nmatmax,nstfv))
! !DESCRIPTION:
!   This routine solves the first-variational eigenvalue equation for the
!   special case when inversion symmetry is present. In this case the
!   Hamiltonian and overlap matrices can be made real by using appropriate
!   linear combinations of the local-orbitals for atoms related by inversion
!   symmetry. These are derived from the effect of parity and complex
!   conjugation on the spherical harmonics: $P Y_{lm}=(-1)^l Y_{lm}$ and
!   $(Y_{lm})^*=(-1)^m Y_{l-m}$.
!
! !REVISION HISTORY:
!   Created May 2011 (JKD)
!EOP
!BOC
implicit none
! arguments
integer, intent(in) :: nmatp,ngp
real(8), intent(in) :: vpc(3)
real(8), target :: h_(*),o_(*)
real(8), intent(out) :: evalfv(nstfv)
complex(8), intent(out) :: evecfv(nmatmax,nstfv)
! local variables
integer is,ia,ja,jas,ilo
integer n2,i,j,k,l,m
integer i1,i2,j1,j2
integer k1,k2,k3,k4
integer l1,l2,m1,m2
real(8) v(3),s1,t1,t2,t3,t4
complex(8) z1
! automatic arrays
logical tr(nlotot),tp(nlotot)
integer idx(nlotot),s(nlotot)
complex(8) zp(nlotot)
! allocatable arrays and pointers
real(8), allocatable :: rh(:)
real(8), pointer, contiguous :: ro(:),rv(:,:)
complex(8), pointer, contiguous :: h(:),o(:)
! reuse already allocated memory
n2=nmatp**2
call c_f_pointer(c_loc(h_),h,shape=[n2])
call c_f_pointer(c_loc(o_),o,shape=[n2])
call c_f_pointer(c_loc(h_),ro,shape=[n2])
call c_f_pointer(c_loc(h_(n2+1)),rv,shape=[nmatp,nstfv])
tp(1:nlotot)=.false.
i=0
do is=1,nspecies
  do ia=1,natoms(is)
! symmetry equivalent atom, mapped with inversion
    ja=ieqatom(ia,is,2)
    jas=idxas(ja,is)
! residual phase factor
    v(1:3)=atposc(1:3,ia,is)+atposc(1:3,ja,is)
    t1=0.5d0*(vpc(1)*v(1)+vpc(2)*v(2)+vpc(3)*v(3))
    z1=cmplx(cos(t1),sin(t1),8)
    do ilo=1,nlorb(is)
      l=lorbl(ilo,is)
      do m=-l,l
        i=i+1
! index to conjugate local-orbital in symmetry equivalent atom
        idx(i)=idxlo(l*(l+1)-m+1,ilo,jas)
        if (ia /= ja) then
! sign of parity and conjugation operators
          if (mod(l+m,2) == 0) then
            s(i)=1
          else
            s(i)=-1
          end if
          if (ia < ja) then
! if ia < ja use the real part of the sum of matrix elements
            tr(i)=.true.
          else if (ia > ja) then
! if ia > ja use the imaginary part of the difference of matrix elements
            s(i)=-s(i)
            tr(i)=.false.
          end if
        else
! if ia = ja then use real function when l even and imaginary when l is odd
          if (mod(m,2) == 0) then
            s(i)=1
          else
            s(i)=-1
          end if
! new function should be real if symmetric or imaginary if antisymmetric
          if (mod(l,2) == 0) then
! l even
            if (m >= 0) then
              tr(i)=.true.
            else
              s(i)=-s(i)
              tr(i)=.false.
            end if
          else
! l odd
            if (m >= 0) then
              tr(i)=.false.
            else
              s(i)=-s(i)
              tr(i)=.true.
            end if
          end if
        end if
! phase factors if required
        if (abs(t1) > 1.d-8) then
          zp(i)=z1
          tp(i)=.true.
        end if
      end do
    end do
  end do
end do
!---------------------------------!
!     real Hamiltonian matrix     !
!---------------------------------!
allocate(rh(n2))
! ⟨APW|H|APW⟩
do j=1,ngp
  k=(j-1)*nmatp+1
  rh(k:k+j-1)=dble(h(k:k+j-1))
end do
! ⟨APW|H|lo⟩
do m1=1,nlotot
  j1=(ngp+m1-1)*nmatp
  j2=(ngp+idx(m1)-1)*nmatp
  z1=zp(m1); s1=s(m1)
  if (tp(m1)) then
    if (tr(m1)) then
      rh(j1+1:j1+ngp)=dble(h(j1+1:j1+ngp)*z1)+s1*dble(h(j2+1:j2+ngp)*z1)
    else
      rh(j1+1:j1+ngp)=aimag(h(j1+1:j1+ngp)*z1)+s1*aimag(h(j2+1:j2+ngp)*z1)
    end if
  else
    if (tr(m1)) then
      rh(j1+1:j1+ngp)=dble(h(j1+1:j1+ngp))+s1*dble(h(j2+1:j2+ngp))
    else
      rh(j1+1:j1+ngp)=aimag(h(j1+1:j1+ngp))+s1*aimag(h(j2+1:j2+ngp))
    end if
  end if
end do
! ⟨lo|H|lo⟩
do m1=1,nlotot
  m2=idx(m1)
  do l1=1,m1
    l2=idx(l1)
    k1=map(l1,m1); k2=map(l1,m2); k3=map(l2,m1); k4=map(l2,m2)
    if ((tr(l1).and.tr(m1)).or.((.not.tr(l1)).and.(.not.tr(m1)))) then
      rh(k1)=h(k1)%re+s(m1)*h(k2)%re+s(l1)*(h(k3)%re+s(m1)*h(k4)%re)
    else
      t2=h(k2)%im
      if (l1 > m2) t2=-t2
      t3=h(k3)%im
      if (l2 > m1) t3=-t3
      t4=h(k4)%im
      if (l2 > m2) t4=-t4
      rh(k1)=h(k1)%im+s(m1)*t2+s(l1)*(t3+s(m1)*t4)
      if (.not.tr(l1)) rh(k1)=-rh(k1)
    end if
  end do
end do
!-----------------------------!
!     real overlap matrix     !
!-----------------------------!
! ⟨APW|O|APW⟩
do j=1,ngp
  k=(j-1)*nmatp+1
  ro(k:k+j-1)=dble(o(k:k+j-1))
end do
! ⟨APW|O|lo⟩
do m1=1,nlotot
  j1=(ngp+m1-1)*nmatp
  j2=(ngp+idx(m1)-1)*nmatp
  z1=zp(m1); s1=s(m1)
  if (tp(m1)) then
    if (tr(m1)) then
      ro(j1+1:j1+ngp)=dble(o(j1+1:j1+ngp)*z1)+s1*dble(o(j2+1:j2+ngp)*z1)
    else
      ro(j1+1:j1+ngp)=aimag(o(j1+1:j1+ngp)*z1)+s1*aimag(o(j2+1:j2+ngp)*z1)
    end if
  else
    if (tr(m1)) then
      ro(j1+1:j1+ngp)=dble(o(j1+1:j1+ngp))+s1*dble(o(j2+1:j2+ngp))
    else
      ro(j1+1:j1+ngp)=aimag(o(j1+1:j1+ngp))+s1*aimag(o(j2+1:j2+ngp))
    end if
  end if
end do
! ⟨lo|O|lo⟩
do m1=1,nlotot
  m2=idx(m1)
  do l1=1,m1
    l2=idx(l1)
    k1=map(l1,m1); k2=map(l1,m2); k3=map(l2,m1); k4=map(l2,m2)
    if ((tr(l1).and.tr(m1)).or.((.not.tr(l1)).and.(.not.tr(m1)))) then
      ro(k1)=o(k1)%re+s(m1)*o(k2)%re+s(l1)*(o(k3)%re+s(m1)*o(k4)%re)
    else
      t2=o(k2)%im
      if (l1 > m2) t2=-t2
      t3=o(k3)%im
      if (l2 > m1) t3=-t3
      t4=o(k4)%im
      if (l2 > m2) t4=-t4
      ro(k1)=o(k1)%im+s(m1)*t2+s(l1)*(t3+s(m1)*t4)
      if (.not.tr(l1)) ro(k1)=-ro(k1)
    end if
  end do
end do
! solve the generalised eigenvalue problem for real symmetric matrices
call dsygvxi(nmatp,nstfv,nmatp,rh,ro,evalfv,nmatp,rv,n2,o_)
! reconstruct the complex eigenvectors
do j=1,nstfv
  evecfv(1:ngp,j)=rv(1:ngp,j)
  evecfv(ngp+1:nmatp,j)=0.d0
  do l1=1,nlotot
    i1=ngp+l1
    i2=ngp+idx(l1)
    t1=rv(i1,j)
    if (tr(l1)) then
      evecfv(i1,j)=evecfv(i1,j)+t1
      evecfv(i2,j)=evecfv(i2,j)+s(l1)*t1
    else
      evecfv(i1,j)%im=evecfv(i1,j)%im-t1
      evecfv(i2,j)%im=evecfv(i2,j)%im-s(l1)*t1
    end if
  end do
  do l1=1,nlotot
    if (tp(l1)) then
      i1=ngp+l1
      evecfv(i1,j)=evecfv(i1,j)*zp(l1)
    end if
  end do
end do
deallocate(rh)

contains

elemental integer function map(i,j)
implicit none
! arguments
integer, intent(in) :: i,j
! map from local-orbital indices to position in matrix
map=merge(ngp+i+(ngp+j-1)*nmatp,ngp+j+(ngp+i-1)*nmatp,i <= j)
end function

end subroutine
!EOC

