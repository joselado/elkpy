
! Copyright (C) 2002-2012 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU Lesser General Public
! License. See the file COPYING for license details.

!BOP
! !ROUTINE: gengvec
! !INTERFACE:
subroutine gengvec
! !USES:
use modmain
! !DESCRIPTION:
!   Generates a set of ${\bf G}$-vectors used for the Fourier transform of the
!   charge density and potential and sorts them according to length. Integers
!   corresponding to the vectors in lattice coordinates are stored, as well as
!   the map from these integer coordinates to the ${\bf G}$-vector index. A map
!   from the ${\bf G}$-vector set to the standard FFT array structure is also
!   generated. Finally, the number of ${\bf G}$-vectors with magnitude less than
!   {\tt gmaxvr} is determined.
!
! !REVISION HISTORY:
!   Created October 2002 (JKD)
!EOP
!BOC
implicit none
! local variables
integer ig,n1,i1,i2,i3,j1,j2,j3
real(8) v1(3),v2(3),v3(3)
! allocatable arrays
integer, allocatable :: idx(:),ivg_(:,:)
real(8), allocatable :: vgc_(:,:),gc_(:)
if (gmaxvr < 0.d0) gmaxvr=abs(gmaxvr)*gkmax
! ensure |G| cut-off is at least twice |G+k| cut-off
gmaxvr=max(gmaxvr,2.d0*gkmax+epslat)
! find the G-vector grid sizes
call gridsize(avec,gmaxvr,npfftg,ngridg,ngtot,intgv)
! number of complex FFT elements for real-complex transforms
n1=ngridg(1)/2+1
nfgrz=n1*ngridg(2)*ngridg(3)
! allocate local arrays
allocate(idx(ngtot),ivg_(3,ngtot))
allocate(vgc_(3,ngtot),gc_(ngtot))
ig=0
do i1=intgv(1,1),intgv(2,1)
  v1(1:3)=dble(i1)*bvec(1:3,1)
  do i2=intgv(1,2),intgv(2,2)
    v2(1:3)=v1(1:3)+dble(i2)*bvec(1:3,2)
    do i3=intgv(1,3),intgv(2,3)
      v3(1:3)=v2(1:3)+dble(i3)*bvec(1:3,3)
      ig=ig+1
! map from G-vector to (i1,i2,i3) index
      ivg_(1,ig)=i1; ivg_(2,ig)=i2; ivg_(3,ig)=i3
! G-vector in Cartesian coordinates
      vgc_(1:3,ig)=v3(1:3)
! length of each G-vector
      gc_(ig)=sqrt(v3(1)**2+v3(2)**2+v3(3)**2)
    end do
  end do
end do
! sort by vector length
call sortidx(ngtot,gc_,idx)
! allocate global G-vector arrays
if (allocated(ivg)) deallocate(ivg)
allocate(ivg(3,ngtot))
if (allocated(vgc)) deallocate(vgc)
allocate(vgc(3,ngtot))
if (allocated(gc)) deallocate(gc)
allocate(gc(ngtot))
! reorder arrays and store globally
ivg(1:3,1:ngtot)=ivg_(1:3,idx(1:ngtot))
vgc(1:3,1:ngtot)=vgc_(1:3,idx(1:ngtot))
gc(1:ngtot)=gc_(idx(1:ngtot))
deallocate(idx,ivg_,vgc_,gc_)
! find the number of vectors with |G| <= gmaxvr
ngvec=1
do ig=2,ngtot
  if (gc(ig) > gmaxvr) then
    ngvec=ig-1
    exit
  end if
end do
! generate index arrays
if (allocated(ivgig)) deallocate(ivgig)
allocate(ivgig(intgv(1,1):intgv(2,1), &
               intgv(1,2):intgv(2,2), &
               intgv(1,3):intgv(2,3)))
if (allocated(igfft)) deallocate(igfft)
allocate(igfft(ngtot))
if (allocated(igrzf)) deallocate(igrzf)
allocate(igrzf(nfgrz))
do ig=1,ngtot
  i1=ivg(1,ig); i2=ivg(2,ig); i3=ivg(3,ig)
! map from (i1,i2,i3) to G-vector index
  ivgig(i1,i2,i3)=ig
! Fourier transform index
  j1=merge(i1,ngridg(1)+i1,i1 >= 0)
  j2=merge(i2,ngridg(2)+i2,i2 >= 0)
  j3=merge(i3,ngridg(3)+i3,i3 >= 0)
  igfft(ig)=j3*ngridg(2)*ngridg(1)+j2*ngridg(1)+j1+1
! map from real-complex FFT index to G-vector index
  if (i1 >= 0) igrzf(j3*ngridg(2)*n1+j2*n1+j1+1)=ig
end do
end subroutine
!EOC

