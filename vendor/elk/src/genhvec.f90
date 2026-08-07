
! Copyright (C) 2010 Alexey I. Baranov.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine genhvec
use modmain
use modpw
implicit none
! local variables
logical lsym(48)
integer ih,jh,kh,lh,k
integer i1,i2,i3,iv(3)
integer nsym,isym,sym(3,3,48)
real(8) v1(3),v2(3),v3(3)
! allocatable arrays
integer, allocatable :: idx(:),ivh_(:,:)
real(8), allocatable :: vhc_(:,:),hc_(:)
! find the H-vector grid sizes
call gridsize(avec,hmaxvr,npfftg,ngridh,nhtot,inthv)
! allocate global H-vector arrays
if (allocated(ivh)) deallocate(ivh)
allocate(ivh(3,nhtot))
if (allocated(mulh)) deallocate(mulh)
allocate(mulh(nhtot))
if (allocated(vhc)) deallocate(vhc)
allocate(vhc(3,nhtot))
if (allocated(hc)) deallocate(hc)
allocate(hc(nhtot))
! allocate local arrays
allocate(idx(nhtot),ivh_(3,nhtot))
allocate(vhc_(3,nhtot),hc_(nhtot))
ih=0
do i1=inthv(1,1),inthv(2,1)
  v1(1:3)=dble(i1)*bvec(1:3,1)
  do i2=inthv(1,2),inthv(2,2)
    v2(1:3)=v1(1:3)+dble(i2)*bvec(1:3,2)
    do i3=inthv(1,3),inthv(2,3)
      v3(1:3)=v2(1:3)+dble(i3)*bvec(1:3,3)
      ih=ih+1
! map from H-vector to (i1,i2,i3) index
      ivh_(1,ih)=i1; ivh_(2,ih)=i2; ivh_(3,ih)=i3
! H-vector in Cartesian coordinates
      vhc_(1:3,ih)=v3(1:3)
! length of each H-vector
      hc_(ih)=sqrt(v3(1)**2+v3(2)**2+v3(3)**2)
    end do
  end do
end do
! sort by vector length
call sortidx(nhtot,hc_,idx)
! reorder arrays
ivh(1:3,1:nhtot)=ivh_(1:3,idx(1:nhtot))
hc(1:nhtot)=hc_(idx(1:nhtot))
vhc(1:3,1:nhtot)=vhc_(1:3,idx(1:nhtot))
! find the number of vectors with H < hmaxvr
nhvec=1
do ih=2,nhtot
  if (hc(ih) > hmaxvr) then
    nhvec=ih-1
    exit
  end if
end do
! find the subgroup of symmorphic, non-magnetic symmetries
lsym(:)=.false.
do isym=1,nsymcrys
  if (tv0symc(isym).and.(lspnsymc(isym) == 1)) lsym(lsplsymc(isym))=.true.
end do
nsym=0
do isym=1,nsymlat
  if (lsym(isym)) then
    nsym=nsym+1
    sym(:,:,nsym)=symlat(:,:,isym)
  end if
end do
if (reduceh) then
! find the subgroup of symmorphic, non-magnetic symmetries
  lsym(:)=.false.
  do isym=1,nsymcrys
    if (tv0symc(isym).and.(lspnsymc(isym) == 1)) lsym(lsplsymc(isym))=.true.
  end do
  nsym=0
  do isym=1,nsymlat
    if (lsym(isym)) then
      nsym=nsym+1
      sym(:,:,nsym)=symlat(:,:,isym)
    end if
  end do
else
! use only the identity element if no reduction is required
  nsym=1
end if
! reduce the H-vector set with the symmetries if required
if (nsym > 1) then
  ivh_(1:3,1:nhvec)=ivh(1:3,1:nhvec)
  hc_(1:nhvec)=hc(1:nhvec)
  vhc_(1:3,1:nhvec)=vhc(1:3,1:nhvec)
  kh=0
  lh=nhvec
  do ih=1,nhvec
    do isym=1,nsym
      call i3mtv(sym(:,:,isym),ivh_(:,ih),iv(:))
      do jh=1,kh
        k=abs(ivh(1,jh)-iv(1))+abs(ivh(2,jh)-iv(2))+abs(ivh(3,jh)-iv(3))
        if (k == 0) then
          ivh(1:3,lh)=ivh_(1:3,ih)
          hc(lh)=hc_(ih)
          vhc(1:3,lh)=vhc_(1:3,ih)
          lh=lh-1
          mulh(jh)=mulh(jh)+1
          goto 10
        end if
      end do
    end do
    kh=kh+1
    ivh(1:3,kh)=ivh_(1:3,ih)
    hc(kh)=hc_(ih)
    vhc(1:3,kh)=vhc_(1:3,ih)
    mulh(kh)=1
10 continue
  end do
  nhvec=kh
else
  mulh(:)=1
end if
deallocate(idx,ivh_,vhc_,hc_)
end subroutine

