
! Copyright (C) 2016 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine strainabg
use modmain
use modgw
use modvars
implicit none
integer is,ia,ig
real(8) ta(3,3),tb(3,3),vc(3)
! compute the strained lattice vectors
if ((istrain >= 1).and.(istrain <= nstrain)) then
  avec(1:3,1:3)=avec0(1:3,1:3)+deltast*strain(1:3,1:3,istrain)
else if (tavref) then
  avec(1:3,1:3)=avec0(1:3,1:3)
else
  return
end if
! generate the strained reciprocal lattice vectors and unit cell volume
call reciplat(avec,bvec,omega,omegabz)
! determine the transformation matrix to the strained vectors
call r3mm(avec,ainv,ta)
call r3mm(bvec,binv,tb)
! recalculate all required variables which depend on avec
call r3minv(avec,ainv)
call r3minv(bvec,binv)
call r3mv(bvec,vqlss,vqcss)
do is=1,nspecies
  do ia=1,natoms(is)
    call r3mv(avec,atposl(:,ia,is),atposc(:,ia,is))
  end do
end do
call r3mv(bvec,vecql,vecqc)
call r3mv(ainv,efieldc,efieldl)
call r3mv(ainv,afieldc,afieldl)
call symmetry
! apply the transformation matrix to the G-vectors
do ig=1,ngtot
  vc(1:3)=vgc(1:3,ig)
  call r3mv(tb,vc,vgc(:,ig))
  gc(ig)=sqrt(vgc(1,ig)**2+vgc(2,ig)**2+vgc(3,ig)**2)
end do
end subroutine

