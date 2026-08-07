
! Copyright (C) 2013 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine genstrain
use modmain
implicit none
! local variables
integer i,j,k
real(8) a(3,3),b(3,3),t1
! set first strain equal to isotropic scaling
t1=norm2(avec(1:3,1:3))
strain(1:3,1:3,1)=avec(1:3,1:3)/t1
nstrain=1
do i=1,3
  do j=1,3
! set strain tensor in lattice coordinates to delta_ij
    a(1:3,1:3)=0.d0
    a(i,j)=1.d0
! symmetrise strain tensor
    call symmat(a)
! convert to mixed Cartesian-lattice coordinates
    call r3mtm(ainv,a,b)
! orthogonalise strain tensor to previous tensors
    do k=1,nstrain
      t1=sum(b(1:3,1:3)*strain(1:3,1:3,k))
      b(1:3,1:3)=b(1:3,1:3)-t1*strain(1:3,1:3,k)
    end do
! compute the norm
    t1=norm2(b(1:3,1:3))
    if (t1 < epslat) cycle
! normalise tensor and store in global array
    nstrain=nstrain+1
    strain(1:3,1:3,nstrain)=b(1:3,1:3)/t1
  end do
end do
! zero small components
do k=1,nstrain
  do i=1,3
    do j=1,3
      if (abs(strain(i,j,k)) < epslat) strain(i,j,k)=0.d0
    end do
  end do
end do
! lattice optimisation case
if ((task == 2).or.(task == 3)) then
  if (latvopt == 2) then
! remove isotropic scaling when latvopt = 2
    strain(1:3,1:3,1)=strain(1:3,1:3,nstrain)
    nstrain=nstrain-1
  else if (latvopt < 0) then
! optimise over particular strain when latvopt < 0
    i=abs(latvopt)
    if (i > nstrain) then
      write(*,*)
      write(*,'("Error(genstrain): |latvopt| > nstrain :",2(X,I0))') i,nstrain
      write(*,*)
      stop
    end if
    strain(1:3,1:3,1)=strain(1:3,1:3,i)
    nstrain=1
  end if
end if
end subroutine

