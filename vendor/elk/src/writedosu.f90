
! Copyright (C) 2024 Wenhan Chen, J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine writedosu
use modmain
use modulr
implicit none
! local variables
integer nsk(3),ik0,iw
real(8) dw
! allocatable arrays
real(8), allocatable :: w(:),f(:,:),g(:)
! no k-point reduction
reducek0=reducek
reducek=0
! initialise global variables
call init0
call init1
call initulr
! read in the Fermi energy from STATE_ULR.OUT
call readstulr
! get the ULR eigenvalues from file
do ik0=1,nkpt0
  call getevalu(ik0)
end do
! subtract the Fermi energy
evalu(1:nstulr,1:nkpt0)=evalu(1:nstulr,1:nkpt0)-efermi
! generate frequency grid
allocate(w(nwplot))
dw=(wplot(2)-wplot(1))/dble(nwplot)
do iw=1,nwplot
  w(iw)=dw*dble(iw-1)+wplot(1)
end do
! number of subdivisions used for interpolation in the Brillouin zone
nsk(:)=max(ngrkf/ngridk(:),1)
! integrate over the Brillouin zone
allocate(f(nstulr,nkpt0),g(nwplot))
! normalise DOS to the unit cell
f(1:nstulr,1:nkpt0)=1.d0/dble(nkpa)
call brzint(nswplot,ngridk,nsk,ivkik,nwplot,wplot,nstulr,nstulr,evalu,f,g)
! write total DOS to file
open(50,file='TDOSULR.OUT',form='FORMATTED',action='WRITE')
do iw=1,nwplot
  write(50,'(2G18.10)') w(iw),g(iw)
end do
close(50)
write(*,*)
write(*,'("Info(writedosu):")')
write(*,'(" Ultra long-range total density of states written to TDOSULR.OUT")')
write(*,*)
write(*,'(" Fermi energy is at zero in plot")')
write(*,*)
write(*,'(" DOS units are states/Hartree/unit cell")')
deallocate(w,f,g)
! restore original parameters
reducek=reducek0
end subroutine

