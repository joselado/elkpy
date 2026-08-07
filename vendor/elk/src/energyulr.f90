
! Copyright (C) 2025 Wenhan Chen, J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine energyulr
use modmain
use modulr
implicit none
!-------------------------------------!
!     sum of occupied eigenvalues     !
!-------------------------------------!
evalsum=sum(occulr(1:nstulr,1:nkpt0)*evalu(1:nstulr,1:nkpt0))
! normalise to the unit cell
evalsum=evalsum/(nkpa*nkpt0)
end subroutine

