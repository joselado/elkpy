
! Copyright (C) 2022 J. K. Dewhurst, S. Sharma and E. Harris-Lee.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

module modtdhfc

! energy window cut-off for TDHFC states around the Fermi energy
real(8) ecutthc
! maximum number of second-variational states within the energy window over all
! k-points
integer nmaxthc
! total number of TDHFC states
integer nthc
! number of second-variational states within the energy window for each k-point
integer, allocatable :: nstthc(:)
! index to used states for each k-point
integer, allocatable :: idxthc(:,:)
! index from state and k-point to TDHFC states
integer, allocatable :: ithc(:,:)

end module

