
! Copyright (C) 2017 T. Mueller, J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

module modulr

!-----------------------------!
!     ultracell variables     !
!-----------------------------!
! ultracell lattice vectors stored column-wise
real(8) avecu(3,3)
! ultracell reciprocal lattice vectors
real(8) bvecu(3,3)
! ultracell volume and Brillouin zone volume
real(8) omegau,omegabzu
! original number of k-points
integer nkpt0
! κ-point grid sizes
integer ngridkpa(3)
! integer grid intervals for the κ-points
integer intkpa(2,3)
! number of κ-points
integer nkpa
! R-vectors in Cartesian coordinates spanning the ultracell
real(8), allocatable :: vrcu(:,:)

!------------------------------!
!     G+Q-vector variables     !
!------------------------------!
! small Q cut-off for non-zero Q-vectors
real(8) q0cut
! G+Q-vectors in Cartesian coordinates
real(8), allocatable :: vgqc(:,:,:)
! |G+Q| for all G+Q-vectors
real(8), allocatable :: gqc(:,:)
! Coulomb Green's function in G+Q-space = 4π / |G+Q|²
real(8), allocatable :: gclgq(:,:)
! spherical Bessel functions jₗ(|G+Q|Rₘₜ)
real(8), allocatable :: jlgqrmt(:,:,:,:)
! spherical harmonics of the G+Q-vectors
complex(8), allocatable :: ylmgq(:,:,:)
! structure factors for the G+Q-vectors
complex(8), allocatable :: sfacgq(:,:,:)
! phase factor functions exp(iQ⋅r) in each muffin-tin
complex(8), allocatable :: expqmt(:,:,:)

!---------------------------------------------------!
!     ultra long-range densities and potentials     !
!---------------------------------------------------!
! R-dependent density and magnetisation
real(8), allocatable, target :: rhmgr(:)
real(8), pointer, contiguous :: rhormt(:,:,:),rhorir(:,:)
real(8), pointer, contiguous :: magrmt(:,:,:,:),magrir(:,:,:)
! muffin-tin charges each R-vector
real(8), allocatable :: chgmtru(:,:)
! muffin-tin, interstitial and total moments for each R-vector
real(8), allocatable :: mommtru(:,:,:),momirru(:,:),momtotru(:,:)
! Q-dependent density and magnetisation
complex(8), allocatable :: rhoqmt(:,:,:),rhoqir(:,:)
complex(8), allocatable :: magqmt(:,:,:,:),magqir(:,:,:)
! trdvclr is .true. if the real-space external Coulomb potential should be read
! in from file
logical trdvclr
! Q-dependent external Coulomb potential (FFT ordering)
complex(8), allocatable :: vclq(:)
! trdbfcr is .true. if the real-space external magnetic field in Cartesian
! coordinates should be read in from file
logical trdbfcr
! Q-dependent external magnetic field
complex(8), allocatable :: bfcq(:,:)
! Q-dependent external muffin-tin magnetic fields
complex(8), allocatable :: bfcmtq(:,:,:)
! global external magnetic field in Cartesian coordinates
real(8) bfieldcu(3)
! electric field vector in Cartesian coordinates
real(8) efieldcu(3)
! if tbdipu is .true. then the spin dipole-dipole interaction is included
logical tbdipu
! Q-dependent magnetic dipole field
complex(8), allocatable :: bdipq(:,:)
! Q-dependent Kohn-Sham potential and magnetic field
complex(8), allocatable, target :: vsbsq(:)
complex(8), pointer, contiguous :: vsqmt(:,:,:),vsqir(:,:)
complex(8), pointer, contiguous :: bsqmt(:,:,:,:),bsqir(:,:,:)
! random amplitude used for initialising the long-range magnetic field
real(8) rndbfcu
! if tplotq0 is .true. then the Q = 0 term is included when generating plots
logical tplotq0

!----------------------------------------------!
!     eigenvalue and eigenvector variables     !
!----------------------------------------------!
! number of ultra long-range states
integer nstulr
! long-range eigenvalues
real(8), allocatable :: evalu(:,:)
! long-range occupation numbers
real(8), allocatable :: occulr(:,:)

end module

