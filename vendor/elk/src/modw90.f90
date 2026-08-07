
! Copyright (C) 2015 Jon Lafuente and Manh Duc Le; 2017-18 Arsenii Gerasimov,
! Yaroslav Kvashnin and Lars Nordstrom. This file is distributed under the terms
! of the GNU General Public License. See the file COPYING for license details.

module modw90

use modmain

!---------------------------------------!
!     Wannier90 interface variables     !
!---------------------------------------!
! seedname for all Wannier90 files
character(256) seedname
! number of extra lines to write to .win file
integer nxlwin
! extra lines to write to .win file
character(256), allocatable :: xlwin(:)
! number of Wannier functions to calculate
integer num_wann
! number of bands to pass to Wannier90
integer num_bands
! index to bands
integer, allocatable :: idxw90(:)
! projw90 is .true. if angular momentum projectors should be evaluated
logical projw90
! angular momentum of projectors for each species
integer lprojw90(4,maxspecies)
! number of projectors for each species
integer nprojw90(maxspecies)
! map from the spin, l and atom projector to the Wannier function
integer, allocatable :: prjwn(:,:,:)
! number of iterations for the minimisation of omega
integer num_iter
! number of iterations for disentanglement
integer dis_num_iter
! trial step for the line search minimisation
real(8) trial_step
! maximum number of nearest neighbours per k-point
integer, parameter :: num_nnmax=12
! total number of nearest neighbours for each k-point
integer nntot
! list of nearest neighbours for each k-point
integer, allocatable :: nnlist(:,:)
! G-vector offset for each nearest neighbour
integer, allocatable :: nncell(:,:,:)
! wrtunk is .true. if the UNKkkkkk.s files are to be written in order to
! enable real-space wavefunction plotting
logical wrtunk

end module

