
! Copyright (C) 2002-2007 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

module modmain

! crystal name
character(256) cname
! number of atoms
integer natoms
! EOS type
integer etype
! number of volume points to plot
integer nvplt
! volume plot range
real(8) vplt1,vplt2
! number of energy data points to fit
integer nevpt
! volume and energy data point sets
real(8), allocatable :: vpt(:)
real(8), allocatable :: ept(:)
! maximum number of parameters for an EOS
integer, parameter :: maxparam=100
! number of parameters
integer nparam
! EOS name
character(256) ename(2)
! optimized parameter set
real(8) popt(maxparam)
! parameter names
character(256) pname(maxparam)

!-----------------------------!
!     numerical constants     !
!-----------------------------!
real(8), parameter :: pi=3.1415926535897932385d0
real(8), parameter :: twopi=6.2831853071795864769d0
! Planck constant in SI units (exact, CODATA 2018)
real(8), parameter :: h_si=6.62607015d-34
! reduced Planck constant in SI units
real(8), parameter :: hbar_si=h_si/twopi
! Hartree in SI units (CODATA 2018)
real(8), parameter :: ha_si=4.3597447222071d-18
! Hartree in eV (CODATA 2018)
real(8), parameter :: ha_ev=27.211386245988d0
! Bohr radius in SI units (CODATA 2018)
real(8), parameter :: br_si=0.529177210903d-10
! Bohr radius in Angstroms
real(8), parameter :: br_ang=br_si*1.d10
! electron mass in SI (CODATA 2018)
real(8), parameter :: em_si=9.1093837015d-31
! atomic unit of time in SI
real(8), parameter :: t_si=hbar_si/ha_si
! atomic pressure unit in GPa
real(8), parameter :: pr_gpa=1.d-9*em_si/(br_si*t_si**2)
! atomic pressure unit in eV/Å³
real(8), parameter :: pr_eva3=ha_ev/br_ang**3

!---------------------------------!
!     miscellaneous variables     !
!---------------------------------!
! code version
integer version(3)
data version /1,5,0/

end module

