
! Copyright (C) 2002-2007 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

real(8) function pveos(etype,param,v)
! pressure-volume equation of state function
implicit none
! arguments
integer, intent(in) :: etype
real(8), intent(in) :: param(*),v
! local variables
real(8) vm,vp,pm,pp,dv
! external functions
real(8), external :: eveos
! use central differences
dv=1.d-3
vm=v-dv
vp=v+dv
pm=eveos(etype,param,vm)
pp=eveos(etype,param,vp)
pveos=-(pp-pm)/(2.d0*dv)
end function

