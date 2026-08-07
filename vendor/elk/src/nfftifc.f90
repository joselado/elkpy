
! Copyright (C) 2002-2005 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: nfftifc
! !INTERFACE:
subroutine nfftifc(np,nd,n)
! !INPUT/OUTPUT PARAMETERS:
!   np : number of allowed primes (in,integer)
!   nd : number of dimensions (in,integer)
!   n  : required/available grid size (inout,integer(nd))
! !DESCRIPTION:
!   Interface to the grid requirements of the fast Fourier transform routine.
!   Most routines restrict $n$ to specific prime factorisations. This routine
!   returns the next largest grid size allowed by the FFT routine.
!
! !REVISION HISTORY:
!   Created October 2002 (JKD)
!EOP
!BOC
implicit none
! arguments
integer, intent(in) :: np,nd
integer, intent(inout) :: n(nd)
! local variables
integer i,j,l
integer, parameter :: p(10)=[2,3,5,7,11,13,17,19,23,29]
if ((np < 1).or.(np > 10)) then
  write(*,*)
  write(*,'("Error(nfftifc): np out of range : ",I0)') np
  write(*,*)
  stop
end if
if (nd < 1) then
  write(*,*)
  write(*,'("Error(nfftifc): nd < 1 : ",I0)') nd
  write(*,*)
  stop
end if
! loop over dimensions
do l=1,nd
  if (n(l) < 1) then
    write(*,*)
    write(*,'("Error(nfftifc): n < 1 : ",I0)') n(l)
    write(*,'(" for dimension ",I0)') l
    write(*,*)
    stop
  end if
  do
    i=n(l)
    do j=1,np
      do while (mod(i,p(j)) == 0)
        i=i/p(j)
      end do
    end do
    if (i == 1) exit
    n(l)=n(l)+1
  end do
end do
end subroutine
!EOC

