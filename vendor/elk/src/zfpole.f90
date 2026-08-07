
! Copyright (C) 2017 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: zfpole
! !INTERFACE:
pure complex(8) function zfpole(n,c,z)
! !INPUT/OUTPUT PARAMETERS:
!   n : number of poles (in,integer)
!   c : constant, pole and residue data (in,complex(*))
!   z : point at which function is to be evaluated (in,complex)
! !DESCRIPTION:
!   Returns the complex function
!   $$ f(z)=C+\sum_{i=1}^n \frac{a_i}{z_i-z}, $$
!   where $C={\tt c(1)}$, $z_1={\tt c(2)}$, $a_1={\tt c(3)}$, and so on.
!
! !REVISION HISTORY:
!   Created October 2017 (JKD)
!EOP
!BOC
implicit none
! arguments
integer, intent(in) :: n
complex(8), intent(in) :: c(*),z
! local variables
integer i
complex(8) z1
zfpole=c(1)
do i=2,2*n,2
  z1=c(i)+z
  if (abs(z1%re)+abs(z1%im) > 1.d-8) then
    zfpole=zfpole+c(i+1)/z1
  end if
end do
end function
!EOC

