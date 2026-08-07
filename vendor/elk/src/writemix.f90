
! Copyright (C) 2025 Eddie Harris-Lee, J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine writemix(nwork,work)
use modmain
implicit none
! arguments
integer, intent(in) :: nwork
real(8), intent(in) :: work(nwork)
open(80,file='MIXWORK'//trim(filext),form='UNFORMATTED',action='WRITE')
write(80) iscl
write(80) nwork
write(80) work
close(80)
end subroutine

