
! Copyright (C) 2025 Eddie Harris-Lee, J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine readmix(trs,iscl0,nwork,work)
use modmain
implicit none
! arguments
logical, intent(out) :: trs
integer, intent(out) :: iscl0
integer, intent(in) :: nwork
real(8), intent(out) :: work(nwork)
! local variables
integer ios,nwork_
trs=.false.
open(80,file='MIXWORK'//trim(filext),form='UNFORMATTED',action='READ', &
 status='OLD',err=10)
read(80,err=10) iscl0
read(80,err=10) nwork_
if (nwork /= nwork_) goto 10
read(80,err=10) work
trs=.true.
10 continue
close(80,iostat=ios)
end subroutine

