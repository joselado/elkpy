
! Copyright (C) 2015 Manh Duc Le, 2017-18 Arsenii Gerasimov, Yaroslav Kvashnin,
! Lars Nordstrom and J. K. Dewhurst. This file is distributed under the terms of
! the GNU General Public License. See the file COPYING for license details.

!BOP
! !ROUTINE: numlist
! !INTERFACE:
subroutine numlist(str,n,list)
! !INPUT/OUTPUT PARAMETERS:
!   str  : string to be converted to list of numbers (in,character(*))
!   n    : on entry, the maximum allowed number of elements; on exit, the number
!          of elements in the list (inout,integer)
!   list : list of elements (out,integer(n))
! !DESCRIPTION:
!   Converts a space- or comma-separated string of integers, including ranges,
!   to a list. For example, the string `{\tt 1,2,10-13 5 6}' would be converted
!   to the list {\tt 1,2,10,11,12,13,5,6}.
!
! !REVISION HISTORY:
!   Created May 2015 (Manh Duc Le)
!EOP
!BOC
implicit none
! arguments
character(*), intent(in) :: str
integer, intent(inout) :: n
integer, intent(out) :: list(n)
! local variables
integer i0,i1,i,j,m,ios
! automatic arrays
integer l(n)
i=0
i0=1
do
  m=index(str(i0:),'-')
  if (m == 0) then
    i1=256
  else
    i1=i0+m-2
  end if
  l(:)=0
  read(str(i0:i1),*,iostat=ios) l
  if (i > 0) then
    do j=list(i)+1,l(1)-1
      if (i == n) return
      i=i+1
      list(i)=j
    end do
  end if
  do j=1,n
    if (l(j) == 0) exit
    if (i == n) return
    i=i+1
    list(i)=l(j)
  end do
  if (m == 0) then
    n=i
    return
  end if
  i0=i0+m
end do
end subroutine
!EOC

