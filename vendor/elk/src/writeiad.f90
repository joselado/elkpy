
! Copyright (C) 2002-2005 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: writeiad
! !INTERFACE:
subroutine writeiad(fnum)
! !USES:
use modmain
! !INPUT/OUTPUT PARAMETERS:
!   fnum : file number for writing output (in,integer)
! !DESCRIPTION:
!   Outputs the interatomic distances to file.
!
! !REVISION HISTORY:
!   Created May 2005 (JKD)
!EOP
!BOC
implicit none
! arguments
integer, intent(in) :: fnum
! local variables
integer is,js,ia,ja
integer i1,i2,i3
real(8) d,dmin,v(3)
do is=1,nspecies
  do ia=1,natoms(is)
    write(fnum,*)
    write(fnum,'("Distance between is = ",I0," (",A,"), ia = ",I0," and")') &
     is,trim(spsymb(is)),ia
    do js=1,nspecies
      do ja=1,natoms(js)
        dmin=1.d8
        do i1=-1,1
          do i2=-1,1
            do i3=-1,1
              v(1:3)=dble(i1)*avec(1:3,1) &
                    +dble(i2)*avec(1:3,2) &
                    +dble(i3)*avec(1:3,3)+atposc(1:3,ja,js)
              v(1:3)=v(1:3)-atposc(1:3,ia,is)
              d=sqrt(v(1)**2+v(2)**2+v(3)**2)
              dmin=min(d,dmin)
            end do
          end do
        end do
        write(fnum,'(" is = ",I0," (",A,"), ia = ",I0," : ",G18.10)') js, &
         trim(spsymb(js)),ja,dmin
      end do
    end do
  end do
end do
end subroutine
!EOC

