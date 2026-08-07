
! Copyright (C) 2025 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine writeelmirep(fext,elm)
use modmain
implicit none
! arguments
character(*), intent(in) :: fext
real(8), intent(in) :: elm(lmmaxdb,natmtot)
! local variables
integer is,ia,ias,l,m,lm
open(50,file='ELMIREP'//trim(fext),form='FORMATTED',action='WRITE')
do ias=1,natmtot
  is=idxis(ias)
  ia=idxia(ias)
  write(50,*)
  write(50,'("Species : ",I4," (",A,"), atom : ",I4)') is,trim(spsymb(is)),ia
  lm=0
  do l=0,lmaxdb
    do m=-l,l
      lm=lm+1
      write(50,'(" l = ",I2,", m = ",I2,", lm = ",I3," : ",G18.10)') l,m,lm, &
       elm(lm,ias)
    end do
  end do
end do
close(50)
end subroutine

