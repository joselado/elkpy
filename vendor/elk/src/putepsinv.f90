
! Copyright (C) 2018 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine putepsinv(iq,epsi)
use modmain
use modramdisk
implicit none
! arguments
integer, intent(in) :: iq
complex(8), intent(in) :: epsi(ngrf,ngrf,nwrf)
! local variables
integer recl
! determine the record length for EPSINV.OUT
inquire(iolength=recl) vql(1:3,iq),ngrf,nwrf,epsi
!$OMP CRITICAL(u245)
! write to RAM disk if required
if (ramdisk) then
  call putrd('EPSINV.OUT',iq,v1=vql(1:3,iq),n1=ngrf,n2=nwrf, &
   nzv=ngrf*ngrf*nwrf,zva=epsi)
end if
! write to disk if required
if (wrtdisk) then
  open(245,file='EPSINV.OUT',form='UNFORMATTED',access='DIRECT',recl=recl)
  write(245,rec=iq) vql(1:3,iq),ngrf,nwrf,epsi
  close(245)
end if
!$OMP END CRITICAL(u245)
end subroutine

