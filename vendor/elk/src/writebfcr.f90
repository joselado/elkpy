
! Copyright (C) 2025 Wenhan Chen, J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine writebfcr
use modmain
use modulr
implicit none
! local variables
integer idm,i1,i2,i3,ir
real(8) v(3)
! automatic arrays
real(8) bfcr(nqpt,ndmag)
complex(8) zfft(nfqrz)
! Fourier transform external magnetic field from Q-space to real-space
do idm=1,ndmag
  zfft(1:nfqrz)=bfcq(idm,1:nfqrz)
  call rzfftifc(3,ngridq,1,bfcr(:,idm),zfft)
end do
! write the real-space magnetic field in Cartesian coordinates to file
open(50,file='BFCR.OUT',form='FORMATTED')
write(50,'(3I6," : grid size")') ngridq
ir=0
do i3=1,ngridq(3)
  do i2=1,ngridq(2)
    do i1=1,ngridq(1)
      ir=ir+1
      if (ncmag) then
        v(1:3)=bfcr(ir,1:3)
      else
        v(1:2)=0.d0
        v(3)=bfcr(ir,1)
      end if
      write(50,'(3I6,3G18.10)') i1,i2,i3,v
    end do
  end do
end do
close(50)
end subroutine

