
! Copyright (C) 2025 Wenhan Chen, J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine readbfcr
use modmain
use modulr
implicit none
! local variables
integer idm,i1,i2,i3,ir
integer ngridq_(3),i1_,i2_,i3_
real(8) v(3)
! automatic arrays
real(8) bfcr(nqpt,ndmag)
complex(8) zfft(nfqrz)
! read the real-space external magnetic field in Cartesian coordinates from file
open(50,file='BFCR.OUT',form='FORMATTED')
read(50,*) ngridq_(1:3)
if (any(ngridq(1:3) /= ngridq_(1:3))) then
  write(*,*)
  write(*,'("Error(readbfcr): differing ngridq")')
  write(*,'(" current  :",3(X,I0))') ngridq
  write(*,'(" BFCR.OUT :",3(X,I0))') ngridq_
  write(*,*)
  stop
end if
ir=0
do i3=1,ngridq(3)
  do i2=1,ngridq(2)
    do i1=1,ngridq(1)
      ir=ir+1
      read(50,*) i1_,i2_,i3_,v(:)
      if ((i1 /= i1_).or.(i2 /= i2_).or.(i3 /= i3_)) then
        write(*,*)
        write(*,'("Error(readbfcr): differing i1, i2 or i3")')
        write(*,'(" current  :",3(X,I0))') i1,i2,i3
        write(*,'(" BFCR.OUT :",3(X,I0))') i1_,i2_,i3_
        write(*,*)
        stop
      end if
      if (ncmag) then
        bfcr(ir,1:3)=v(1:3)
      else
        bfcr(ir,1)=v(3)
      end if
    end do
  end do
end do
close(50)
! Fourier transform external magnetic field from real-space to Q-space
do idm=1,ndmag
  call rzfftifc(3,ngridq,-1,bfcr(:,idm),zfft)
  bfcq(idm,1:nfqrz)=zfft(1:nfqrz)
end do
end subroutine

