
! Copyright (C) 2025 Wenhan Chen, J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine writemomqu
use modmain
use modulr
implicit none
! local variables
integer idm,is,ia,ias,iq,ifq
! automatic arrays
complex(8) zfft(nqpt),mmtqu(ndmag,natmtot,nqpt)
complex(8) mirqu(ndmag,nqpt),mtotqu(ndmag,nqpt)
do idm=1,ndmag
! muffin-tin moments
  do ias=1,natmtot
    is=idxis(ias)
    zfft(1:nqpt)=mommtru(idm,ias,1:nqpt)
    call zfftifc(3,ngridq,-1,zfft)
    mmtqu(idm,ias,1:nqpt)=zfft(1:nqpt)
  end do
! interstitial moment
  zfft(1:nqpt)=momirru(idm,1:nqpt)
  call zfftifc(3,ngridq,-1,zfft)
  mirqu(idm,1:nqpt)=zfft(1:nqpt)
! total moment
  zfft(1:nqpt)=momtotru(idm,1:nqpt)
  call zfftifc(3,ngridq,-1,zfft)
  mtotqu(idm,1:nqpt)=zfft(1:nqpt)
end do
open(50,file='MOMENTQU.OUT',form='FORMATTED')
do iq=1,nqpt
  ifq=iqfft(iq)
  write(50,*)
  write(50,'("Q-point number ",I6," of ",I6)') iq,nqpt
  write(50,'("Q-point (lattice coordinates) :")')
  write(50,'(3G18.10)') vql(:,iq)
  write(50,'("Q-point (Cartesian coordinates) :")')
  write(50,'(3G18.10)') vqc(:,iq)
  write(50,'("Moments (complex):")')
  write(50,'(" interstitial :")')
  do idm=1,ndmag
    write(50,'(2G18.10)') mirqu(idm,ifq)
  end do
  write(50,'(" muffin-tins")')
  do is=1,nspecies
    write(50,'("  species : ",I4," (",A,")")') is,trim(spsymb(is))
    do ia=1,natoms(is)
      ias=idxas(ia,is)
      write(50,'("   atom ",I4," :")') ia
      do idm=1,ndmag
        write(50,'(2G18.10)') mmtqu(idm,ias,ifq)
      end do
    end do
  end do
  write(50,'(" total moment :")')
  do idm=1,ndmag
    write(50,'(2G18.10)') mtotqu(idm,ifq)
  end do
end do
close(50)
end subroutine

