
! Copyright (C) 2025 Wenhan Chen, J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine bfcqinit
use modmain
use modulr
use modrandom
implicit none
! local variables
integer ifq,idm,ias
real(8) cb,t1
! coupling constant of the external field (g_e/4c)
cb=gfacte/(4.d0*solsc)
! zero the external magnetic fields
bfcq(1:ndmag,1:nfqrz)=0.d0
bfcmtq(1:natmtot,1:ndmag,1:nfqrz)=0.d0
! read the external fields from file if required
if (trdbfcr) call readbfcr
! add random numbers to magnetic fields if required
if (abs(rndbfcu) > 0.d0) then
  t1=cb*rndbfcu
  do ifq=1,nfqrz
    do idm=1,ndmag
      bfcq(idm,ifq)=bfcq(idm,ifq)+t1*cmplx(randomu()-0.5d0,randomu()-0.5d0,8)
      do ias=1,natmtot
        bfcmtq(ias,idm,ifq)=bfcmtq(ias,idm,ifq) &
         +t1*cmplx(randomu()-0.5d0,randomu()-0.5d0,8)
      end do
    end do
  end do
  bfcq(1:ndmag,1)=dble(bfcq(1:ndmag,1))
  bfcmtq(1:natmtot,1:ndmag,1)=dble(bfcmtq(1:natmtot,1:ndmag,1))
end if
! add the global external magnetic field
do idm=1,ndmag
  if (ncmag) then
    t1=cb*bfieldcu(idm)
  else
    t1=cb*bfieldcu(3)
  end if
  bfcq(idm,1)=bfcq(idm,1)+t1
  bfcmtq(1:natmtot,idm,1)=bfcmtq(1:natmtot,idm,1)+t1
end do
! write the external magnetic fields to file if required
if (.not.trdbfcr) call writebfcr
end subroutine

