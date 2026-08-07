
! Copyright (C) 2002-2005 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine addbfsm
use modmain
implicit none
! local variables
integer idm,is,ias
real(8) t1
! add the global fixed spin moment B-field to the Kohn-Sham field
if (any(abs(fsmtype) == [1,3,4,6])) then
  do idm=1,ndmag
    t1=bfsmc(idm)
    do ias=1,natmtot
      is=idxis(ias)
      bsmt(1:npcmt(is),ias,idm)=bsmt(1:npcmt(is),ias,idm)+t1
    end do
    bsir(1:ngtot,idm)=bsir(1:ngtot,idm)+t1
    bsirc(1:ngtc,idm)=bsirc(1:ngtc,idm)+t1*cfrc(1:ngtc)
  end do
end if
! add the muffin-tin fields
if (any(abs(fsmtype) == [2,3,5,6])) then
  do idm=1,ndmag
    do ias=1,natmtot
      is=idxis(ias)
      t1=bfsmcmt(idm,ias)
      bsmt(1:npcmt(is),ias,idm)=bsmt(1:npcmt(is),ias,idm)+t1
    end do
  end do
end if
end subroutine

