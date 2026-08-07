
! Copyright (C) 2023 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

complex(8) function zcfinp(cfmt1,cfir1,cfmt2,cfir2)
use modmain
use modomp
implicit none
! arguments
complex(4), intent(in) :: cfmt1(npcmtmax,natmtot),cfir1(ngtc)
complex(4), intent(in) :: cfmt2(npcmtmax,natmtot),cfir2(ngtc)
! local variables
integer is,ias,nthd
! external functions
complex(4), external :: cdotc
complex(8), external :: zcfmtinp
zcfinp=0.d0
call holdthd(natmtot+1,nthd)
!$OMP PARALLEL DEFAULT(SHARED) &
!$OMP PRIVATE(is) REDUCTION(+:zcfinp) &
!$OMP NUM_THREADS(nthd)
!$OMP DO SCHEDULE(DYNAMIC)
do ias=1,natmtot
  is=idxis(ias)
! muffin-tin contribution
  zcfinp=zcfinp+zcfmtinp(nrcmt(is),nrcmti(is),wr2cmt(:,is),cfmt1(:,ias), &
   cfmt2(:,ias))
end do
!$OMP END DO NOWAIT
! interstitial contribution (requires that one of the functions has been
! multiplied by the characteristic function)
!$OMP SINGLE
zcfinp=zcfinp+(omega/ngtc)*cdotc(ngtc,cfir1,1,cfir2,1)
!$OMP END SINGLE
!$OMP END PARALLEL
call freethd(nthd)
end function

