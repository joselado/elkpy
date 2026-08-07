
! Copyright (C) 2012 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine nuclei
use modmain
implicit none
! local variables
integer is,nr
real(8) rmn,t1
! external functions
real(8), external :: radnucl
do is=1,nspecies
  nr=nrmt(is)
  rmn=rminsp(is)
  t1=dble(nr-1)/log(rmt(is)/rmn)
! approximate nuclear radius
  rnucl(is)=radnucl(spzn(is))
! number of radial mesh points to nuclear radius
  nrnucl(is)=nint(t1*log(rnucl(is)/rmn))+1
  nrnucl(is)=max(min(nrnucl(is),nr),1)
! nuclear volume determined from integrating to nuclear radius; this is to
! ensure integrals over the nuclear volume are correctly normalised
  volnucl(is)=fourpi*sum(wr2mt(1:nrnucl(is),is))
! Thomson radius
  rtmsn(is)=abs(spzn(is))/solsc**2
! number of radial mesh points to Thomson radius
  nrtmsn(is)=nint(t1*log(rtmsn(is)/rmn))+1
  nrtmsn(is)=max(min(nrtmsn(is),nr),1)
! Thomson volume
  voltmsn(is)=fourpi*sum(wr2mt(1:nrtmsn(is),is))
end do
end subroutine

