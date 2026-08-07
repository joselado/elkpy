
! Copyright (C) 2016 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: rfmtlm
! !INTERFACE:
pure subroutine rfmtlm(lm,nr,nri,rfmt,fr)
! !USES:
use modmain
!   lm   : required (l,m) component (in,integer)
!   nr   : number of radial mesh points (in,integer)
!   nri  : number of points on inner part of muffin-tin (in,integer)
!   rfmt : real muffin-tin function (in,real(npmtmax))
!   fr   : (l,m) component function on radial mesh (out,real(nr))
! !DESCRIPTION:
!   Given a function expanded in real spherical harmonics
!   $$ f({\bf r})=\sum_{l=0}^{l_{\rm max}}\sum_{m=-l}^l f_{lm}(r)
!    R_{lm}(\hat{\bf r}), $$
!   where $l_{\rm max}$ corresponds to {\tt lmaxi} and {\tt lmaxo} on the inner
!   and outer parts of the muffin-tin, respectively, this routine returns a
!   particular component function $f_{lm}$. See also {\tt genrlmv}.
!
! !REVISION HISTORY:
!   Created April 2016 (JKD)
!EOP
!BOC
implicit none
! arguments
integer, intent(in) :: lm,nr,nri
real(8), intent(in) :: rfmt(npmtmax)
real(8), intent(out) :: fr(nr)
! local variables
integer iro,i0,i1
if (lm > lmmaxi) then
  fr(1:nri)=0.d0
else
  i1=lmmaxi*(nri-1)+lm
  fr(1:nri)=rfmt(lm:i1:lmmaxi)
end if
iro=nri+1
if (lm > lmmaxo) then
  fr(iro:nr)=0.d0
else
  i0=lmmaxi*nri+lm
  i1=lmmaxo*(nr-iro)+i0
  fr(iro:nr)=rfmt(i0:i1:lmmaxo)
end if
end subroutine
!EOC

