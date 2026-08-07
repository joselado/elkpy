
! Copyright (C) 2020 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: rfmtint
! !INTERFACE:
pure real(8) function rfmtint(nr,nri,wr,rfmt)
! !USES:
use modmain
! !INPUT/OUTPUT PARAMETERS:
!   nr   : number of radial mesh points (in,integer)
!   nri  : number of points on inner part of muffin-tin (in,integer)
!   wr   : weights for integration on radial mesh (in,real(nr))
!   rfmt : real function inside muffin-tin (in,real(*))
! !DESCRIPTION:
!   Computes the integral of a real muffin-tin function. In other words, if
!   $$ f({\bf r})=\sum_{l=0}^{l_{\rm max}}\sum_{m=-l}^{l}f_{lm}(r)R_{lm}
!    (\hat{\bf r}) $$
!   where $R_{lm}$ are the real spherical harmonics, then this routine returns
!   $$ I=4\pi Y_{00}\int f_{00}(r)r^2 dr\;. $$
!
! !REVISION HISTORY:
!   Created July 2020 (JKD)
!EOP
!BOC
implicit none
! arguments
integer, intent(in) :: nr,nri
real(8), intent(in) :: wr(nr),rfmt(*)
! local variables
integer iro,i0,i1
i1=lmmaxi*(nri-1)+1
rfmtint=sum(wr(1:nri)*rfmt(1:i1:lmmaxi))
iro=nri+1
i0=i1+lmmaxi
i1=lmmaxo*(nr-iro)+i0
rfmtint=rfmtint+sum(wr(iro:nr)*rfmt(i0:i1:lmmaxo))
rfmtint=fourpi*y00*rfmtint
end function
!EOC

