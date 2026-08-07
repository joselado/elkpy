
! Copyright (C) 2025 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: potdmag
! !INTERFACE:
subroutine potdmag
! !USES:
use modmain
! !DESCRIPTION:
!   Calculates the scalar potential associated with the diamagnetic coupling to
!   an external magnetic field. The vector potential corresponding to a constant
!   magnetic field ${\bf B}$ is given by
!   $$ {\bf A}({\bf r})=\frac{1}{2}{\bf B}\times{\bf r}. $$
!   Substituting this into
!   $$ \hat{H}=\frac{1}{2}\big(\hat{p}+\frac{1}{c}{\bf A}({\bf r})\big)^2 $$
!   yields (among other terms) the diamagnetic contribution to the electronic
!   Hamiltonian:
!   $$ H_{\rm dia}=\frac{B^2 r^2}{8c^2}\big(1-(\hat{\bf B}\cdot
!    \hat{\bf r})^2\big). $$
!   This is applied in the muffin-tins by noting that the term in the brackets
!   is purely angular and can be represented as coefficients $f_{lm}$ of the
!   real spherical harmonics $R_{lm}(\theta,\phi)$. These are given by
!   \begin{gather*}
!     f_{00}=\sqrt{5/3}\,t, \qquad
!     f_{2-2}=t \hat{B}_x \hat{B}_y, \qquad
!     f_{2-1}=t \hat{B}_y \hat{B}_z, \\
!     f_{20}=(t/\sqrt{12})(\hat{B}_x^2+\hat{B}_y^2-2\hat{B}_z^2), \qquad
!     f_{21}=t \hat{B}_x \hat{B}_z, \qquad
!     f_{22}=(t/2)(\hat{B}_y^2-\hat{B}_x^2),
!   \end{gather*}
!   where $t=4\sqrt{\pi/15}$.
!
! !REVISION HISTORY:
!   Suggested by M. Fechner
!   Created June 2025 (JKD)
!EOP
!BOC
implicit none
! local variables
integer is,ias,nr,nri,ir,i
real(8) cbd,bx,by,bz,r2
real(8) f01,f21,f22,f23,f24,f25
! diamagnetic coupling constant of the external field (1/8c²)
cbd=1.d0/(8.d0*solsc**2)
! external B-field only
if (.not.tbdip) then
  bx=bfieldc(1); by=bfieldc(2); bz=bfieldc(3)
  call cfr02
end if
do ias=1,natmtot
  is=idxis(ias)
  nr=nrmt(is)
  nri=nrmti(is)
  if (tbdip) then
! external plus average muffin-tin dipole field
    bx=bfieldc(1)+bdmta(1,ias)
    by=bfieldc(2)+bdmta(2,ias)
    bz=bfieldc(3)+bdmta(3,ias)
    call cfr02
  end if
  i=1
  do ir=1,nr
    r2=rlmt(ir,2,is)
    vsmt(i,ias)=vsmt(i,ias)+f01*r2
    vsmt(i+4,ias)=vsmt(i+4,ias)+f21*r2
    vsmt(i+5,ias)=vsmt(i+5,ias)+f22*r2
    vsmt(i+6,ias)=vsmt(i+6,ias)+f23*r2
    vsmt(i+7,ias)=vsmt(i+7,ias)+f24*r2
    vsmt(i+8,ias)=vsmt(i+8,ias)+f25*r2
    if (ir <= nri) then
      i=i+lmmaxi
    else
      i=i+lmmaxo
    end if
  end do
end do

contains

subroutine cfr02
implicit none
! local variables
real(8) b,t1,t2
! coefficients of the real spherical harmonics for l = 0 and l = 2
b=sqrt(bx**2+by**2+bz**2)
if (b > 1.d-8) then
  bx=bx/b; by=by/b; bz=bz/b
end if
t1=cbd*b**2
t2=t1*4.d0*sqrt(pi/15.d0)
f01=sqrt(5.d0/3.d0)*t2
f21=t2*bx*by
f22=t2*by*bz
f23=(t2/sqrt(12.d0))*(bx**2+by**2-2.d0*bz**2)
f24=t2*bx*bz
f25=(t2/2.d0)*(by**2-bx**2)
end subroutine

end subroutine
!EOC

