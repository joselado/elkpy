
! Copyright (C) 2022 Leon Kerber, J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: tm3vdl
! !INTERFACE:
subroutine tm3vdl(l,k,p,r,ld,wkpr,wkpr_v)
! !INPUT/OUTPUT PARAMETERS:
!   l      : angular momentum quantum number (in,integer)
!   k      : k-index of tensor moment (in,integer)
!   p      : p-index of tensor moment (in,integer)
!   r      : r-index of tensor moment (in,integer)
!   ld     : leading dimension (in,integer)
!   wkpr   : 3-index real tensor moment components (in,real(-ld:ld))
!   wkpr_v : complex van der Laan components (out,complex(-ld:ld))
! !DESCRIPTION:
!   Converts tensor moments from the real form used internally to the original
!   complex convention of G. van der Laan and B. T. Thole, Appendix A in
!   {\it J. Phys.: Condens. Matter} {\bf 7}, 9947 (1995). The coupled 3-index
!   tensors are given by
!   $$ \Gamma_t^{kpr}=\sum_{x=-k}^k\sum_{y=-p}^p
!    \Gamma_{xy}^{kp}
!    \begin{pmatrix} k & r & p \\ -x & t & -y \end{pmatrix}
!    (-1)^{k-x+p-y} \underline{n}_{kpr}^{-1}, $$
!   where
!   $$ \Big(\Gamma_{xy}^{kp}\Big)_{m_1\sigma_1,m_2\sigma_2}=(-1)^{l-m_2}
!    \begin{pmatrix} l & k & l \\ -m_2 & x & m_1 \end{pmatrix}
!    (-1)^{s-\sigma_2}
!    \begin{pmatrix} s & p & s \\ -\sigma_2 & y & \sigma_1 \end{pmatrix}
!    n_{lk}^{-1} n_{sp}^{-1}, $$
!   $$ n_{lk}=\frac{(2l)!}{\sqrt{(2l-k)!(2l+1+k)!}} $$
!   and
!   $$ \underline{n}_{kpr}=
!    \begin{pmatrix} k & p & r \\ 0 & 0 & 0 \end{pmatrix} $$
!   if $g=k+p+r$ is even, and
!   $$ \underline{n}_{kpr}=
!    i^g\left[\frac{(g-2k)!(g-2p)!(g-2r)!}{(g+1)!}\right]^{\frac{1}{2}}
!    \frac{g!!}{(g-2k)!!(g-2p)!!(g-2r)!!} $$
!   if $g$ is odd.
!
! !REVISION HISTORY:
!   Created October 2022 (JKD and Leon Kerber)
!EOP
!BOC
implicit none
! arguments
integer, intent(in) :: l,k,p,r,ld
real(8), intent(in) :: wkpr(-ld:ld)
complex(8), intent(out) :: wkpr_v(-ld:ld)
! local variables
integer g,t
real(8) t0,t1
complex(8), parameter :: zi=(0.d0,1.d0)
complex(8) z1
! external functions
real(8), external :: wigner3j,factn,factn2,factr
! complex convention normalisation factors
g=k+p+r
if (mod(g,2) == 0) then
  t0=1.d0/wigner3j(k,p,r,0,0,0)
else
  t0=sqrt(factr(g+1,g-2*k)/(factn(g-2*p)*factn(g-2*r)))
  t0=t0*factn2(g-2*k)*factn2(g-2*p)*factn2(g-2*r)/factn2(g)
end if
t0=t0*sqrt(factn(2+p)*factn(2*l-k)*factn(2*l+k+1))/factn(2*l)
if (mod(k+p,2) /= 0) t0=-t0
! remove Hermitian convention normalisation factors
t0=t0/sqrt(dble((2*k+1)*(2*p+1)*(2*r+1)))
t0=t0/2.d0
! construct complex tensor moments
wkpr_v(-r:r)=0.d0
do t=-r,r
  t1=t0*wkpr(t)
  if (mod(t,2) /= 0) t1=-t1
  wkpr_v(t)=wkpr_v(t)+t1*(1.d0,1.d0)
  if (mod(k+p+r+t,2) /= 0) t1=-t1
  wkpr_v(-t)=wkpr_v(-t)+t1*(1.d0,-1.d0)
end do
! multiply by i⁻ᵍ if g is odd
if (mod(g,2) /= 0) then
  z1=zi**(-g)
  wkpr_v(-r:r)=wkpr_v(-r:r)*z1
end if
end subroutine
!EOC

