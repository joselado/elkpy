
! Copyright (C) 2002-2005 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU Lesser General Public
! License. See the file COPYING for license details.

!BOP
! !ROUTINE: roteuler
! !INTERFACE:
pure subroutine roteuler(rot,ang)
! !INPUT/OUTPUT PARAMETERS:
!   rot : proper rotation matrix (in,real(3,3))
!   ang : Euler angles (alpha, beta, gamma) (out,real(3))
! !DESCRIPTION:
!   Given a proper $3\times 3$ rotation matrix
!   \begin{align*}
!    &R(\alpha,\beta,\gamma)=\\
!    &\left(\begin{matrix}
!     \cos\gamma\cos\beta\cos\alpha-\sin\gamma\sin\alpha &
!     \cos\gamma\cos\beta\sin\alpha+\sin\gamma\cos\alpha &
!    -\cos\gamma\sin\beta \\
!    -\sin\gamma\cos\beta\cos\alpha-\cos\gamma\sin\alpha &
!    -\sin\gamma\cos\beta\sin\alpha+\cos\gamma\cos\alpha &
!     \sin\gamma\sin\beta \\
!     \sin\beta\cos\alpha &
!     \sin\beta\sin\alpha &
!     \cos\beta
!    \end{matrix}\right),
!   \end{align*}
!   this routine determines the Euler angles, $(\alpha,\beta,\gamma)$. This
!   corresponds to the so-called `$y$-convention', which involves the following
!   successive rotations of the coordinate system:
!   \begin{itemize}
!    \item[1.]{The $x_1$-, $x_2$-, $x_3$-axes are rotated anticlockwise through
!     an angle $\alpha$ about the $x_3$ axis}
!    \item[2.]{The $x_1'$-, $x_2'$-, $x_3'$-axes are rotated anticlockwise
!     through an angle $\beta$ about the $x_2'$ axis}
!    \item[3.]{The $x_1''$-, $x_2''$-, $x_3''$-axes are rotated anticlockwise
!     through an angle $\gamma$ about the $x_3''$ axis}
!   \end{itemize}
!   Note that the Euler angles are not necessarily unique for a given rotation
!   matrix. It is the responsibility of the calling routine to provide a valid
!   proper rotation matrix, i.e. $R$ is orthogonal and $|R|=1$, since no
!   checking is performed by this routine.
!
! !REVISION HISTORY:
!   Created May 2003 (JKD)
!   Fixed problem thanks to Frank Wagner, June 2013 (JKD)
!   Removed check for invalid rotation matrices, March 2026 (JKD)
!EOP
!BOC
implicit none
! arguments
real(8), intent(in) :: rot(3,3)
real(8), intent(out) :: ang(3)
! local variables
real(8), parameter :: eps=1.d-8
real(8), parameter :: pi=3.1415926535897932385d0
if ((abs(rot(3,1)) > eps).or.(abs(rot(3,2)) > eps)) then
  ang(1)=atan2(rot(3,2),rot(3,1))
  if (abs(rot(3,1)) > abs(rot(3,2))) then
    ang(2)=atan2(rot(3,1)/cos(ang(1)),rot(3,3))
  else
    ang(2)=atan2(rot(3,2)/sin(ang(1)),rot(3,3))
  end if
  ang(3)=atan2(rot(2,3),-rot(1,3))
else
  ang(1)=atan2(rot(1,2),rot(1,1))
  if (rot(3,3) > 0.d0) then
    ang(2)=0.d0
    ang(3)=0.d0
  else
    ang(2)=pi
    ang(3)=pi
  end if
end if
end subroutine
!EOC

