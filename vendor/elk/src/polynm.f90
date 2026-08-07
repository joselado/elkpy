
! Copyright (C) 2002-2005 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU Lesser General Public
! License. See the file COPYING for license details.

!BOP
! !ROUTINE: polynm
! !INTERFACE:
pure real(8) function polynm(m,np,xa,ya,x)
! !INPUT/OUTPUT PARAMETERS:
!   m  : order of derivative (in,integer)
!   np : number of points to fit (in,integer)
!   ip : point at which m'th derivative is to be evaluated (in,integer)
!   xa : abscissa array (in,real(np))
!   ya : ordinate array (in,real(np))
!   x  : evaluation abscissa (in,real)
! !DESCRIPTION:
!   Fits a polynomial of order $n_p-1$ to a set of $n_p$ points. If $m\ge 0$ the
!   function returns the $m$th derviative of the polynomial at $x$, while for
!   $m<0$ the integral of the polynomial from the first point in the array to
!   $x$ is returned.
!
! !REVISION HISTORY:
!   Created October 2002 (JKD)
!   Simplified, January 2025 (JKD)
!EOP
!BOC
implicit none
! argmuments
integer, intent(in) :: m,np
real(8), intent(in) :: xa(np),ya(np),x
! local variables
integer i,j,k
real(8) x0,t1
! automatic arrays
real(8) c(np)
if ((np < 1).or.(m >= np)) then
  polynm=0.d0
  return
end if
! find the polynomial coefficients in divided differences form
c(1:np)=ya(1:np)
do i=2,np
  do j=np,i,-1
    c(j)=(c(j)-c(j-1))/(xa(j)-xa(j+1-i))
  end do
end do
! special case m = 0
if (m == 0) then
  polynm=c(1)
  t1=1.d0
  do i=2,np
    t1=t1*(x-xa(i-1))
    polynm=polynm+c(i)*t1
  end do
  return
end if
x0=xa(1)
! convert to standard form
do j=1,np-1
  do i=1,np-j
    k=np-i
    c(k)=c(k)+(x0-xa(k-j+1))*c(k+1)
  end do
end do
if (m > 0) then
! take the m'th derivative
  do j=1,m
    do i=m+1,np
      c(i)=c(i)*dble(i-j)
    end do
  end do
  polynm=c(np)
  t1=x-x0
  do i=np-1,m+1,-1
    polynm=polynm*t1+c(i)
  end do
else
! find the integral
  polynm=c(np)/dble(np)
  t1=x-x0
  do i=np-1,1,-1
    polynm=polynm*t1+c(i)/dble(i)
  end do
  polynm=polynm*t1
end if
end function
!EOC

