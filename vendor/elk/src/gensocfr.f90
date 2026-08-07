
! Copyright (C) 2009 J. K. Dewhurst, S. Sharma and E. K. U. Gross
! This file is distributed under the terms of the GNU Lesser General Public
! License. See the file COPYING for license details.

!BOP
! !ROUTINE: gensocfr
! !INTERFACE:
subroutine gensocfr
! !USES:
use modmain
use modomp
! !DESCRIPTION:
!   Calculates the radial part of the spin-orbit coupling term which is added to
!   the second-variational Hamiltonian. In a particular muffin-tin, this is
!   given by
!   $$ f_{\rm soc}(r)=\frac{1}{(2Mc)^2}\frac{1}{r}
!    \frac{\partial V_s}{\partial r}, $$
!   where
!   $$ M(r)=1+\frac{1}{2c^2}(E-V_s) $$
!   (with $E$ set to zero) and $V_s$ is the spherical part of the Kohn-Sham
!   potential. The term added to the Hamiltonian is then
!   $$ \hat{H}_{\rm soc}(r)=f_{\rm soc}\hat{\bf L}\cdot\boldsymbol{\sigma}. $$
!   See Koelling and Harmon, {\it J. Phys. C: Solid State Phys.} {\bf 10}, 3107
!   (1977) for details.
!
! !REVISION HISTORY:
!   Created April 2009 (JKD)
!EOP
!BOC
implicit none
! local variables
integer is,ias,nthd
integer nr,nri,ir,irc
real(8) cso,rm
! automatic arrays
real(8) vr(nrmtmax),dvr(nrmtmax)
if (.not.spinorb) return
! coefficient of spin-orbit coupling
cso=y00*socscf/(4.d0*solsc**2)
call holdthd(natmtot,nthd)
!$OMP PARALLEL DO DEFAULT(SHARED) &
!$OMP PRIVATE(vr,dvr,is,nr,nri) &
!$OMP PRIVATE(ir,irc,rm) &
!$OMP SCHEDULE(DYNAMIC) &
!$OMP NUM_THREADS(nthd)
do ias=1,natmtot
  is=idxis(ias)
  nr=nrmt(is)
  nri=nrmti(is)
! radial derivative of the spherical part of the Kohn-Sham potential
  call rfmtlm(1,nr,nri,vsmt(:,ias),vr)
  call splined(nr,wcrmt(:,:,is),vr,dvr)
  do ir=1,nr,lradstp
    irc=(ir-1)/lradstp+1
    rm=1.d0-2.d0*cso*vr(ir)
    socfr(irc,ias)=cso*dvr(ir)/(rsp(ir,is)*rm**2)
  end do
end do
!$OMP END PARALLEL DO
call freethd(nthd)

contains

pure subroutine splined(n,wc,f,df)
implicit none
! arguments
integer, intent(in) :: n
real(8), intent(in) :: wc(12,n),f(n)
real(8), intent(out) :: df(n)
! local variables
integer i
df(1)=wc(1,1)*f(1)+wc(2,1)*f(2)+wc(3,1)*f(3)+wc(4,1)*f(4)
df(2)=wc(1,2)*f(1)+wc(2,2)*f(2)+wc(3,2)*f(3)+wc(4,2)*f(4)
do i=3,n-2
  df(i)=wc(1,i)*f(i-1)+wc(2,i)*f(i)+wc(3,i)*f(i+1)+wc(4,i)*f(i+2)
end do
i=n-1
df(i)=wc(1,i)*f(n-3)+wc(2,i)*f(n-2)+wc(3,i)*f(n-1)+wc(4,i)*f(n)
df(n)=wc(1,n)*f(n-3)+wc(2,n)*f(n-2)+wc(3,n)*f(n-1)+wc(4,n)*f(n)
end subroutine

end subroutine
!EOC

