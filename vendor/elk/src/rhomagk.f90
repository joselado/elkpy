
! Copyright (C) 2002-2010 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: rhomagk
! !INTERFACE:
subroutine rhomagk(ngp,igpig,wppt,occsvp,apwalm,evecfv,evecsv,rhomt_,rhoir_, &
 magmt_,magir_)
! !USES:
use modmain
use modomp
! !INPUT/OUTPUT PARAMETERS:
!   ngp    : number of G+p-vectors (in,integer(nspnfv))
!   igpig  : index from G+p-vectors to G-vectors (in,integer(ngkmax,nspnfv))
!   wppt   : weight of input p-point (in,real)
!   occsvp : occupation number for each state (in,real(nstsv))
!   apwalm : APW matching coefficients
!            (in,complex(ngkmax,apwordmax,lmmaxapw,natmtot,nspnfv))
!   evecfv : first-variational eigenvectors (in,complex(nmatmax,nstfv,nspnfv))
!   evecsv : second-variational eigenvectors (in,complex(nstsv,nstsv))
!   rhomt_ : muffin-tin density (inout,real(npcmtmax,natmtot))
!   rhoir_ : interstitial density (inout,real(ngtc))
!   magmt_ : muffin-tin magnetisation (inout,real(npcmtmax,natmtot,ndmag))
!   magir_ : interstitial magnetisation (inout,real(ngtc,ndmag))
! !DESCRIPTION:
!   Generates the partial valence charge density and magnetisation from the
!   eigenvectors at a particular $k$-point. In the muffin-tin region, the
!   wavefunction is obtained in terms of its $(l,m)$-components from both the
!   APW and local-orbital functions. Using a backward spherical harmonic
!   transform (SHT), the wavefunction is converted to real-space and the density
!   obtained from its modulus squared. A similar process is used for the
!   interstitial density in which the wavefunction in real-space is obtained
!   from a Fourier transform of the APW functions. See routines {\tt wfmtsv},
!   {\tt genshtmat} and {\tt eveqn}.
!
! !REVISION HISTORY:
!   Created April 2003 (JKD)
!   Removed conversion to spherical harmonics, January 2009 (JKD)
!   Partially de-phased the muffin-tin magnetisation for spin-spirals,
!    February 2009 (FC, FB & LN)
!   Optimisations, July 2010 (JKD)
!EOP
!BOC
implicit none
! arguments
integer, intent(in) :: ngp(nspnfv),igpig(ngkmax,nspnfv)
real(8), intent(in) :: wppt,occsvp(nstsv)
complex(8), intent(in) :: apwalm(ngkmax,apwordmax,lmmaxapw,natmtot,nspnfv)
complex(8), intent(in) :: evecfv(nmatmax,nstfv,nspnfv),evecsv(nstsv,nstsv)
real(8), intent(inout) :: rhomt_(npcmtmax,natmtot),rhoir_(ngtc)
real(8), intent(inout) :: magmt_(npcmtmax,natmtot,ndmag),magir_(ngtc,ndmag)
! local variables
integer ispn,jspn,nst,ist
integer is,ias,npc,j,k
integer n,igp,nthd
real(8) wo,ts0,ts1
complex(8) z1
! automatic arrays
integer idx(nstsv)
complex(8) wfir(ngtc,nspinor),wfgp(ngkmax)
! allocatable arrays
complex(8), allocatable :: wfmt(:,:,:)
call timesec(ts0)
! number of and index to occupied states
nst=0
do ist=1,nstsv
  if (abs(occsvp(ist)) < epsocc) cycle
  nst=nst+1
  idx(nst)=ist
end do
!----------------------------------------------!
!     muffin-tin density and magnetisation     !
!----------------------------------------------!
call holdthd(max(natmtot,nst),nthd)
!$OMP PARALLEL DEFAULT(SHARED) &
!$OMP PRIVATE(wfmt,wfir,wfgp) &
!$OMP PRIVATE(ias,is,npc,j,k,wo) &
!$OMP PRIVATE(ispn,jspn,n,ist,z1,igp) &
!$OMP NUM_THREADS(nthd)
allocate(wfmt(npcmtmax,nspinor,nst))
!$OMP DO SCHEDULE(DYNAMIC)
do ias=1,natmtot
  is=idxis(ias)
  npc=npcmt(is)
  call wfmtsv(.false.,lradstp,is,ias,nst,idx,ngp,apwalm,evecfv,evecsv,npcmtmax,&
   wfmt)
  do j=1,nst
    k=idx(j)
    wo=occsvp(k)*wppt
! add to density and magnetisation
    if (spinpol) then
! spin-polarised
      if (ncmag) then
! non-collinear
        call rmk1(npc,wo,wfmt(:,1,j),wfmt(:,2,j),rhomt_(:,ias),magmt_(:,ias,1),&
         magmt_(:,ias,2),magmt_(:,ias,3))
      else
! collinear
        call rmk2(npc,wo,wfmt(:,1,j),wfmt(:,2,j),rhomt_(:,ias),magmt_(:,ias,1))
      end if
    else
! spin-unpolarised
      call rmk3(npc,wo,wfmt(:,1,j),rhomt_(:,ias))
    end if
  end do
end do
!$OMP END DO NOWAIT
deallocate(wfmt)
!------------------------------------------------!
!     interstitial density and magnetisation     !
!------------------------------------------------!
!$OMP DO ORDERED SCHEDULE(DYNAMIC)
do j=1,nst
  k=idx(j)
  wo=occsvp(k)*wppt/omega
  if (tevecsv) then
! generate spinor wavefunction from second-variational eigenvectors
    do ispn=1,nspinor
      jspn=jspnfv(ispn)
      n=ngp(jspn)
      wfgp(1:n)=0.d0
      do ist=1,nstfv
        z1=evecsv((ispn-1)*nstfv+ist,k)
        if (abs(z1%re)+abs(z1%im) > 1.d-12) &
         wfgp(1:n)=wfgp(1:n)+z1*evecfv(1:n,ist,jspn)
      end do
      wfir(1:ngtc,ispn)=0.d0
      do igp=1,n
        wfir(igfc(igpig(igp,jspn)),ispn)=wfgp(igp)
      end do
! Fourier transform wavefunction to real-space
      call zfftifc(3,ngdgc,1,wfir(:,ispn))
    end do
  else
! spin-unpolarised wavefunction
    wfir(1:ngtc,1)=0.d0
    do igp=1,ngp(1)
      wfir(igfc(igpig(igp,1)),1)=evecfv(igp,k,1)
    end do
    call zfftifc(3,ngdgc,1,wfir)
  end if
! add to density and magnetisation
!$OMP ORDERED
  if (spinpol) then
! spin-polarised
    if (ncmag) then
! non-collinear
      call rmk1(ngtc,wo,wfir,wfir(:,2),rhoir_,magir_,magir_(:,2),magir_(:,3))
    else
! collinear
      call rmk2(ngtc,wo,wfir,wfir(:,2),rhoir_,magir_)
    end if
  else
! spin-unpolarised
    call rmk3(ngtc,wo,wfir,rhoir_)
  end if
!$OMP END ORDERED
end do
!$OMP END DO
!$OMP END PARALLEL
call freethd(nthd)
call timesec(ts1)
!$OMP ATOMIC
timerho=timerho+ts1-ts0

contains

pure subroutine rmk1(n,wo,wf1,wf2,rho,mag1,mag2,mag3)
implicit none
! arguments
integer, intent(in) :: n
real(8), intent(in) :: wo
complex(8), intent(in) :: wf1(n),wf2(n)
real(8), intent(inout) :: rho(n),mag1(n),mag2(n),mag3(n)
! local variables
integer i
real(8) wo2,t1,t2
real(8) a1,b1,a2,b2
wo2=2.d0*wo
!$OMP SIMD PRIVATE(a1,b1,a2,b2,t1,t2) SIMDLEN(8)
do i=1,n
  a1=dble(wf1(i)); b1=aimag(wf1(i))
  a2=dble(wf2(i)); b2=aimag(wf2(i))
  t1=a1**2+b1**2; t2=a2**2+b2**2
  mag1(i)=mag1(i)+wo2*(a1*a2+b1*b2)
  mag2(i)=mag2(i)+wo2*(a1*b2-b1*a2)
  mag3(i)=mag3(i)+wo*(t1-t2)
  rho(i)=rho(i)+wo*(t1+t2)
end do
end subroutine

pure subroutine rmk2(n,wo,wf1,wf2,rho,mag)
implicit none
! arguments
integer, intent(in) :: n
real(8), intent(in) :: wo
complex(8), intent(in) :: wf1(n),wf2(n)
real(8), intent(inout) :: rho(n),mag(n)
! local variables
integer i
real(8) t1,t2
!$OMP SIMD PRIVATE(t1,t2) SIMDLEN(8)
do i=1,n
  t1=dble(wf1(i))**2+aimag(wf1(i))**2
  t2=dble(wf2(i))**2+aimag(wf2(i))**2
  mag(i)=mag(i)+wo*(t1-t2)
  rho(i)=rho(i)+wo*(t1+t2)
end do
end subroutine

pure subroutine rmk3(n,wo,wf,rho)
implicit none
! arguments
integer, intent(in) :: n
real(8), intent(in) :: wo
complex(8), intent(in) :: wf(n)
real(8), intent(inout) :: rho(n)
rho(1:n)=rho(1:n)+wo*(dble(wf(1:n))**2+aimag(wf(1:n))**2)
end subroutine

end subroutine
!EOC

