
! Copyright (C) 2002-2005 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: wfmtfv
! !INTERFACE:
subroutine wfmtfv(ias,ngp,apwalm,evecfv,wfmt)
! !USES:
use modmain
! !INPUT/OUTPUT PARAMETERS:
!   ias    : joint atom and species number (in,integer)
!   ngp    : number of G+p-vectors (in,integer)
!   apwalm : APW matching coefficients (in,complex(ngkmax,apwordmax,lmmaxapw))
!   evecfv : first-variational eigenvector (in,complex(nmatmax))
!   wfmt   : complex muffin-tin wavefunction passed in as real array
!            (out,real(2,*))
! !DESCRIPTION:
!   Calculates the first-variational wavefunction in the muffin-tin in terms of
!   a spherical harmonic expansion. For atom $\alpha$ and a particular $k$-point
!   ${\bf p}$, the $r$-dependent $(l,m)$-coefficients of the wavefunction for
!   the $i$th state are given by
!   $$ \Phi^{i{\bf p}}_{\alpha lm}(r)=\sum_{\bf G}b^{i{\bf p}}_{\bf G}
!    \sum_{j=1}^{M^{\alpha}_l}A^{\alpha}_{jlm}({\bf G+p})u^{\alpha}_{jl}(r)
!    +\sum_{j=1}^{N^{\alpha}}b^{i{\bf p}}_{(\alpha,j,m)}v^{\alpha}_j(r)
!    \delta_{l,l_j}, $$
!   where $b^{i{\bf p}}$ is the $i$th eigenvector returned from routine
!   {\tt eveqn}; $A^{\alpha}_{jlm}({\bf G+p})$ is the matching coefficient;
!   $M^{\alpha}_l$ is the order of the APW; $u^{\alpha}_{jl}$ is the APW radial
!   function; $N^{\alpha}$ is the number of local-orbitals; $v^{\alpha}_j$ is
!   the $j$th local-orbital radial function; and $(\alpha,j,m)$ is a compound
!   index for the location of the local-orbital in the eigenvector. See routines
!   {\tt genapwfr}, {\tt genlofr}, {\tt match} and {\tt eveqn}.
!
! !REVISION HISTORY:
!   Created April 2003 (JKD)
!   Fixed description, October 2004 (C. Brouder)
!   Removed argument ist, November 2006 (JKD)
!   Changed arguments and optimised, December 2014 (JKD)
!EOP
!BOC
implicit none
! arguments
integer, intent(in) :: ias,ngp
complex(8), intent(in) :: apwalm(ngkmax,apwordmax,lmmaxapw),evecfv(nmatmax)
complex(8), intent(out) :: wfmt(*)
! local variables
integer is,io,ilo,ld
integer nrci,nrco,iro
integer l,lma,nm,npci
! automatic arrays
complex(8) z(2*lmaxo+1)
is=idxis(ias)
iro=nrmti(is)+lradstp
nrci=nrcmti(is)
nrco=nrcmt(is)-nrci
npci=npcmti(is)
! zero the wavefunction
wfmt(1:npcmt(is))=0.d0
!---------------------------------!
!     local-orbital functions     !
!---------------------------------!
do ilo=1,nlorb(is)
  l=lorbl(ilo,is)
  nm=2*l+1
  lma=l**2+1
  z(1:nm)=evecfv(ngp+idxlo(lma:lma+2*l,ilo,ias))
  if (l <= lmaxi) call zfzrf(nm,nrci,lofr(1,1,ilo,ias),z,lmmaxi,wfmt(lma))
  call zfzrf(nm,nrco,lofr(iro,1,ilo,ias),z,lmmaxo,wfmt(npci+lma))
end do
!-----------------------!
!     APW functions     !
!-----------------------!
ld=ngkmax*apwordmax
do l=0,lmaxo
  nm=2*l+1
  lma=l**2+1
  do io=1,apword(l,is)
    call zgemv('T',ngp,nm,zone,apwalm(:,io,lma),ld,evecfv,1,zzero,z,1)
    if (l <= lmaxi) call zfzrf(nm,nrci,apwfr(1,1,io,l,ias),z,lmmaxi,wfmt(lma))
    call zfzrf(nm,nrco,apwfr(iro,1,io,l,ias),z,lmmaxo,wfmt(npci+lma))
  end do
end do

contains

pure subroutine zfzrf(m,n,rf,z,ld,zf)
implicit none
! arguments
integer, intent(in) :: m,n
real(8), intent(in) :: rf(lradstp,n)
complex(8), intent(in) :: z(m)
integer, intent(in) :: ld
complex(8), intent(inout) :: zf(ld,n)
! local variables
integer i
do i=1,m
  zf(i,1:n)=zf(i,1:n)+z(i)*rf(1,1:n)
end do
end subroutine

end subroutine
!EOC

