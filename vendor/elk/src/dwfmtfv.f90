
! Copyright (C) 2013 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine dwfmtfv(ias,ngp,ngpq,apwalmq,dapwalm,evecfv,devecfv,dwfmt)
use modmain
use modphonon
implicit none
! arguments
integer, intent(in) :: ias,ngp,ngpq
complex(8), intent(in) :: apwalmq(ngkmax,apwordmax,lmmaxapw)
complex(8), intent(in) :: dapwalm(ngkmax,apwordmax,lmmaxapw)
complex(8), intent(in) :: evecfv(nmatmax),devecfv(nmatmax)
complex(8), intent(out) :: dwfmt(*)
! local variables
integer is,io,ilo
integer nrci,nrco,iro
integer l,lm,npci,i
complex(8) z
! external functions
complex(8), external :: zdotu
is=idxis(ias)
iro=nrmti(is)+lradstp
nrci=nrcmti(is)
nrco=nrcmt(is)-nrci
npci=npcmti(is)
! zero the wavefunction derivative
dwfmt(1:npcmt(is))=0.d0
!---------------------------------!
!     local-orbital functions     !
!---------------------------------!
do ilo=1,nlorb(is)
  l=lorbl(ilo,is)
  do lm=l**2+1,(l+1)**2
    i=npci+lm
    z=devecfv(ngpq+idxlo(lm,ilo,ias))
    if (l <= lmaxi) then
      call zfzrf(nrci,lradstp,lofr(1,1,ilo,ias),lmmaxi,dwfmt(lm))
    end if
    call zfzrf(nrco,lradstp,lofr(iro,1,ilo,ias),lmmaxo,dwfmt(i))
  end do
end do
!-----------------------!
!     APW functions     !
!-----------------------!
do l=0,lmaxo
  do lm=l**2+1,(l+1)**2
    i=npci+lm
    do io=1,apword(l,is)
      z=zdotu(ngpq,devecfv,1,apwalmq(:,io,lm),1)
      if (ias == iasph) then
        z=z+zdotu(ngp,evecfv,1,dapwalm(:,io,lm),1)
      end if
      if (l <= lmaxi) then
        call zfzrf(nrci,lradstp,apwfr(1,1,io,l,ias),lmmaxi,dwfmt(lm))
      end if
      call zfzrf(nrco,lradstp,apwfr(iro,1,io,l,ias),lmmaxo,dwfmt(i))
    end do
  end do
end do

contains

pure subroutine zfzrf(n,ld1,rf,ld2,zf)
implicit none
! arguments
integer, intent(in) :: n
integer, intent(in) :: ld1
real(8), intent(in) :: rf(ld1,n)
integer, intent(in) :: ld2
complex(8), intent(inout) :: zf(ld2,n)
zf(1,1:n)=zf(1,1:n)+z*rf(1,1:n)
end subroutine

end subroutine

