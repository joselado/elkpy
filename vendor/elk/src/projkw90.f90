
! Copyright (C) 2026 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine projkw90(ik,amn)
use modmain
use modw90
implicit none
! arguments
integer, intent(in) :: ik
complex(8), intent(out) :: amn(num_bands,num_wann)
! local variables
integer ispn,ist,is,ias
integer nrc,nrci,irco
integer l,m,lm,i,j,k
integer npci,ni,no
real(8) sm,t0,t1
complex(8) f(-3:3),g(7),zsm
! allocatable arrays
complex(8), allocatable :: apwalm(:,:,:,:,:),evecfv(:,:,:),evecsv(:,:)
complex(8), allocatable :: wfmt(:,:,:)
! allocate local arrays
allocate(apwalm(ngkmax,apwordmax,lmmaxapw,natmtot,nspnfv))
allocate(evecfv(nmatmax,nstfv,nspnfv),evecsv(nstsv,nstsv))
allocate(wfmt(npcmtmax,nspinor,num_bands))
! find the matching coefficients
do ispn=1,nspnfv
  call match(ngk(ispn,ik),vgkc(:,:,ispn,ik),gkc(:,ispn,ik),sfacgk(:,:,ispn,ik),&
   apwalm(:,:,:,:,ispn))
end do
! get the eigenvectors from file
call getevecfv(filext,0,vkl(:,ik),vgkl(:,:,:,ik),evecfv)
call getevecsv(filext,0,vkl(:,ik),evecsv)
! loop over atoms
do ias=1,natmtot
  is=idxis(ias)
  nrc=nrcmt(is)
  nrci=nrcmti(is)
  irco=nrci+1
  npci=npcmti(is)
  ni=npci-1
  no=npcmt(is)-npci-1
! normalisation prefactor
  t0=1.d0/sqrt((fourpi/3.d0)*rmt(is)**3)
! generate the second-variational muffin-tin wavefunctions at k
  call wfmtsv(.true.,lradstp,is,ias,num_bands,idxw90,ngk(:,ik),apwalm,evecfv, &
   evecsv,npcmtmax,wfmt)
  do ist=1,num_bands
    do ispn=1,nspinor
      do i=1,nprojw90(is)
        l=lprojw90(i,is)
        lm=l**2
        do m=-l,l
          lm=lm+1
          if (l <= lmaxi) then
            sm=sum(abs(wfmt(lm:lm+ni:lmmaxi,ispn,ist))*wr2cmt(1:nrci,is))
            zsm=sum(wfmt(lm:lm+ni:lmmaxi,ispn,ist)*wr2cmt(1:nrci,is))
          else
            sm=0.d0
            zsm=0.d0
          end if
          j=npci+lm
          sm=sm+sum(abs(wfmt(j:j+no:lmmaxo,ispn,ist))*wr2cmt(irco:nrc,is))
          zsm=zsm+sum(wfmt(j:j+no:lmmaxo,ispn,ist)*wr2cmt(irco:nrc,is))
! use sm as the magnitude and zsm as the phase of the projector
          t1=abs(zsm)
          if (t1 > 1.d-12) then
            f(m)=t0*sm*zsm/t1
          else
            f(m)=t0*sm
          end if
        end do
! convert to cubic harmonics in the Wannier90 convention
        call ftogw90(l,f,g)
! store in the output matrix
        j=prjwn(ispn,i,ias)
        do k=1,2*l+1
          amn(ist,j)=g(k)
          j=j+nspinor
        end do
      end do
    end do
  end do
end do
deallocate(apwalm,evecfv,evecsv,wfmt)

contains

pure subroutine ftogw90(l,f,g)
implicit none
! arguments
integer, intent(in) :: l
complex(8), intent(in) :: f(-3:3)
complex(8), intent(out) :: g(7)
! real constant 1/√2
real(8), parameter :: c1=0.7071067811865475244d0
! s, pz, dz2, fz3
g(1)=f(0)
if (l == 0) return
! px, dxz, fxz2
g(2)=c1*(f(-1)-f(1))
! py, dyz, fyz2
g(3)=-c1*zi*(f(-1)+f(1))
if (l == 1) return
! dx2-y2, fz(x2-y2)
g(4)=c1*(f(-2)+f(2))
! dxy, fxyz
g(5)=c1*zi*(f(2)-f(-2))
if (l == 2) return
! fx(x2-3y2)
g(6)=c1*(f(-3)-f(3))
! fy(3x2-y2)
g(7)=-c1*zi*(f(-3)+f(3))
end subroutine

end subroutine

