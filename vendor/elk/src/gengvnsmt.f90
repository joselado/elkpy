
! Copyright (C) 2020 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine gengvnsmt
use modmain
use modtddft
implicit none
! local variables
integer is,ias,l
integer nr,nri,iro
integer np,i0,i1
! automatic arrays
complex(8) zvclmt(npmtmax),gzfmt(npmtmax,3)
! allocate global array
if (allocated(gvnsmt)) deallocate(gvnsmt)
allocate(gvnsmt(npmtmax,3,natmtot))
! loop over atoms
do ias=1,natmtot
  is=idxis(ias)
  nr=nrmt(is)
  nri=nrmti(is)
  iro=nri+1
  np=npmt(is)
  do l=1,3
! convert static density to complex spherical harmonic expansion
    call rtozfmt(nr,nri,rhosmt(:,ias,l),zvclmt)
! solve Poisson's equation in the muffin-tin
    call zpotclmt(nr,nri,nrmtmax,rlmt(:,:,is),wprmt(:,:,is),zvclmt)
! add the nuclear Coulomb potential
    i1=lmmaxi*(nri-1)+1
    zvclmt(1:i1:lmmaxi)=zvclmt(1:i1:lmmaxi)+vcln(1:nri,is)
    i0=i1+lmmaxi
    i1=lmmaxo*(nr-iro)+i0
    zvclmt(i0:i1:lmmaxo)=zvclmt(i0:i1:lmmaxo)+vcln(iro:nr,is)
! compute the gradient of the potential and store in global array
    call gradzfmt(nr,nri,rlmt(:,-1,is),wcrmt(:,:,is),zvclmt,npmtmax,gzfmt)
    gvnsmt(1:np,l,ias)=gzfmt(1:np,l)
  end do
end do
end subroutine

