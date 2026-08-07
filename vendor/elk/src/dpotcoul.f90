
! Copyright (C) 2011 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine dpotcoul
use modmain
use modphonon
implicit none
! local variables
integer np
! automatic arrays
complex(8) gzfmt(npmtmax,3)
! solve the complex Poisson's equation in the muffin-tins
dvclmt(1:npmtmax,1:natmtot)=drhomt(1:npmtmax,1:natmtot)
call genzvclmt(nrmt,nrmti,nrmtmax,rlmt,wprmt,npmtmax,dvclmt)
! calculate the gradient of the nuclear potential
call gradzvcln(isph,gzfmt)
! subtract gradient component corresponding to the phonon polarisation
np=npmt(isph)
dvclmt(1:np,iasph)=dvclmt(1:np,iasph)-gzfmt(1:np,ipph)
! solve Poisson's equation in the entire unit cell
dvclir(1:ngtot)=drhoir(1:ngtot)
call zpotcoul(0,nrmt,nrmti,npmt,nrmtmax,rlmt,ngridg,igfft,ngvec,gqc,gclgq, &
 ngvec,jlgqrmt,ylmgq,sfacgq,npmtmax,dvclmt,dvclir)
end subroutine

