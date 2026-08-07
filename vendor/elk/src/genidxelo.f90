
! Copyright (C) 2025 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine genidxelo
use modmain
implicit none
! local variables
integer is,n,i,j,k,l
! automatic arrays
integer idx(nlomax)
real(8) e0(nlomax)
! generate the index which arranges the local-orbitals in ascending energy
do is=1,nspecies
  n=nlorb(is)
  do i=1,n
    e0(i)=minval(lorbe0(1:lorbord(i,is),i,is))
    idx(i)=i
  end do
  do i=1,n
    k=i
    do j=i+1,n
      if (e0(idx(j)) < e0(idx(k))) k=j
    end do
    if (k /= i) then
      l=idx(i)
      idx(i)=idx(k)
      idx(k)=l
    end if
  end do
  idxelo(1:n,is)=idx(1:n)
end do
end subroutine

