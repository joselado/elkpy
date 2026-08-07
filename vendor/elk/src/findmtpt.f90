
! Copyright (C) 2025 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine findmtpt(vrl,ias,ir,v,r)
use modmain
implicit none
! arguments
real(8), intent(in) :: vrl(3)
integer, intent(out) :: ias,ir
real(8), intent(out) :: v(3),r
! local variables
integer is,ia,nr,i1,i2,i3
real(8) rmn,rmt2,r2,t1
real(8) v1(3),v2(3),v3(3),v4(3)
v2(1:3)=vrl(1:3)
call r3frac(epslat,v2)
! convert point to Cartesian coordinates
v1(1:3)=v2(1)*avec(1:3,1)+v2(2)*avec(1:3,2)+v2(3)*avec(1:3,3)
! check if point is in a muffin-tin
do is=1,nspecies
  nr=nrmt(is)
  rmn=rminsp(is)
  rmt2=rmt(is)**2
  t1=dble(nr-1)/log(rmt(is)/rmn)
  do ia=1,natoms(is)
    ias=idxas(ia,is)
    v2(1:3)=v1(1:3)-atposc(1:3,ia,is)
    do i1=-1,1
      v3(1:3)=v2(1:3)+dble(i1)*avec(1:3,1)
      do i2=-1,1
        v4(1:3)=v3(1:3)+dble(i2)*avec(1:3,2)
        do i3=-1,1
          v(1:3)=v4(1:3)+dble(i3)*avec(1:3,3)
          r2=v(1)**2+v(2)**2+v(3)**2
          if (r2 < rmt2) then
            r=sqrt(r2)
            if (r > rmn) then
              ir=nint(t1*log(r/rmn))+1
              if (ir > nr) ir=nr
            else
              ir=1
            end if
            return
          end if
        end do
      end do
    end do
  end do
end do
ir=0
end subroutine

