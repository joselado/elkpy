
! Copyright (C) 2007-2008 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: getevecfv
! !INTERFACE:
subroutine getevecfv(fext,ikp,vpl,vgpl,evecfv)
! !USES:
use modmain
use modramdisk
! !INPUT/OUTPUT PARAMETERS:
!   fext   : filename extension (in,character(*))
!   ikp    : p-point vector index (in,integer)
!   vpl    : p-point vector in lattice coordinates (in,real(3))
!   vgpl   : G+p-vectors in lattice coordinates (out,real(3,ngkmax,nspnfv))
!   evecfv : first-variational eigenvectors (out,complex(nmatmax,nstfv,nspnfv))
! !DESCRIPTION:
!   Reads in a first-variational eigenvector from file. If the input $k$-point,
!   ${\bf p}$, is not in the reduced set, then the eigenvector of the equivalent
!   point is read in and the required rotation/translation operations applied.
!
! !REVISION HISTORY:
!   Created Feburary 2007 (JKD)
!   Fixed transformation error, October 2007 (JKD, Anton Kozhevnikov)
!   Fixed l.o. rotation, June 2010 (A. Kozhevnikov)
!EOP
!BOC
implicit none
! arguments
character(*), intent(in) :: fext
integer, intent(in) :: ikp
real(8), intent(in) :: vpl(3),vgpl(3,ngkmax,nspnfv)
complex(8), intent(out) :: evecfv(nmatmax,nstfv,nspnfv)
! local variables
logical tgs
integer isym,lspl,ilspl
integer jspn,ilo,l,lm,i,j
integer ik,ngk_,igp,igk,ig
integer is,ia,ja,ias,jas
integer recl,nmatmax_,nstfv_,nspnfv_
real(8) vkl_(3),v(3),si(3,3),t1
complex(8) z1
character(256) fname
if (ikp > 0) then
  ik=ikp
else
! find the equivalent k-point number and crystal symmetry element
  call findkpt(vpl,isym,ik)
end if
! construct the filename
fname=trim(scrpath)//'EVECFV'//trim(fext)
!$OMP CRITICAL(u202)
! read from RAM disk if required
if (ramdisk) then
  call getrd(fname,ik,tgs,v1=vkl_,n1=nmatmax_,n2=nstfv_,n3=nspnfv_, &
   nzv=nmatmax*nstfv*nspnfv,zva=evecfv)
  if (tgs) goto 10
end if
! find the record length
inquire(iolength=recl) vkl_,nmatmax_,nstfv_,nspnfv_,evecfv
open(202,file=fname,form='UNFORMATTED',access='DIRECT',recl=recl)
read(202,rec=ik) vkl_,nmatmax_,nstfv_,nspnfv_,evecfv
close(202)
10 continue
!$OMP END CRITICAL(u202)
t1=abs(vkl(1,ik)-vkl_(1))+abs(vkl(2,ik)-vkl_(2))+abs(vkl(3,ik)-vkl_(3))
if (t1 > epslat) then
  write(*,*)
  write(*,'("Error(getevecfv): differing vectors for k-point ",I0)') ik
  write(*,'(" current    : ",3G18.10)') vkl(:,ik)
  write(*,'(" EVECFV.OUT : ",3G18.10)') vkl_
  write(*,*)
  stop
end if
if (nmatmax /= nmatmax_) then
  write(*,*)
  write(*,'("Error(getevecfv): differing nmatmax for k-point ",I0)') ik
  write(*,'(" current    : ",I0)') nmatmax
  write(*,'(" EVECFV.OUT : ",I0)') nmatmax_
  write(*,*)
  stop
end if
if (nstfv /= nstfv_) then
  write(*,*)
  write(*,'("Error(getevecfv): differing nstfv for k-point ",I0)') ik
  write(*,'(" current    : ",I0)') nstfv
  write(*,'(" EVECFV.OUT : ",I0)') nstfv_
  write(*,*)
  stop
end if
if (nspnfv /= nspnfv_) then
  write(*,*)
  write(*,'("Error(getevecfv): differing nspnfv for k-point ",I0)') ik
  write(*,'(" current    : ",I0)') nspnfv
  write(*,'(" EVECFV.OUT : ",I0)') nspnfv_
  write(*,*)
  stop
end if
! if p = k then return
if (ikp > 0) return
t1=abs(vpl(1)-vkl(1,ik))+abs(vpl(2)-vkl(2,ik))+abs(vpl(3)-vkl(3,ik))
if (t1 < epslat) return
block
logical done(ngkmax)
complex(8) evecfv_(nmatmax,nstfv)
! index to spatial rotation in lattice point group
lspl=lsplsymc(isym)
! the inverse of the spatial symmetry rotates k into p
ilspl=isymlat(lspl)
si(1:3,1:3)=dble(symlat(1:3,1:3,ilspl))
!-----------------------------------------------!
!     translate and rotate APW coefficients     !
!-----------------------------------------------!
! loop over the first-variational spins
do jspn=1,nspnfv
  ngk_=ngk(jspn,ik)
  if (tv0symc(isym)) then
! translation vector is zero
    evecfv_(1:ngk_,1:nstfv)=evecfv(1:ngk_,1:nstfv,jspn)
  else
! non-zero translation vector gives a phase factor
    v(1:3)=vtcsymc(1:3,isym)
    do igk=1,ngk_
      ig=igkig(igk,jspn,ik)
      t1=vgc(1,ig)*v(1)+vgc(2,ig)*v(2)+vgc(3,ig)*v(3)
      z1=cmplx(cos(t1),-sin(t1),8)
      evecfv_(igk,1:nstfv)=z1*evecfv(igk,1:nstfv,jspn)
    end do
  end if
! apply spatial rotation operation (passive transformation)
  done(1:ngk_)=.false.
  i=1
  do igk=1,ngk_
    call r3mtv(si,vgkl(:,igk,jspn,ik),v)
    j=0
    do igp=i,ngk_
      if (done(igp)) cycle
      if (j == 0) j=igp
      t1=abs(v(1)-vgpl(1,igp,jspn)) &
        +abs(v(2)-vgpl(2,igp,jspn)) &
        +abs(v(3)-vgpl(3,igp,jspn))
      if (t1 < epslat) then
        evecfv(igp,1:nstfv,jspn)=evecfv_(igk,1:nstfv)
        done(igp)=.true.
        if (igp == j) j=j+1
        exit
      end if
    end do
    if (j > 0) i=j
  end do
end do
!---------------------------------------------------------!
!     translate and rotate local-orbital coefficients     !
!---------------------------------------------------------!
if (nlotot > 0) then
! rotate k-point by inverse symmetry matrix
  call r3mtv(si,vkl(:,ik),v)
! loop over the first-variational spins
  do jspn=1,nspnfv
    ngk_=ngk(jspn,ik)
! make a copy of the local-orbital coefficients
    do i=ngk_+1,nmat(jspn,ik)
      evecfv_(i,1:nstfv)=evecfv(i,1:nstfv,jspn)
    end do
    do is=1,nspecies
      do ia=1,natoms(is)
        ias=idxas(ia,is)
! equivalent atom for this symmetry
        ja=ieqatom(ia,is,isym)
        jas=idxas(ja,is)
! phase factor from translation
        t1=-twopi*dot_product(vkl(1:3,ik),atposl(1:3,ja,is))
        z1=cmplx(cos(t1),sin(t1),8)
        t1=twopi*dot_product(v(1:3),atposl(1:3,ia,is))
        z1=z1*cmplx(cos(t1),sin(t1),8)
! rotate local-orbitals (active transformation)
        do ilo=1,nlorb(is)
          l=lorbl(ilo,is)
          lm=l**2+1
          i=ngk_+idxlo(lm,ilo,ias)
          j=ngk_+idxlo(lm,ilo,jas)
          call rotzflm(symlatc(:,:,lspl),l,l,lolmmax,nstfv,nmatmax, &
           evecfv_(j,1),evecfv(i,1,jspn))
          evecfv(i:i+2*l,1:nstfv,jspn)=z1*evecfv(i:i+2*l,1:nstfv,jspn)
        end do
      end do
    end do
  end do
end if
end block
end subroutine
!EOC

