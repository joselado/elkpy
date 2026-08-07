
! Copyright (C) 2024 Peter Elliott, J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine writexrmk
use modmain
use modtddft
use modmpi
use modomp
use moddelf
implicit none
! local variables
integer ik,jk,ist,jst
integer i1,i2,i3,i,lp,nthd
real(8) xchg,t1,t2
complex(8) z1
! automatic arrays
real(8) xrk(nkptnr),xmk(ndmag,nkptnr)
complex(8) zv(nstsv)
character(32) fext
! allocatable arrays
complex(8), allocatable :: evecsv(:,:),evecsvt(:,:)
! external functions
complex(8), external :: zdotc
! zero the excited density and magnetisation
xrk(:)=0.d0
if (spinpol) xmk(:,:)=0.d0
call holdthd(nkptnr/np_mpi,nthd)
!$OMP PARALLEL DEFAULT(SHARED) &
!$OMP PRIVATE(evecsv,evecsvt,zv) &
!$OMP PRIVATE(jk,ist,jst,z1,i,t1,t2) &
!$OMP NUM_THREADS(nthd)
allocate(evecsv(nstsv,nstsv),evecsvt(nstsv,nstsv))
!$OMP DO SCHEDULE(DYNAMIC)
do ik=1,nkptnr
! distribute among MPI processes
  if (mod(ik-1,np_mpi) /= lp_mpi) cycle
! equivalent reduced k-point
  jk=ivkik(ivk(1,ik),ivk(2,ik),ivk(3,ik))
! read in ground-state eigenvectors
  call getevecsv('.OUT',0,vkl(:,ik),evecsv)
! read in time-dependent Kohn-Sham eigenvectors (first-variational basis)
  call getevecsv(filext,0,vkl(:,ik),evecsvt)
  do ist=1,nstsv
    if (evalsv(ist,jk) > efermi) cycle
! project time-dependent Kohn-Sham state onto ground-state conduction bands
    zv(1:nstsv)=0.d0
    do jst=1,nstsv
      if (evalsv(jst,jk) < efermi) cycle
      z1=zdotc(nstsv,evecsv(:,jst),1,evecsvt(:,ist),1)
      zv(1:nstsv)=zv(1:nstsv)+z1*evecsv(1:nstsv,jst)
    end do
! calculate the excited density and magnetisation
    if (spinpol) then
      i=nstfv+1
      t1=dble(zdotc(nstfv,zv,1,zv,1))
      t2=dble(zdotc(nstfv,zv(i),1,zv(i),1))
      xrk(ik)=xrk(ik)+t1+t2
      xmk(ndmag,ik)=xmk(ndmag,ik)+t1-t2
      if (ncmag) then
        z1=zdotc(nstfv,zv,1,zv(i),1)
        xmk(1,ik)=xmk(1,ik)+2.d0*z1%re
        xmk(2,ik)=xmk(2,ik)+2.d0*z1%im
      end if
    else
      xrk(ik)=xrk(ik)+occmax*dble(zdotc(nstsv,zv,1,zv,1))
    end if
  end do
end do
!$OMP END DO
deallocate(evecsv,evecsvt)
!$OMP END PARALLEL
call freethd(nthd)
! broadcast arrays to every MPI process
if (np_mpi > 1) then
  do ik=1,nkptnr
    lp=mod(ik-1,np_mpi)
    call mpi_bcast(xrk(ik),1,mpi_double_precision,lp,mpicom,ierror)
    call mpi_bcast(xmk(:,ik),3,mpi_double_precision,lp,mpicom,ierror)
  end do
end if
! compute the excited charge
xchg=sum(wkpt(1:nkptnr)*xrk(1:nkptnr))
if (mp_mpi) then
! file extension
  write(fext,'("_TS",I8.8,".OUT")') itimes
! write k-point dependent excited density and magnetisation to file
  open(50,file='XRHOK'//trim(fext),form='FORMATTED',action='WRITE')
  if (spinpol) then
    open(51,file='XMAGK'//trim(fext),form='FORMATTED',action='WRITE')
  end if
  do i3=0,ngridk(3)-1
    do i2=0,ngridk(2)-1
      do i1=0,ngridk(1)-1
        ik=ivkiknr(i1,i2,i3)
        write(50,'(I6,4G18.10)') ik,vkl(:,ik),xrk(ik)
        if (spinpol) then
          write(51,'(I6,6G18.10)') ik,vkl(:,ik),xmk(:,ik)
        end if
      end do
    end do
  end do
  close(50)
  if (spinpol) close(51)
! write the excited charge to file
  if (itimes <= 1) call delfile('XCHARGE_TD.OUT')
  open(50,file='XCHARGE_TD.OUT',form='FORMATTED',position='APPEND')
  write(50,'(2G18.10)') times(itimes),xchg
  close(50)
end if
! synchronise MPI processes
call mpi_barrier(mpicom,ierror)
end subroutine

