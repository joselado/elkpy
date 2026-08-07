
! Copyright (C) 2024 Wenhan Chen, J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine bandstrulr
use modmain
use modulr
use modmpi
use modomp
use moddelf
implicit none
! local variables
integer nkpa0,ikpa,nthd
integer ik0,ist,iv,iw,lp
real(8) dw,emin,emax,t1
! allocatable arrays
real(8), allocatable :: vvlp1d0(:,:),vql0(:,:)
real(8), allocatable :: w(:),chkpa(:,:),sfu(:,:)
complex(8), allocatable :: evecu(:,:)
! store the 1D plot vertices
allocate(vvlp1d0(3,nvp1d))
vvlp1d0(1:3,1:nvp1d)=vvlp1d(1:3,1:nvp1d)
! initialise global variables
call init0
call init1
if (task == 720) then
! use only κ = 0
  nkpa0=1
else
! use all κ-points
  nkpa0=nkpa
end if
! store the κ-points
allocate(vql0(3,nkpa0))
vql0(1:3,1:nkpa0)=vql(1:3,1:nkpa0)
! generate frequency grid
allocate(w(nwplot))
dw=(wplot(2)-wplot(1))/dble(nwplot)
do iw=1,nwplot
  w(iw)=dw*dble(iw-1)+wplot(1)
end do
allocate(chkpa(nstsv*nkpa,nkpt0))
! allocate and zero the ULR spectral function
allocate(sfu(nwplot,nkpt0))
sfu(1:nwplot,1:nkpt0)=0.d0
! delete the BANDULR.OUT file
call delfile('BANDULR.OUT')
! loop over the κ-points
do ikpa=1,nkpa0
  if (mp_mpi) then
    write(*,*)
    write(*,'("Info(bandstrulr): ",I0," of ",I0," κ-points")') ikpa,nkpa0
    write(*,*)
  end if
! subtract current κ-point from 1D plot vertices
  do iv=1,nvp1d
    vvlp1d(1:3,iv)=vvlp1d0(1:3,iv)-vql0(1:3,ikpa)
  end do
  call init0
  call init1
  call readstate
  call genvsig
  call gencore
  call linengy
  call genapwlofr
  call gensocfr
  call genevfsv
  call occupy
  call initulr
! read in the potential STATE_ULR.OUT
  call readstulr
! initialise the external Coulomb potential
  call vclqinit
! apply required local operations to the potential and magnetic field
  call vblocalu
! loop over original k-points
  call holdthd(nkpt0/np_mpi,nthd)
!$OMP PARALLEL DEFAULT(SHARED) &
!$OMP PRIVATE(evecu) &
!$OMP NUM_THREADS(nthd)
  allocate(evecu(nstulr,nstulr))
!$OMP DO SCHEDULE(DYNAMIC)
  do ik0=1,nkpt0
! distribute among MPI processes
    if (mod(ik0-1,np_mpi) /= lp_mpi) cycle
!$OMP CRITICAL(bandstrulr_)
    write(*,'("Info(bandstrulr): ",I0," of ",I0," k-points")') ik0,nkpt0
!$OMP END CRITICAL(bandstrulr_)
! solve the ultra long-range eigenvalue equation
    call eveqnulr(ik0,evecu)
! determine the current κ-point characteristic for each ULR state
    call charkpa(ikpa,evecu,chkpa(:,ik0))
! add to the ULR spectral function
    call sfuadd(ik0,w,chkpa(:,ik0),sfu(:,ik0))
  end do
!$OMP END DO
  deallocate(evecu)
!$OMP END PARALLEL
  call freethd(nthd)
! broadcast arrays to every process
  if (np_mpi > 1) then
    do ik0=1,nkpt0
      lp=mod(ik0-1,np_mpi)
      call mpi_bcast(evalu(:,ik0),nstulr,mpi_double_precision,lp,mpicom,ierror)
      call mpi_bcast(chkpa(:,ik0),nstulr,mpi_double_precision,lp,mpicom,ierror)
    end do
  end if
! subtract the Fermi energy
  evalu(:,:)=evalu(:,:)-efermi
  if (mp_mpi) then
! output the band structure
    open(50,file='BANDULR.OUT',form='FORMATTED',action='WRITE', &
     position='APPEND')
    do ist=1,nstulr
      do ik0=1,nkpt0
        write(50,'(3G18.10)') dpp1d(ik0),evalu(ist,ik0),chkpa(ist,ik0)
      end do
      write(50,*)
    end do
    close(50)
! output the vertex location lines
    if (ikpa == 1) then
! find the minimum and maximum eigenvalues
      emin=minval(evalu(:,:))
      emax=maxval(evalu(:,:))
      open(50,file='BANDLINES.OUT',form='FORMATTED',action='WRITE')
      do iv=1,nvp1d
        write(50,'(2G18.10)') dvp1d(iv),emin
        write(50,'(2G18.10)') dvp1d(iv),emax
        write(50,*)
      end do
      close(50)
    end if
  end if
! synchronise MPI processes
  call mpi_barrier(mpicom,ierror)
end do
! add the spectral function from each process and redistribute
call mpi_allreduce(mpi_in_place,sfu,nwplot*nkpt0,mpi_double_precision,mpi_sum, &
 mpicom,ierror)
! normalise
t1=1.d0/nkpa0
sfu(1:nwplot,1:nkpt0)=t1*sfu(1:nwplot,1:nkpt0)
! write spectral function band structure
if (mp_mpi) then
  open(50,file='BANDSFU.OUT',form='FORMATTED')
  write(50,'(2I6," : grid size")') nkpt0,nwplot
  do iw=1,nwplot
    do ik0=1,nkpt0
      write(50,'(3G18.10)') dpp1d(ik0),w(iw),sfu(iw,ik0)
    end do
  end do
  close(50)
  write(*,*)
  write(*,'("Info(bandstrulr):")')
  write(*,'(" Ultra long-range band structure plot written to BANDULR.OUT")')
  write(*,'(" Plotted k-point character written in third column")')
  write(*,*)
  write(*,'(" Vertex location lines written to BANDLINES.OUT")')
  write(*,*)
  write(*,'(" Ultra long-range spectral function band structure written to &
   &BANDSFU.OUT")')
end if
deallocate(vvlp1d0,vql0,w,chkpa,sfu)
end subroutine

