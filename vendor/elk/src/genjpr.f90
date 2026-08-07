
! Copyright (C) 2018 S. Sharma, J. K. Dewhurst and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine genjpr
use modmain
use modmpi
use modomp
implicit none
! local variables
integer ik,is,ias,n,i,nthd
! allocatable arrays
real(8), allocatable :: jrmt_(:,:,:),jrir_(:,:)
! current density cannot be computed if wavefunctions do not exist
if (iscl < 1) then
  jrmt(:,:,:)=0.d0
  jrir(:,:)=0.d0
  return
end if
! set the current density to zero
allocate(jrmt_(npcmtmax,natmtot,3),jrir_(ngtc,3))
jrmt_(:,:,:)=0.d0
jrir_(:,:)=0.d0
call holdthd(nkpt/np_mpi,nthd)
!$OMP PARALLEL DO DEFAULT(SHARED) &
!$OMP REDUCTION(+:jrmt_,jrir_) &
!$OMP SCHEDULE(STATIC) &
!$OMP NUM_THREADS(nthd)
do ik=1,nkpt
! distribute among MPI processes
  if (mod(ik-1,np_mpi) /= lp_mpi) cycle
  call genjprk(ik,jrmt_,jrir_)
end do
!$OMP END PARALLEL DO
call freethd(nthd)
! add current densities from each process and redistribute
if (np_mpi > 1) then
  n=npcmtmax*natmtot*3
  call mpi_allreduce(mpi_in_place,jrmt_,n,mpi_double_precision,mpi_sum,mpicom, &
   ierror)
  n=ngtc*3
  call mpi_allreduce(mpi_in_place,jrir_,n,mpi_double_precision,mpi_sum,mpicom, &
   ierror)
end if
! copy current density to global arrays
do i=1,3
  do ias=1,natmtot
    is=idxis(ias)
    jrmt(1:npcmt(is),ias,i)=jrmt_(1:npcmt(is),ias,i)
  end do
end do
jrir(1:ngtc,1:3)=jrir_(1:ngtc,1:3)
deallocate(jrmt_,jrir_)
! convert muffin-tin current density to spherical harmonics
call holdthd(natmtot,nthd)
!$OMP PARALLEL DEFAULT(SHARED) &
!$OMP PRIVATE(ias,is,i) &
!$OMP NUM_THREADS(nthd)
do i=1,3
!$OMP DO SCHEDULE(DYNAMIC)
  do ias=1,natmtot
    is=idxis(ias)
    call rfshtip(nrcmt(is),nrcmti(is),jrmt(:,ias,i))
  end do
!$OMP END DO NOWAIT
end do
!$OMP END PARALLEL
call freethd(nthd)
! symmetrise the current density
call symrvf(.false.,.true.,nrcmt,nrcmti,npcmt,ngdgc,ngtc,ngvc,nfgrzc,igfc, &
 igrzfc,npmtmax,jrmt,ngtot,jrir)
! convert muffin-tin and interstitial current density from coarse to fine grids
call holdthd(6,nthd)
!$OMP PARALLEL DEFAULT(SHARED) &
!$OMP NUM_THREADS(nthd)
!$OMP DO SCHEDULE(DYNAMIC)
do i=1,3
  call rfmtctof(jrmt(:,:,i))
end do
!$OMP END DO NOWAIT
!$OMP DO SCHEDULE(DYNAMIC)
do i=1,3
  call rfirctof(jrir(:,i),jrir(:,i))
end do
!$OMP END DO
!$OMP END PARALLEL
call freethd(nthd)
end subroutine

