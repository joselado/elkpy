
! Copyright (C) 2020 J. K. Dewhurst and S. Sharma.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine rhomagv
use modmain
use modmpi
use modomp
implicit none
! local variables
integer ik,ispn,idm
integer is,ias,n,nthd
! allocatable arrays
real(8), allocatable :: rhomt_(:,:),rhoir_(:),magmt_(:,:,:),magir_(:,:)
complex(8), allocatable :: apwalm(:,:,:,:,:),evecfv(:,:,:),evecsv(:,:)
! set the charge density and magnetisation to zero
allocate(rhomt_(npcmtmax,natmtot),rhoir_(ngtc))
rhomt_(:,:)=0.d0
rhoir_(:)=0.d0
if (spinpol) then
  allocate(magmt_(npcmtmax,natmtot,ndmag),magir_(ngtc,ndmag))
else
  allocate(magmt_(1,1,1),magir_(1,1))
end if
magmt_(:,:,:)=0.d0
magir_(:,:)=0.d0
call holdthd(nkpt/np_mpi,nthd)
!$OMP PARALLEL DEFAULT(SHARED) &
!$OMP PRIVATE(apwalm,evecfv,evecsv,ispn) &
!$OMP REDUCTION(+:rhomt_,rhoir_,magmt_,magir_) &
!$OMP NUM_THREADS(nthd)
allocate(apwalm(ngkmax,apwordmax,lmmaxapw,natmtot,nspnfv))
allocate(evecfv(nmatmax,nstfv,nspnfv),evecsv(nstsv,nstsv))
!$OMP DO SCHEDULE(STATIC)
do ik=1,nkpt
! distribute among MPI processes
  if (mod(ik-1,np_mpi) /= lp_mpi) cycle
! get the eigenvectors from file
  call getevecfv(filext,ik,vkl(:,ik),vgkl(:,:,:,ik),evecfv)
  call getevecsv(filext,ik,vkl(:,ik),evecsv)
! find the matching coefficients
  do ispn=1,nspnfv
    call match(ngk(ispn,ik),vgkc(:,:,ispn,ik),gkc(:,ispn,ik), &
     sfacgk(:,:,ispn,ik),apwalm(:,:,:,:,ispn))
  end do
! add to the density and magnetisation
  call rhomagk(ngk(:,ik),igkig(:,:,ik),wkpt(ik),occsv(:,ik),apwalm,evecfv, &
   evecsv,rhomt_,rhoir_,magmt_,magir_)
end do
!$OMP END DO
deallocate(apwalm,evecfv,evecsv)
!$OMP END PARALLEL
call freethd(nthd)
! add density from each process and redistribute
if (np_mpi > 1) then
  n=npcmtmax*natmtot
  call mpi_allreduce(mpi_in_place,rhomt_,n,mpi_double_precision,mpi_sum,mpicom,&
   ierror)
  call mpi_allreduce(mpi_in_place,rhoir_,ngtc,mpi_double_precision,mpi_sum, &
   mpicom,ierror)
  if (spinpol) then
    n=n*ndmag
    call mpi_allreduce(mpi_in_place,magmt_,n,mpi_double_precision,mpi_sum, &
     mpicom,ierror)
    n=ngtc*ndmag
    call mpi_allreduce(mpi_in_place,magir_,n,mpi_double_precision,mpi_sum, &
     mpicom,ierror)
  end if
end if
! copy density to the global arrays
do ias=1,natmtot
  is=idxis(ias)
  rhomt(1:npcmt(is),ias)=rhomt_(1:npcmt(is),ias)
end do
rhoir(1:ngtc)=rhoir_(1:ngtc)
if (spinpol) then
  do idm=1,ndmag
    do ias=1,natmtot
      is=idxis(ias)
      magmt(1:npcmt(is),ias,idm)=magmt_(1:npcmt(is),ias,idm)
    end do
    magir(1:ngtc,idm)=magir_(1:ngtc,idm)
  end do
end if
deallocate(rhomt_,rhoir_,magmt_,magir_)
! convert muffin-tin density/magnetisation to spherical harmonics
call rhomagsh
call holdthd(2,nthd)
!$OMP PARALLEL SECTIONS DEFAULT(SHARED) &
!$OMP PRIVATE(idm) &
!$OMP NUM_THREADS(nthd)
!$OMP SECTION
! symmetrise the density
call symrf(nrcmt,nrcmti,npcmt,ngdgc,ngtc,ngvc,nfgrzc,igfc,igrzfc,npmtmax,rhomt,&
 rhoir)
! convert the muffin-tin density from coarse to fine radial mesh
call rfmtctof(rhomt)
! convert the interstitial density from coarse to fine grid
call rfirctof(rhoir,rhoir)
!$OMP SECTION
if (spinpol) then
! symmetrise the magnetisation
  call symrvf(.true.,ncmag,nrcmt,nrcmti,npcmt,ngdgc,ngtc,ngvc,nfgrzc,igfc, &
   igrzfc,npmtmax,magmt,ngtot,magir)
! convert the muffin-tin magnetisation from coarse to fine radial mesh
  do idm=1,ndmag
    call rfmtctof(magmt(:,:,idm))
  end do
! convert the interstitial magnetisation from coarse to fine grid
  do idm=1,ndmag
    call rfirctof(magir(:,idm),magir(:,idm))
  end do
end if
!$OMP END PARALLEL SECTIONS
call freethd(nthd)
end subroutine

