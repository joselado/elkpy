
! Copyright (C) 2002-2005 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: bandstr
! !INTERFACE:
subroutine bandstr
! !USES:
use modmain
use modomp
! !DESCRIPTION:
!   Produces a band structure along the path in reciprocal space which connects
!   the vertices in the array {\tt vvlp1d}. The band structure is obtained from
!   the second-variational eigenvalues and is written to the file {\tt BAND.OUT}
!   with the Fermi energy set to zero. If required, band structures are plotted
!   to files {\tt BAND\_Sss\_Aaaaa.OUT} for atom {\tt aaaa} of species {\tt ss},
!   which include the band characters for each $l$ component of that atom in
!   columns 4 onwards. Column 3 contains the sum over $l$ of the characters.
!   Vertex location lines are written to {\tt BANDLINES.OUT}.
!
! !REVISION HISTORY:
!   Created June 2003 (JKD)
!EOP
!BOC
implicit none
! local variables
logical tspndg,tlmdg,tsqaz
integer ik,ist,ispn,idm
integer is,ia,ias,nthd
integer l,m,lm,iv
real(8) emin,emax,sm
real(8) v(ndmag),t1,t2
character(256) fname
complex(8) su2(2,2),z1
! allocatable arrays
real(8), allocatable :: evalfv(:,:)
! low precision for band character array saves memory
real(4), allocatable :: bc(:,:,:,:),elm(:,:)
complex(8), allocatable :: ulm(:,:,:),dmat(:,:,:,:,:)
complex(8), allocatable :: apwalm(:,:,:,:,:),evecfv(:,:,:),evecsv(:,:)
! initialise universal variables
call init0
call init1
! calculate only the diagonal parts of the density matrices by default
tspndg=.true.
tlmdg=.true.
! spin-quantisation axis in +z direction by default
tsqaz=.true.
select case(task)
case(21)
  allocate(bc(0:lmaxdb,natmtot,nstsv,nkpt))
case(22)
  allocate(bc(lmmaxdb,natmtot,nstsv,nkpt))
! generate unitary matrices which convert the Yₗₘ basis into the irreducible
! representation basis of the symmetry group at each atomic site
  if (lmirep) then
    allocate(elm(lmmaxdb,natmtot),ulm(lmmaxdb,lmmaxdb,natmtot))
    call genlmirep(elm,ulm)
! write the eigenvalues of a pseudorandom matrix symmetrised by the site
! symmetries in the Yₗₘ basis
    call writeelmirep('.OUT',elm)
! require full density matrix in (l,m) degrees of freedom
    tlmdg=.false.
  end if
case(23)
  if (.not.spinpol) goto 10
  allocate(bc(nspinor,natmtot,nstsv,nkpt))
! compute the SU(2) operator used for rotating the density matrix to the
! desired spin-quantisation axis; if axis is not z then the full matrix in
! spin degrees of freedom is required
  call sqasu2(sqaxis,tsqaz,su2)
  tspndg=tsqaz
case(24)
  if (.not.spinpol) goto 10
  allocate(bc(ndmag,natmtot,nstsv,nkpt))
  tspndg=.not.ncmag
end select
! read density and potentials from file
call readstate
! Fourier transform Kohn-Sham potential to G-space
call genvsig
! find the new linearisation energies
call linengy
! generate the APW and local-orbital radial functions and integrals
call genapwlofr
! generate the spin-orbit coupling radial functions
call gensocfr
! begin parallel loop over k-points
call holdthd(nkpt,nthd)
!$OMP PARALLEL DEFAULT(SHARED) &
!$OMP PRIVATE(evalfv,evecfv,evecsv) &
!$OMP PRIVATE(dmat,apwalm,ispn,ias,ist) &
!$OMP PRIVATE(l,m,lm,sm,v,t1,t2,z1) &
!$OMP NUM_THREADS(nthd)
allocate(evalfv(nstfv,nspnfv))
allocate(evecfv(nmatmax,nstfv,nspnfv),evecsv(nstsv,nstsv))
if (task >= 21) then
  allocate(dmat(lmmaxdb,nspinor,lmmaxdb,nspinor,nstsv))
  allocate(apwalm(ngkmax,apwordmax,lmmaxapw,natmtot,nspnfv))
end if
!$OMP DO SCHEDULE(DYNAMIC)
do ik=1,nkpt
!$OMP CRITICAL(bandstr_)
  write(*,'("Info(bandstr): ",I0," of ",I0," k-points")') ik,nkpt
!$OMP END CRITICAL(bandstr_)
! solve the first- and second-variational eigenvalue equations
  call eveqn(ik,evalfv,evecfv,evecsv)
! compute the band characters if required
  if (task >= 21) then
! find the matching coefficients
    do ispn=1,nspnfv
      call match(ngk(ispn,ik),vgkc(:,:,ispn,ik),gkc(:,ispn,ik), &
       sfacgk(:,:,ispn,ik),apwalm(:,:,:,:,ispn))
    end do
! average band character over spin and m for all atoms
    do ias=1,natmtot
! generate the density matrix
      call gendmatk(tspndg,tlmdg,0,lmaxdb,ias,nstsv,[0],ngk(:,ik),apwalm, &
       evecfv,evecsv,lmmaxdb,dmat)
! convert (l,m) part to an irreducible representation if required
      if (.not.tlmdg) call dmatulm(ulm(:,:,ias),dmat)
! spin rotate the density matrices to desired spin-quantisation axis if required
      if (.not.tsqaz) call dmatsu2(lmmaxdb,su2,dmat)
      do ist=1,nstsv
        select case(task)
        case(21)
! l character of band
          lm=0
          do l=0,lmaxdb
            sm=0.d0
            do m=-l,l
              lm=lm+1
              do ispn=1,nspinor
                sm=sm+dble(dmat(lm,ispn,lm,ispn,ist))
              end do
            end do
            bc(l,ias,ist,ik)=sm
          end do
        case(22)
! (l,m) character of band
          do lm=1,lmmaxdb
            sm=0.d0
            do ispn=1,nspinor
              sm=sm+dble(dmat(lm,ispn,lm,ispn,ist))
            end do
            bc(lm,ias,ist,ik)=sm
          end do
        case(23)
! spin character of band
          do ispn=1,nspinor
            sm=0.d0
            do lm=1,lmmaxdb
              sm=sm+dble(dmat(lm,ispn,lm,ispn,ist))
            end do
            bc(ispn,ias,ist,ik)=sm
          end do
        case(24)
! magnetic moment character of band
          v(1:ndmag)=0.d0
          do lm=1,lmmaxdb
            t1=dble(dmat(lm,1,lm,1,ist))
            t2=dble(dmat(lm,2,lm,2,ist))
            v(ndmag)=v(ndmag)+t1-t2
            if (ncmag) then
              z1=2.d0*dmat(lm,1,lm,2,ist)
              v(1)=v(1)+z1%re
              v(2)=v(2)-z1%im
            end if
          end do
          bc(1:ndmag,ias,ist,ik)=v(1:ndmag)
        end select
      end do
    end do
  end if
! end loop over k-points
end do
!$OMP END DO
deallocate(evalfv,evecfv,evecsv)
if (task >= 21) deallocate(dmat,apwalm)
!$OMP END PARALLEL
call freethd(nthd)
! subtract the Fermi energy
evalsv(1:nstsv,1:nkpt)=evalsv(1:nstsv,1:nkpt)-efermi
! find the minimum and maximum eigenvalues
emin=minval(evalsv(1:nstsv,1:nkpt))
emax=maxval(evalsv(1:nstsv,1:nkpt))
t1=(emax-emin)*0.5d0
emin=emin-t1
emax=emax+t1
! output the band structure
if (task == 20) then
  open(50,file='BAND.OUT',form='FORMATTED',action='WRITE')
  do ist=1,nstsv
    do ik=1,nkpt
      write(50,'(2G18.10)') dpp1d(ik),evalsv(ist,ik)
    end do
    write(50,*)
  end do
  close(50)
  write(*,*)
  write(*,'("Info(bandstr):")')
  write(*,'(" Band structure plot written to BAND.OUT")')
else
  do ias=1,natmtot
    is=idxis(ias)
    ia=idxia(ias)
    write(fname,'("BAND_S",I2.2,"_A",I4.4,".OUT")') is,ia
    open(50,file=trim(fname),form='FORMATTED',action='WRITE')
    do ist=1,nstsv
      do ik=1,nkpt
        select case(task)
        case(21)
! sum band character over l to find total atomic character
          sm=sum(bc(0:lmaxdb,ias,ist,ik))
          write(50,'(2G18.10,F12.6)',advance='NO') dpp1d(ik),evalsv(ist,ik),sm
          do l=0,lmaxdb
            write(50,'(F12.6)',advance='NO') bc(l,ias,ist,ik)
          end do
          write(50,*)
        case(22)
          write(50,'(2G18.10)',advance='NO') dpp1d(ik),evalsv(ist,ik)
          do lm=1,lmmaxdb
            write(50,'(F12.6)',advance='NO') bc(lm,ias,ist,ik)
          end do
          write(50,*)
        case(23)
          write(50,'(2G18.10,2F12.6)') dpp1d(ik),evalsv(ist,ik), &
           (bc(ispn,ias,ist,ik),ispn=1,nspinor)
        case(24)
          write(50,'(2G18.10,3F12.6)') dpp1d(ik),evalsv(ist,ik), &
           (bc(idm,ias,ist,ik),idm=1,ndmag)
        end select
      end do
      write(50,*)
    end do
    close(50)
  end do
  write(*,*)
  write(*,'("Info(bandstr):")')
  write(*,'(" Band structure plot written to BAND_Sss_Aaaaa.OUT")')
  write(*,'(" for all species and atoms")')
  write(*,*)
  write(*,'(" Columns in the file are :")')
  select case(task)
  case(21)
    write(*,'("  distance, eigenvalue, total atomic character, l character &
     &(l = 0...",I0,")")') lmaxdb
  case(22)
    write(*,'("  distance, eigenvalue, (l,m) character &
     &(l = 0...",I0,", m = -l...l)")') lmaxdb
    if (lmirep) then
      write(*,*)
      write(*,'(" Eigenvalues of a random matrix symmetrised with the site")')
      write(*,'(" symmetries in the Yₗₘ basis written to ELMIREP.OUT for all")')
      write(*,'(" species and atoms. Degenerate eigenvalues correspond to")')
      write(*,'(" irreducible representations of each site symmetry group")')
    end if
  case(23)
    write(*,'("  distance, eigenvalue, spin-up and spin-down characters")')
    write(*,*)
    write(*,'(" Spin-quantisation axis : ",3G18.10)') sqaxis(:)
  case(24)
    write(*,'("  distance, eigenvalue, moment character")')
  end select
end if
write(*,*)
write(*,'(" Fermi energy is at zero in plot")')
! output the vertex location lines
open(50,file='BANDLINES.OUT',form='FORMATTED',action='WRITE')
do iv=1,nvp1d
  write(50,'(2G18.10)') dvp1d(iv),emin
  write(50,'(2G18.10)') dvp1d(iv),emax
  write(50,*)
end do
close(50)
write(*,*)
write(*,'(" Vertex location lines written to BANDLINES.OUT")')
if (task >= 21) deallocate(bc)
if ((task == 22).and.lmirep) deallocate(elm,ulm)
return
10 continue
write(*,*)
write(*,'("Error(bandstr): spin-unpolarised calculation")')
write(*,*)
stop
end subroutine
!EOC

