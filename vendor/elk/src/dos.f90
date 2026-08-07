
! Copyright (C) 2002-2008 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: dos
! !INTERFACE:
subroutine dos(fext,tocc,occsvp)
! !USES:
use modmain
use modomp
use modtest
! !INPUT/OUTPUT PARAMETERS:
!   fext   : filename extension (in,character(*))
!   tocc   : .true. if just the occupied orbitals should contribute to the DOS
!            (in,logical)
!   occsvp : occupation numbers of second-variational orbitals
!            (in,real(nstsv,nkpt))
! !DESCRIPTION:
!   Produces a total and partial density of states (DOS) for plotting. The total
!   DOS is written to the file {\tt TDOS.OUT} while the partial DOS is written
!   to the file {\tt PDOS\_Sss\_Aaaaa.OUT} for atom {\tt aaaa} of species
!   {\tt ss}. In the case of the partial DOS, each symmetrised
!   $(l,m)$-projection is written consecutively and separated by blank lines.
!   If the global variable {\tt lmirep} is {\tt .true.}, then the density matrix
!   from which the $(l,m)$-projections are obtained is first rotated into a
!   irreducible representation basis, i.e. one that block diagonalises all the
!   site symmetry matrices in the $Y_{lm}$ basis. Eigenvalues of a quasi-random
!   matrix in the $Y_{lm}$ basis which has been symmetrised with the site
!   symmetries are written to {\tt ELMIREP.OUT}. This allows for identification
!   of the irreducible representations of the site symmetries, for example $e_g$
!   or $t_{2g}$, by the degeneracies of the eigenvalues. In the plot, spin-up is
!   made positive and spin-down negative. See the routines {\tt gendmatk} and
!   {\tt brzint}.
!
! !REVISION HISTORY:
!   Created January 2004 (JKD)
!   Parallelised and included sum over m, November 2009 (F. Cricchio)
!EOP
!BOC
implicit none
! arguments
character(*), intent(in) :: fext
logical, intent(in) :: tocc
real(8), intent(in) :: occsvp(nstsv,nkpt)
! local variables
logical tspndg,tlmdg
integer nsk(3),ik,jk,ist,iw
integer nsd,ispn,sps(2)
integer is,ia,ias,nthd
integer l0,l1,l,lm
real(8) dw,vl(3),vc(3)
complex(8) su2(2,2),b(2,2)
character(256) fname
! allocatable arrays
! low precision for band/spin character array saves memory
real(4), allocatable :: bc(:,:,:,:,:),sc(:,:,:)
real(8), allocatable :: w(:),e(:,:,:),f(:,:),g(:)
real(8), allocatable :: dt(:,:),dp(:,:,:),elm(:,:)
complex(8), allocatable :: ulm(:,:,:),dmat(:,:,:,:,:),sdmat(:,:,:)
complex(8), allocatable :: apwalm(:,:,:,:,:),evecfv(:,:,:),evecsv(:,:)
if (dosssum) then
  nsd=1
else
  nsd=nspinor
end if
if (dosmsum) then
  l0=0; l1=lmaxdb
else
  l0=1; l1=lmmaxdb
end if
! calculate only the diagonal parts of the density matrices by default
tspndg=.true.
tlmdg=.true.
allocate(bc(lmmaxdb,nspinor,natmtot,nstsv,nkptnr))
allocate(sc(nspinor,nstsv,nkptnr))
! generate unitary matrices which convert the Yₗₘ basis into the irreducible
! representation basis of the symmetry group at each atomic site
if (lmirep) then
  allocate(elm(lmmaxdb,natmtot),ulm(lmmaxdb,lmmaxdb,natmtot))
  call genlmirep(elm,ulm)
! write the eigenvalues of a pseudorandom matrix symmetrised by the site
! symmetries in the Yₗₘ basis
  call writeelmirep(fext,elm)
  tlmdg=.false.
end if
! compute the SU(2) operator used for rotating the density matrix to the
! desired spin-quantisation axis
if (spinpol) call sqasu2(sqaxis,tspndg,su2)
! begin parallel loop over k-points
call holdthd(nkptnr,nthd)
!$OMP PARALLEL DEFAULT(SHARED) &
!$OMP PRIVATE(apwalm,evecfv,evecsv,dmat,sdmat) &
!$OMP PRIVATE(jk,ispn,vl,vc) &
!$OMP PRIVATE(ias,is,ist,lm,b) &
!$OMP NUM_THREADS(nthd)
allocate(apwalm(ngkmax,apwordmax,lmmaxapw,natmtot,nspnfv))
allocate(evecfv(nmatmax,nstfv,nspnfv),evecsv(nstsv,nstsv))
allocate(dmat(lmmaxdb,nspinor,lmmaxdb,nspinor,nstsv))
allocate(sdmat(nspinor,nspinor,nstsv))
!$OMP DO SCHEDULE(DYNAMIC)
do ik=1,nkptnr
! equivalent reduced k-point
  jk=ivkik(ivk(1,ik),ivk(2,ik),ivk(3,ik))
! loop over first-variational spins
  do ispn=1,nspnfv
    vl(:)=vkl(:,ik)
    vc(:)=vkc(:,ik)
! spin-spiral case
    if (spinsprl) then
      if (ispn == 1) then
        vl(:)=vl(:)+0.5d0*vqlss(:)
        vc(:)=vc(:)+0.5d0*vqcss(:)
      else
        vl(:)=vl(:)-0.5d0*vqlss(:)
        vc(:)=vc(:)-0.5d0*vqcss(:)
      end if
    end if
! find the matching coefficients
    call match(ngk(ispn,ik),vgkc(:,:,ispn,ik),gkc(:,ispn,ik), &
     sfacgk(:,:,ispn,ik),apwalm(:,:,:,:,ispn))
  end do
! get the eigenvectors from file for non-reduced k-point
  call getevecfv('.OUT',0,vkl(:,ik),vgkl(:,:,:,ik),evecfv)
  call getevecsv('.OUT',0,vkl(:,ik),evecsv)
  do ias=1,natmtot
    is=idxis(ias)
! generate the density matrix for all states
    call gendmatk(tspndg,tlmdg,0,lmaxdb,ias,nstsv,[0],ngk(:,ik),apwalm, &
     evecfv,evecsv,lmmaxdb,dmat)
! convert (l,m) part to an irreducible representation if required
    if (.not.tlmdg) call dmatulm(ulm(:,:,ias),dmat)
! spin rotate the density matrices to desired spin-quantisation axis
    if (.not.tspndg) call dmatsu2(lmmaxdb,su2,dmat)
! determine the band characters from the density matrix
    do ist=1,nstsv
      do ispn=1,nspinor
        do lm=1,lmmaxdb
          bc(lm,ispn,ias,ist,ik)=dmat(lm,ispn,lm,ispn,ist)%re
        end do
      end do
    end do
  end do
! compute the spin density matrices of the second-variational states
  call gensdmat(evecsv,sdmat)
! spin rotate the density matrices to desired spin-quantisation axis
  if (.not.tspndg) then
    do ist=1,nstsv
      call z2mm(su2,sdmat(:,:,ist),b)
      call z2mmct(b,su2,sdmat(:,:,ist))
    end do
  end if
  do ist=1,nstsv
    do ispn=1,nspinor
      sc(ispn,ist,ik)=sdmat(ispn,ispn,ist)%re
    end do
  end do
end do
!$OMP END DO
deallocate(apwalm,evecfv,evecsv,dmat,sdmat)
!$OMP END PARALLEL
call freethd(nthd)
allocate(w(nwplot),e(nstsv,nkptnr,nspinor))
allocate(dt(nwplot,nsd),dp(nwplot,l0:l1,nsd))
! generate frequency grid
dw=(wplot(2)-wplot(1))/dble(nwplot)
do iw=1,nwplot
  w(iw)=dw*dble(iw-1)+wplot(1)
end do
! number of subdivisions used for interpolation in the Brillouin zone
nsk(:)=max(ngrkf/ngridk(:),1)
! sign for spin in DOS plot
sps(1)=1; sps(2)=-1
!-------------------!
!     total DOS     !
!-------------------!
allocate(f(nstsv,nkptnr),g(nwplot))
dt(:,:)=0.d0
do ispn=1,nspinor
  do ik=1,nkptnr
    jk=ivkik(ivk(1,ik),ivk(2,ik),ivk(3,ik))
    do ist=1,nstsv
! subtract the Fermi energy
      e(ist,ik,ispn)=evalsv(ist,jk)-efermi
! use diagonal of spin density matrix for weight
      f(ist,ik)=sc(ispn,ist,ik)
      if (tocc) then
        f(ist,ik)=f(ist,ik)*occsvp(ist,jk)
      else
        f(ist,ik)=f(ist,ik)*occmax
      end if
    end do
  end do
! integrate over the Brillouin zone
  call brzint(nswplot,ngridk,nsk,ivkiknr,nwplot,wplot,nstsv,nstsv,e(:,:,ispn), &
   f,g)
  if (dosssum) then
    dt(:,1)=dt(:,1)+g(:)
  else
    dt(:,ispn)=g(:)
  end if
end do
deallocate(f,g)
! output to file
open(50,file='TDOS'//trim(fext),form='FORMATTED',action='WRITE')
do ispn=1,nsd
  do iw=1,nwplot
    write(50,'(2G18.10)') w(iw),dt(iw,ispn)*sps(ispn)
  end do
  write(50,*)
end do
close(50)
! skip the partial DOS if required
if (.not.tpdos) goto 10
!---------------------!
!     partial DOS     !
!---------------------!
call holdthd(lmaxdb+1,nthd)
!$OMP PARALLEL DEFAULT(SHARED) &
!$OMP PRIVATE(f,g,ias,is,ia,ispn) &
!$OMP PRIVATE(l,lm,ik,jk,ist) &
!$OMP NUM_THREADS(nthd)
allocate(f(nstsv,nkptnr),g(nwplot))
do ias=1,natmtot
  is=idxis(ias)
  ia=idxia(ias)
!$OMP BARRIER
!$OMP SINGLE
  dp(:,:,:)=0.d0
!$OMP END SINGLE
  do ispn=1,nspinor
!$OMP DO SCHEDULE(DYNAMIC)
    do l=0,lmaxdb
      do lm=l**2+1,(l+1)**2
        do ik=1,nkptnr
          jk=ivkik(ivk(1,ik),ivk(2,ik),ivk(3,ik))
          do ist=1,nstsv
            f(ist,ik)=bc(lm,ispn,ias,ist,ik)
            if (tocc) then
              f(ist,ik)=f(ist,ik)*occsvp(ist,jk)
            else
              f(ist,ik)=f(ist,ik)*occmax
            end if
          end do
        end do
        call brzint(nswplot,ngridk,nsk,ivkiknr,nwplot,wplot,nstsv,nstsv, &
         e(:,:,ispn),f,g)
        if (dosmsum) then
          if (dosssum) then
            dp(:,l,1)=dp(:,l,1)+g(:)
          else
            dp(:,l,ispn)=dp(:,l,ispn)+g(:)
          end if
        else
          if (dosssum) then
            dp(:,lm,1)=dp(:,lm,1)+g(:)
          else
            dp(:,lm,ispn)=g(:)
          end if
        end if
! subtract from interstitial DOS
!$OMP CRITICAL(dos_)
        if (dosssum) then
          dt(:,1)=dt(:,1)-g(:)
        else
          dt(:,ispn)=dt(:,ispn)-g(:)
        end if
!$OMP END CRITICAL(dos_)
      end do
    end do
!$OMP END DO
  end do
! output to file
!$OMP SINGLE
  write(fname,'("PDOS_S",I2.2,"_A",I4.4)') is,ia
  open(50,file=trim(fname)//trim(fext),form='FORMATTED',action='WRITE')
  do ispn=1,nsd
    do l=l0,l1
      do iw=1,nwplot
        write(50,'(2G18.10)') w(iw),dp(iw,l,ispn)*sps(ispn)
      end do
      write(50,*)
    end do
  end do
  close(50)
!$OMP END SINGLE
end do
deallocate(f,g)
!$OMP END PARALLEL
call freethd(nthd)
!--------------------------!
!     interstitial DOS     !
!--------------------------!
open(50,file='IDOS'//trim(fext),form='FORMATTED',action='WRITE')
do ispn=1,nsd
  do iw=1,nwplot
    write(50,'(2G18.10)') w(iw),dt(iw,ispn)*sps(ispn)
  end do
  write(50,*)
end do
close(50)
10 continue
! write the total DOS to test file
call writetest(10,'total DOS',nv=nwplot*nsd,tol=2.d-2,rva=dt)
deallocate(bc,sc,w,e,dt,dp)
if (lmirep) deallocate(elm,ulm)
end subroutine
!EOC

