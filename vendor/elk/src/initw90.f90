
! Copyright (C) 2017-18 Arsenii Gerasimov, Yaroslav Kvashnin and Lars Nordstrom.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

!BOP
! !ROUTINE: initw90
! !INTERFACE:
subroutine initw90
! !USES:
use modmain
use modw90
! !DESCRIPTION:
!   Initialises global variables for the Wannier90 interface.
!
! !REVISION HISTORY:
!   Created November 2018 (Arsenii Gerasimov)
!EOP
!BOC
implicit none
! local variables
integer ik,ist,ispn
integer is,ias,l,m,n,i,j
! initialise universal variables
call init0
call init1
if (num_bands > nstsv) then
  write(*,*)
  write(*,'("Error(initw90): num_bands > nstsv :",2(X,I0))') num_bands,nstsv
  write(*,*)
  stop
end if
! if num_bands is not positive then assume all states are used
if (num_bands < 1) then
  if (allocated(idxw90)) deallocate(idxw90)
  allocate(idxw90(nstsv))
  do ist=1,nstsv
    idxw90(ist)=ist
  end do
  num_bands=nstsv
end if
! check that each state index is in range
do i=1,num_bands
  ist=idxw90(i)
  if ((ist < 1).or.(ist > nstsv)) then
    write(*,*)
    write(*,'("Error(initw90): state index out of range : ",I0)') ist
    write(*,*)
    stop
  end if
end do
if (projw90) then
! determine the map from the spin, l and atom projector to the Wannier function
  if (allocated(prjwn)) deallocate(prjwn)
  allocate(prjwn(nspinor,4,natmtot))
  i=0
  do ias=1,natmtot
    is=idxis(ias)
    n=0
    do j=1,4
      l=lprojw90(j,is)
      if ((l < 0).or.(l > 3)) exit
      n=n+1
      do m=-l,l
        do ispn=1,nspinor
          i=i+1
          if (m == -l) prjwn(ispn,j,ias)=i
        end do
      end do
    end do
    nprojw90(is)=n
  end do
  num_wann=i
  if (num_wann == 0) then
    write(*,*)
    write(*,'("Error(init0): no projectors specified")')
    write(*,*)
    stop
  end if
else if (num_wann < 1) then
  num_wann=num_bands+num_wann
  num_wann=max(num_wann,1)
end if
if (num_wann > num_bands) then
  write(*,*)
  write(*,'("Error(initw90): num_wann > num_bands :",2(X,I0))') num_wann, &
   num_bands
  write(*,*)
  stop
end if
! read density and potentials from file
call readstate
! find the new linearisation energies
call linengy
! generate the APW and local-orbital radial functions and integrals
call genapwlofr
! read in the second-variational eigenvalues
do ik=1,nkpt
  call getevalsv(filext,ik,vkl(:,ik),evalsv(:,ik))
end do
end subroutine
!EOC

