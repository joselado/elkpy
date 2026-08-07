
! Copyright (C) 2021 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
! This file is distributed under the terms of the GNU General Public License.
! See the file COPYING for license details.

subroutine wfirsv(tgp,nst,idx,ngridg_,igfft_,ngp,igpig,evecfv,evecsv,ld,wfir)
use modmain
use modomp
implicit none
! arguments
logical, intent(in) :: tgp
integer, intent(in) :: nst,idx(*),ngridg_(3),igfft_(*)
integer, intent(in) :: ngp(nspnfv),igpig(ngkmax,nspnfv)
complex(8), intent(in) :: evecfv(nmatmax,nstfv,nspnfv),evecsv(nstsv,nstsv)
integer, intent(in) :: ld
complex(8), intent(out) :: wfir(ld,nspinor,nst)
! local variables
logical tasv
integer ist,ispn,jspn
integer n,igp,j,k,nthd
real(8) t0
complex(8) z1
! automatic arrays
complex(8) wfgp(ngkmax)
! check if all the second-variational wavefunctions should be calculated
tasv=(idx(1) == 0)
t0=1.d0/sqrt(omega)
call holdthd(nst,nthd)
!$OMP PARALLEL DO DEFAULT(SHARED) &
!$OMP PRIVATE(wfgp,k,ispn,jspn) &
!$OMP PRIVATE(n,ist,z1,igp) &
!$OMP SCHEDULE(DYNAMIC) &
!$OMP NUM_THREADS(nthd)
do j=1,nst
! index to state in evecsv
  k=merge(j,idx(j),tasv)
  if (tevecsv) then
! generate spinor wavefunction from second-variational eigenvectors
    do ispn=1,nspinor
      jspn=jspnfv(ispn)
      n=ngp(jspn)
      if (tgp) then
        wfir(1:n,ispn,j)=0.d0
      else
        wfgp(1:n)=0.d0
      end if
      do ist=1,nstfv
        z1=evecsv((ispn-1)*nstfv+ist,k)
        if (abs(z1%re)+abs(z1%im) < 1.d-12) cycle
        if (tgp) then
! wavefunction in G+p-space
          wfir(1:n,ispn,j)=wfir(1:n,ispn,j)+z1*evecfv(1:n,ist,jspn)
        else
! wavefunction in real-space
          z1=t0*z1
          wfgp(1:n)=wfgp(1:n)+z1*evecfv(1:n,ist,jspn)
        end if
      end do
! Fourier transform wavefunction to real-space if required
      if (.not.tgp) then
        wfir(1:ld,ispn,j)=0.d0
        do igp=1,n
          wfir(igfft_(igpig(igp,jspn)),ispn,j)=wfgp(igp)
        end do
        call zfftifc(3,ngridg_,1,wfir(:,ispn,j))
      end if
    end do
  else
! spin-unpolarised wavefunction
    n=ngp(1)
    if (tgp) then
      wfir(1:n,1,j)=evecfv(1:n,k,1)
    else
      wfir(1:ld,1,j)=0.d0
      do igp=1,n
        wfir(igfft_(igpig(igp,1)),1,j)=t0*evecfv(igp,k,1)
      end do
      call zfftifc(3,ngridg_,1,wfir(:,1,j))
    end if
  end if
end do
!$OMP END PARALLEL DO
call freethd(nthd)
end subroutine

