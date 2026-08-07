
! Copyright (C) 2016 A. Davydov, A. Sanna, J. K. Dewhurst, S. Sharma and
! E. K. U. Gross. This file is distributed under the terms of the GNU General
! Public License. See the file COPYING for license details.

subroutine gwsefmk(ikp,vmt,vir,bmt,bir,se)
use modmain
use modgw
use modomp
implicit none
! arguments
integer, intent(in) :: ikp
real(8), intent(in) :: vmt(npcmtmax,natmtot),vir(ngtc)
real(8), intent(in) :: bmt(npcmtmax,natmtot,ndmag),bir(ngtc,ndmag)
complex(8), intent(out) :: se(nstsv,nstsv,0:nwfm)
! local variables
integer ik,jk,ist1,ist2,ist3
integer iv(3),iq,ig0,ig1,ig,jg
integer iw,jw,it,nthd
real(8) vl(3),vc(3),wo,t1,t2
complex(4) c1,c2
! automatic arrays
integer(omp_lock_kind) lock(nwgw)
real(8) vgqc(3,ngvc),gqc(ngvc),gclgq(ngvc)
complex(8) zfgq(ngrf)
complex(4) cvclmt(npcmtmax,natmtot),cvclir(ngtc)
complex(4) y(max(nstsv,nwgw))
! allocatable arrays
real(8), allocatable :: jlgqr(:,:,:),jlgqrmt(:,:,:)
complex(8), allocatable :: apwalm(:,:,:,:),evecfv(:,:),evecsv(:,:)
complex(8), allocatable :: ylmgq(:,:),sfacgq(:,:)
complex(4), allocatable :: wfmt1(:,:,:,:),wfir1(:,:,:)
complex(4), allocatable :: wfmt2(:,:,:,:),wfir2(:,:,:)
complex(4), allocatable :: crhomt(:,:,:),crhoir(:,:)
complex(4), allocatable :: crgq(:,:,:),gs(:,:),stau(:,:,:),wc(:,:)
complex(8), allocatable :: epsi(:,:,:),v(:,:)
! external functions
complex(8), external :: zcfinp
! allocate local arrays
allocate(jlgqr(njcmax,nspecies,ngrf),jlgqrmt(0:lnpsd,ngvc,nspecies))
allocate(apwalm(ngkmax,apwordmax,lmmaxapw,natmtot))
allocate(ylmgq(lmmaxo,ngvc),sfacgq(ngvc,natmtot))
allocate(evecfv(nmatmax,nstfv),evecsv(nstsv,nstsv))
allocate(wfmt1(npcmtmax,natmtot,nspinor,nstsv),wfir1(ngtc,nspinor,nstsv))
allocate(wfmt2(npcmtmax,natmtot,nspinor,nstsv),wfir2(ngtc,nspinor,nstsv))
allocate(crhomt(npcmtmax,natmtot,nstsv),crhoir(ngtc,nstsv))
allocate(crgq(nstsv,ngrf,nstsv),gs(nwgw,nstsv),stau(nstsv,nstsv,nwgw))
allocate(epsi(ngrf,ngrf,nwrf),v(nstsv,nstsv))
! initialise the OpenMP locks
do it=1,nwgw
  call omp_init_lock(lock(it))
end do
! get the eigenvectors from file for input reduced k-point
call getevecfv(filext,ikp,vkl(:,ikp),vgkl(:,:,:,ikp),evecfv)
call getevecsv(filext,ikp,vkl(:,ikp),evecsv)
! find the matching coefficients
call match(ngk(1,ikp),vgkc(:,:,1,ikp),gkc(:,1,ikp),sfacgk(:,:,1,ikp),apwalm)
! calculate the wavefunctions for all states of the input k-point
call genwfsv_sp(.false.,.true.,nstsv,[0],ngdgc,igfc,ngk(1,ikp),igkig(:,1,ikp), &
 apwalm,evecfv,evecsv,wfmt1,ngtc,wfir1)
! local -V_xc and -B_xc matrix elements
if (spinpol) then
  call genvbmatk(vmt,vir,bmt,bir,ngk(1,ikp),igkig(:,1,ikp),wfmt1,ngtc,wfir1,v)
else
  call genvmatk(vmt,vir,ngk(1,ikp),igkig(:,1,ikp),wfmt1,ngtc,wfir1,v)
end if
! Fourier transform wavefunctions to real-space
call cftwfir(ngk(1,ikp),igkig(:,1,ikp),wfir1)
! add the core Fock matrix elements
call vclcore(wfmt1,v)
! zero the self-energy matrix elements in tau-space
stau(1:nstsv,1:nstsv,1:nwgw)=0.e0
! loop over non-reduced k-point set
do ik=1,nkptnr
! equivalent reduced k-point
  jk=ivkik(ivk(1,ik),ivk(2,ik),ivk(3,ik))
! determine the q-vector
  iv(1:3)=ivk(1:3,ikp)-ivk(1:3,ik)
  iv(1:3)=modulo(iv(1:3),ngridk(1:3))
! check if the q-point is in user-defined set
  iv(1:3)=iv(1:3)*ngridq(1:3)
  if (any(mod(iv(1:3),ngridk(1:3)) /= 0)) cycle
  iv(1:3)=iv(1:3)/ngridk(1:3)
  iq=ivqiq(iv(1),iv(2),iv(3))
  vl(1:3)=vkl(1:3,ikp)-vkl(1:3,ik)
  vc(1:3)=vkc(1:3,ikp)-vkc(1:3,ik)
  do ig=1,ngvc
! determine the G+q-vectors
    vgqc(1:3,ig)=vgc(1:3,ig)+vc(1:3)
! G+q-vector length
    gqc(ig)=sqrt(vgqc(1,ig)**2+vgqc(2,ig)**2+vgqc(3,ig)**2)
! spherical harmonics for G+q-vectors
    call genylmv(.true.,lmaxo,vgqc(:,ig),ylmgq(:,ig))
  end do
! structure factors for G+q
  call gensfacgp(ngvc,vgqc,ngvc,sfacgq)
! generate the regularised Coulomb Green's function in G+q-space
  call gengclgq(.true.,iq,ngvc,gqc,gclgq)
! compute the required spherical Bessel functions
  call genjlgprmt(lnpsd,ngvc,gqc,ngvc,jlgqrmt)
  call genjlgpr(ngrf,gqc,jlgqr)
! find the matching coefficients
  call match(ngk(1,ik),vgkc(:,:,1,ik),gkc(:,1,ik),sfacgk(:,:,1,ik),apwalm)
! get the eigenvectors from file for non-reduced k-point
  call getevecfv(filext,0,vkl(:,ik),vgkl(:,:,1,ik),evecfv)
  call getevecsv(filext,0,vkl(:,ik),evecsv)
! calculate the wavefunctions for all states
  call genwfsv_sp(.false.,.false.,nstsv,[0],ngdgc,igfc,ngk(1,ik),igkig(:,1,ik),&
   apwalm,evecfv,evecsv,wfmt2,ngtc,wfir2)
  call holdthd(nstsv,nthd)
!$OMP PARALLEL DEFAULT(SHARED) &
!$OMP PRIVATE(cvclmt,cvclir,zfgq) &
!$OMP PRIVATE(ist1,ist2,ist3,wo,t1,iw) &
!$OMP NUM_THREADS(nthd)
! determine the complex densities and Fourier transform to G+q-space
  do ist3=1,nstsv
!$OMP DO SCHEDULE(DYNAMIC)
    do ist1=1,nstsv
      call gencrho(.true.,.true.,ngtc,wfmt2(:,:,:,ist3),wfir2(:,:,ist3), &
       wfmt1(:,:,:,ist1),wfir1(:,:,ist1),crhomt(:,:,ist1),crhoir(:,ist1))
      call zftcf(ngrf,jlgqr,ylmgq,ngvc,sfacgq,crhomt(:,:,ist1),crhoir(:,ist1), &
       zfgq)
      crgq(ist1,1:ngrf,ist3)=conjg(zfgq(1:ngrf))
    end do
!$OMP END DO
!--------------------------------------!
!     valence Fock matrix elements     !
!--------------------------------------!
    wo=wqptnr*occsv(ist3,jk)/occmax
    if (abs(wo) < epsocc) cycle
!$OMP DO SCHEDULE(DYNAMIC)
    do ist2=1,nstsv
! calculate the Coulomb potential
      call gencvclmt(nrcmt,nrcmti,nrcmtmax,rlcmt,wprcmt,npcmtmax, &
       crhomt(:,:,ist2),cvclmt)
      call cpotcoul(nrcmt,nrcmti,npcmt,nrcmtmax,rlcmt,ngdgc,igfc,ngvc,gqc, &
       gclgq,ngvc,jlgqrmt,ylmgq,sfacgq,crhoir(:,ist2),npcmtmax,cvclmt,cvclir)
      cvclir(1:ngtc)=cvclir(1:ngtc)*cfrc(1:ngtc)
      do ist1=1,ist2
        v(ist1,ist2)=v(ist1,ist2) &
         -wo*zcfinp(crhomt(:,:,ist1),crhoir(:,ist1),cvclmt,cvclir)
      end do
    end do
!$OMP END DO
  end do
!-------------------------------------!
!     correlation matrix elements     !
!-------------------------------------!
! generate Gₛ in state and tau-space
!$OMP DO SCHEDULE(DYNAMIC)
  do ist1=1,nstsv
    t1=efermi-evalsv(ist1,jk)
    gs(1:nwgw,ist1)=0.e0
    do iw=-nwfm,nwfm,2
      gs(iwfft(iw),ist1)=1.e0/cmplx(t1,wgw(iw),4)
    end do
    call cfftifc(1,nwgw,1,gs(:,ist1))
  end do
!$OMP END DO
!$OMP END PARALLEL
  call freethd(nthd)
! get RPA inverse dielectric function from file
! this is the symmetric version: ϵ⁻¹ = 1 - v¹⸍² χ₀ v¹⸍²
  call getcfgq('EPSINV.OUT',vl,ngrf,nwrf,epsi)
! symmetrise the Coulomb Green's function
  gclgq(1:ngrf)=sqrt(gclgq(1:ngrf))
  call holdthd(ngrf,nthd)
!$OMP PARALLEL DEFAULT(SHARED) &
!$OMP PRIVATE(wc,y,ig0,ig1,t1,t2,ig) &
!$OMP PRIVATE(iw,jw,it,ist2,ist3,c2) &
!$OMP NUM_THREADS(nthd)
  allocate(wc(nwgw,ngrf))
!$OMP DO SCHEDULE(DYNAMIC)
  do jg=1,ngrf
! if the real part of epsi is exactly zero then there is no entry for this
! particular G'+q-vector so we cycle to the next
    if (dble(epsi(jg,jg,1)) == 0.d0) cycle
! determine the range of ig from the desired matrix bandwidth
    if (mbwgrf >= 0) then
      ig0=max(jg-mbwgrf,1)
      ig1=min(jg+mbwgrf,ngrf)
    else
      ig0=1; ig1=ngrf
    end if
! subtract one from ϵ⁻¹ to leave just the correlation part
    epsi(jg,jg,1:nwrf)=epsi(jg,jg,1:nwrf)-1.d0
! compute the correlation part of the screened interaction W_c
    t1=gclgq(jg)
    do ig=ig0,ig1
      t2=t1*gclgq(ig)
      wc(1:nwgw,ig)=0.e0
      do iw=-nwbs,nwbs,2
        jw=(iw+nwbs)/2+1
        wc(iwfft(iw),ig)=t2*epsi(ig,jg,jw)
      end do
! Fourier transform W_c to tau-space
      call cfftifc(1,nwgw,1,wc(:,ig))
    end do
! loop over tau-points
    do it=1,nwgw
      do ist3=1,nstsv
        c1=gs(it,ist3)
        y(1:nstsv)=0.e0
        do ig=ig0,ig1
          c2=c1*wc(it,ig)
          y(1:nstsv)=y(1:nstsv)+c2*crgq(1:nstsv,ig,ist3)
        end do
        call omp_set_lock(lock(it))
        if (tsediag) then
! compute only the diagonal elements of the self-energy
          do ist2=1,nstsv
            c2=conjg(crgq(ist2,jg,ist3))
            stau(ist2,ist2,it)=stau(ist2,ist2,it)+c2*y(ist2)
          end do
        else
! compute the full self-energy matrix
          do ist2=1,nstsv
            c2=conjg(crgq(ist2,jg,ist3))
            stau(1:nstsv,ist2,it)=stau(1:nstsv,ist2,it)+c2*y(1:nstsv)
          end do
        end if
        call omp_unset_lock(lock(it))
      end do
    end do
  end do
!$OMP END DO
  deallocate(wc)
!$OMP END PARALLEL
  call freethd(nthd)
! end loop over k-points
end do
! destroy the OpenMP locks
do it=1,nwgw
  call omp_destroy_lock(lock(it))
end do
! Fourier transform the self-energy to frequency space, multiply by GW diagram
! prefactor and store in output array
t1=-wqptnr*omega*kboltz*tempk
call holdthd(nstsv,nthd)
!$OMP PARALLEL DO DEFAULT(SHARED) &
!$OMP PRIVATE(y,ist1,iw,jw) &
!$OMP NUM_THREADS(nthd) &
!$OMP SCHEDULE(DYNAMIC)
do ist2=1,nstsv
  do ist1=1,nstsv
    y(1:nwgw)=stau(ist1,ist2,1:nwgw)
    call cfftifc(1,nwgw,-1,y)
    do iw=-nwfm,nwfm,2
      jw=(iw+nwfm)/2
      se(ist1,ist2,jw)=t1*y(iwfft(iw))
    end do
  end do
end do
!$OMP END PARALLEL DO
call freethd(nthd)
! add the local potential and Fock matrix elements to the self-energy for each
! Matsubara frequency
call holdthd(nwfm+1,nthd)
!$OMP PARALLEL DO DEFAULT(SHARED) &
!$OMP PRIVATE(ist1,ist2) &
!$OMP SCHEDULE(DYNAMIC) &
!$OMP NUM_THREADS(nthd)
do iw=0,nwfm
  do ist2=1,nstsv
    do ist1=1,ist2
      se(ist1,ist2,iw)=se(ist1,ist2,iw)+v(ist1,ist2)
    end do
    do ist1=ist2+1,nstsv
      se(ist1,ist2,iw)=se(ist1,ist2,iw)+conjg(v(ist2,ist1))
    end do
  end do
end do
!$OMP END PARALLEL DO
call freethd(nthd)
deallocate(jlgqr,jlgqrmt)
deallocate(ylmgq,sfacgq,apwalm,evecfv,evecsv)
deallocate(wfmt1,wfir1,wfmt2,wfir2)
deallocate(crhomt,crhoir,gs,stau,crgq,epsi,v)
end subroutine

