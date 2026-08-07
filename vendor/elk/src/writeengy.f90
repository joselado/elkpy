
! Copyright (C) 2002-2006 J. K. Dewhurst, S. Sharma and C. Ambrosch-Draxl.
! This file is distributed under the terms of the GNU Lesser General Public
! License. See the file COPYING for license details.

subroutine writeengy(fnum)
use modmain
use moddftu
implicit none
! arguments
integer, intent(in) :: fnum
write(fnum,*)
write(fnum,'("Energies :")')
write(fnum,'(" Fermi",T30,": ",G24.14)') efermi
write(fnum,'(" sum of occupied eigenvalues",T30,": ",G24.14)') evalsum
write(fnum,'(" electron kinetic",T30,": ",G24.14)') engykn
write(fnum,'(" core electron kinetic",T30,": ",G24.14)') engykncr
write(fnum,'(" Coulomb",T30,": ",G24.14)') engycl
write(fnum,'(" Coulomb potential",T30,": ",G24.14)') engyvcl
write(fnum,'(" nuclear-nuclear",T30,": ",G24.14)') engynn
write(fnum,'(" electron-nuclear",T30,": ",G24.14)') engyen
write(fnum,'(" Hartree",T30,": ",G24.14)') engyhar
write(fnum,'(" Madelung",T30,": ",G24.14)') engymad
write(fnum,'(" xc potential",T30,": ",G24.14)') engyvxc
if (spinpol) then
  write(fnum,'(" xc effective B-field",T30,": ",G24.14)') engybxc
  write(fnum,'(" external B-field",T30,": ",G24.14)') engybext
end if
write(fnum,'(" exchange",T30,": ",G24.14)') engyx
write(fnum,'(" correlation",T30,": ",G24.14)') engyc
if (dftu /= 0) then
  write(fnum,'(" DFT+U",T30,": ",G24.14)') engydu
end if
if (stype == 3) then
  write(fnum,'(" electron entropic",T30,": ",G24.14)') engyts
end if
write(fnum,'(" total energy",T30,": ",G24.14)') engytot
if (spinpol) then
  write(fnum,'(" (external B-field energy excluded from total)")')
end if
if (autodlefe) then
  write(fnum,*)
  write(fnum,'("Difference between fixed linearisation and Fermi energies &
   &(dlefe) : ",G18.10)') dlefe
end if
flush(fnum)
end subroutine

