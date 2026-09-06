"""Unit tests for parsers/nonlinopt.py (task 125, second-harmonic
generation) -- pure text parsing, no Elk run needed.

Fixtures are transcribed from the write statements in
vendor/elk/src/nonlinopt.f90's output section (the `if (mp_mpi) then`
block near the end of the file):

    write(50,'(2G18.10)') t1, dble(chi2w(iw))      ! nwplot lines
    write(50,*)                                    ! blank separator
    write(50,'(2G18.10)') t1, aimag(chi2w(iw))     ! nwplot lines

and the grid statements just above the k-loop:

    t1 = wplot(2)/dble(nwplot)
    do iw=1,nwplot ; w(iw) = t1*dble(iw-1) ; end do
"""

import numpy as np
import pytest

from elkpy.parsers.nonlinopt import energies_from_wplot, parse_chi2


def _write_two_blocks(path, energies, real_part, imag_part):
    """Reproduce nonlinopt.f90's (2G18.10) two-block layout."""
    with open(path, "w") as fh:
        for w, y in zip(energies, real_part):
            fh.write(f"{w:18.10G}{y:18.10G}\n")
        fh.write("\n")
        for w, y in zip(energies, imag_part):
            fh.write(f"{w:18.10G}{y:18.10G}\n")


def test_energies_from_wplot_matches_fortran_grid():
    # nonlinopt.f90: t1 = wplot(2)/nwplot ; w(iw) = t1*(iw-1)
    w = energies_from_wplot(1.0, 200)
    assert len(w) == 200
    assert w[0] == 0.0                 # starts at ZERO, not at wplot(1)
    assert w[1] == pytest.approx(0.005)
    assert w[-1] == pytest.approx(1.0 - 1.0 / 200)   # upper endpoint EXCLUDED


def test_energies_from_wplot_rejects_tiny_grid():
    # readinput.f90 case('wplot'): nwplot < 2 is a hard stop in Elk
    with pytest.raises(ValueError):
        energies_from_wplot(1.0, 1)


def test_parse_chi2_round_trip(tmp_path):
    energies = energies_from_wplot(0.5, 8)
    real_part = np.linspace(-1e-6, 3e-6, 8)
    imag_part = np.linspace(2e-6, -4e-6, 8)
    path = tmp_path / "CHI_2WWW_123.OUT"
    _write_two_blocks(path, energies, real_part, imag_part)

    parsed_w, chi = parse_chi2(path)
    assert parsed_w == pytest.approx(energies, rel=1e-9, abs=1e-12)
    assert chi.real == pytest.approx(real_part, rel=1e-9, abs=1e-14)
    assert chi.imag == pytest.approx(imag_part, rel=1e-9, abs=1e-14)


def test_parse_chi2_rejects_one_block(tmp_path):
    # a truncated file (an interrupted run) must not silently return half
    # a spectrum with the imaginary part missing
    path = tmp_path / "CHI_2WWW_111.OUT"
    with open(path, "w") as fh:
        fh.write("   0.000000000      0.1000000000E-05\n")
    with pytest.raises(ValueError, match="expected 2 blocks"):
        parse_chi2(path)
