"""Unit tests for parsers/bse.py (tasks 185/186/187) -- pure text parsing,
no Elk run needed.

EIGVAL_BSE.OUT fixtures are transcribed from vendor/elk/src/writeevbse.f90:

    write(50,'(I6," : nmbse")') nmbse
    if (bsefull) then
      do a=1,nmbse ; write(50,'(I6,2G18.10)') a, dble(w(a)), aimag(w(a))
    else
      do a=1,nmbse ; write(50,'(I6,G18.10)') a, evalbse(a)

EPSILON_BSE_ij.OUT is vendor/elk/src/dielectric_bse.f90's two-block
(2G18.10) layout, identical to src/dielectric.f90's.
"""

import numpy as np
import pytest

from elkpy.parsers.bse import parse_bse_eigenvalues, parse_epsilon_bse


def _write_hermitian_eigenvalues(path, values):
    """writeevbse.f90's `.not.bsefull` branch: (I6,G18.10)."""
    with open(path, "w") as fh:
        fh.write(f"{len(values):6d} : nmbse\n")
        for a, e in enumerate(values, start=1):
            fh.write(f"{a:6d}{e:18.10G}\n")


def _write_full_eigenvalues(path, values):
    """writeevbse.f90's `bsefull` branch: (I6,2G18.10)."""
    with open(path, "w") as fh:
        fh.write(f"{len(values):6d} : nmbse\n")
        for a, e in enumerate(values, start=1):
            fh.write(f"{a:6d}{e.real:18.10G}{e.imag:18.10G}\n")


def test_parse_bse_eigenvalues_hermitian(tmp_path):
    values = np.array([0.10, 0.12, 0.155, 0.21])
    path = tmp_path / "EIGVAL_BSE.OUT"
    _write_hermitian_eigenvalues(path, values)

    parsed = parse_bse_eigenvalues(path)
    assert parsed.dtype == complex
    assert parsed.real == pytest.approx(values)
    # the Tamm-Dancoff block is Hermitian, so every eigenvalue is real
    assert np.all(parsed.imag == 0.0)


def test_parse_bse_eigenvalues_bsefull_keeps_imaginary_part(tmp_path):
    # the full non-Hermitian BSE matrix is diagonalised with zgeevi, so its
    # eigenvalues are genuinely complex -- the column-count branch is what
    # decides, not a flag the parser is told
    values = np.array([0.10 + 0.001j, 0.12 - 0.002j, -0.10 + 0.0j])
    path = tmp_path / "EIGVAL_BSE.OUT"
    _write_full_eigenvalues(path, values)

    parsed = parse_bse_eigenvalues(path)
    assert parsed == pytest.approx(values, abs=1e-12)


def test_parse_bse_eigenvalues_checks_declared_count(tmp_path):
    path = tmp_path / "EIGVAL_BSE.OUT"
    with open(path, "w") as fh:
        fh.write("     4 : nmbse\n")
        fh.write("     1      0.1000000000\n")
    with pytest.raises(ValueError, match="declares nmbse=4"):
        parse_bse_eigenvalues(path)


def test_parse_epsilon_bse_round_trip(tmp_path):
    # dielectric_bse.f90's grid: w(iw) = wplot(2)/nwplot*(iw-1), from zero
    energies = 0.5 / 6 * np.arange(6)
    real_part = 1.0 + np.linspace(0.0, 2.0, 6)
    imag_part = np.linspace(0.0, 5.0, 6)
    path = tmp_path / "EPSILON_BSE_11.OUT"
    with open(path, "w") as fh:
        for w, y in zip(energies, real_part):
            fh.write(f"{w:18.10G}{y:18.10G}\n")
        fh.write("\n")
        for w, y in zip(energies, imag_part):
            fh.write(f"{w:18.10G}{y:18.10G}\n")

    parsed_w, eps = parse_epsilon_bse(path)
    # G18.10 carries 10 significant figures, so the abscissa round-trips
    # only to that relative precision
    assert parsed_w == pytest.approx(energies, rel=1e-9, abs=1e-12)
    assert eps.real == pytest.approx(real_part)
    assert eps.imag == pytest.approx(imag_part)
