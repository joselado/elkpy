"""Unit tests for parsers/eigenstates.py -- pure token-list parsing, no Elk
run needed (mirrors tests/test_berry_gauge_invariance.py's spirit of
testing the Python-side arithmetic/parsing independently of Fortran)."""

import numpy as np
import pytest

from elkpy.parsers.eigenstates import (
    parse_angular_momentum_response,
    parse_eigenstates_response,
    parse_momentum_response,
    parse_orbital_projection_response,
    parse_overlap_response,
    parse_projection_response,
)


def _complex_matrix_tokens(mat):
    """Flatten a complex matrix into the same token order
    src/elkpy_eigenstates.f90 writes: "do b; do a" (a innermost), each
    entry as two tokens (re, im) -- i.e. Fortran/column-major order."""
    tokens = []
    for value in mat.flatten(order="F"):
        tokens.append(repr(float(value.real)))
        tokens.append(repr(float(value.imag)))
    return tokens


def test_parse_eigenstates_response_round_trip():
    rng = np.random.default_rng(0)
    nstsv = 5
    energies = rng.uniform(-1, 1, size=nstsv)
    # a random unitary matrix, standing in for a real evecsv
    a = rng.normal(size=(nstsv, nstsv)) + 1j * rng.normal(size=(nstsv, nstsv))
    q, _ = np.linalg.qr(a)

    tokens = [str(nstsv)] + [repr(float(e)) for e in energies] + _complex_matrix_tokens(q)
    parsed_energies, parsed_evecsv = parse_eigenstates_response(tokens)

    assert parsed_energies == pytest.approx(energies, abs=1e-12)
    assert parsed_evecsv == pytest.approx(q, abs=1e-12)


def test_parse_overlap_response_round_trip():
    rng = np.random.default_rng(1)
    nst = 3
    mat = rng.normal(size=(nst, nst)) + 1j * rng.normal(size=(nst, nst))

    tokens = [str(nst)] + _complex_matrix_tokens(mat)
    parsed = parse_overlap_response(tokens)

    assert parsed == pytest.approx(mat, abs=1e-12)


def test_parse_projection_response_round_trip():
    rng = np.random.default_rng(2)
    nst, natmtot = 3, 4
    mats = rng.normal(size=(natmtot, nst, nst)) + 1j * rng.normal(size=(natmtot, nst, nst))

    # src/elkpy_eigenstates.f90's PROJECTION case writes "do ias; do b; do a"
    # (a innermost) -- each atom's nst x nst block column-major, blocks
    # consecutive.
    tokens = [str(nst), str(natmtot)]
    for ias in range(natmtot):
        tokens += _complex_matrix_tokens(mats[ias])
    parsed = parse_projection_response(tokens)

    assert parsed.shape == (natmtot, nst, nst)
    assert parsed == pytest.approx(mats, abs=1e-12)


def test_parse_orbital_projection_response_round_trip():
    rng = np.random.default_rng(3)
    nst, natmtot, nl = 2, 3, 4
    mats = rng.normal(size=(natmtot, nl, nst, nst)) + 1j * rng.normal(
        size=(natmtot, nl, nst, nst)
    )

    # src/elkpy_eigenstates.f90's ORBITAL case writes "do ias; do lsel; do b;
    # do a" (a innermost) -- each (atom, l) block column-major, blocks
    # consecutive, l fastest-varying after (a, b).
    tokens = [str(nst), str(natmtot), str(nl)]
    for ias in range(natmtot):
        for lsel in range(nl):
            tokens += _complex_matrix_tokens(mats[ias, lsel])
    parsed = parse_orbital_projection_response(tokens)

    assert parsed.shape == (natmtot, nl, nst, nst)
    assert parsed == pytest.approx(mats, abs=1e-12)


def test_parse_angular_momentum_response_round_trip():
    rng = np.random.default_rng(4)
    nst, natmtot, nl, ncomp = 2, 3, 4, 3
    mats = rng.normal(size=(natmtot, nl, ncomp, nst, nst)) + 1j * rng.normal(
        size=(natmtot, nl, ncomp, nst, nst)
    )

    # src/elkpy_eigenstates.f90's ANGMOM case writes "do ias; do lsel; do
    # comp; do b; do a" (a innermost) -- each (atom, l, comp) block
    # column-major, blocks consecutive, comp fastest-varying after (a, b),
    # then l, then ias.
    tokens = [str(nst), str(natmtot), str(nl), str(ncomp)]
    for ias in range(natmtot):
        for lsel in range(nl):
            for comp in range(ncomp):
                tokens += _complex_matrix_tokens(mats[ias, lsel, comp])
    parsed = parse_angular_momentum_response(tokens)

    assert parsed.shape == (natmtot, nl, ncomp, nst, nst)
    assert parsed == pytest.approx(mats, abs=1e-12)


def test_parse_momentum_response_round_trip():
    rng = np.random.default_rng(5)
    nstsv, ncomp = 4, 3
    energies = rng.normal(size=nstsv)
    pmat = rng.normal(size=(ncomp, nstsv, nstsv)) + 1j * rng.normal(
        size=(ncomp, nstsv, nstsv)
    )

    # src/elkpy_eigenstates.f90's MOMENTUM case writes nstsv, then nstsv
    # eigenvalues, then "do comp; do b; do a" (a innermost) -- each
    # Cartesian component's block column-major, the three blocks
    # consecutive. Unlike every other query here there is no band window
    # in the protocol at all (genpmatk's array is hard-dimensioned nstsv).
    tokens = [str(nstsv)] + [repr(float(e)) for e in energies]
    for comp in range(ncomp):
        tokens += _complex_matrix_tokens(pmat[comp])
    parsed_energies, parsed_pmat = parse_momentum_response(tokens)

    assert parsed_energies.shape == (nstsv,)
    assert parsed_energies == pytest.approx(energies, abs=1e-12)
    assert parsed_pmat.shape == (ncomp, nstsv, nstsv)
    assert parsed_pmat == pytest.approx(pmat, abs=1e-12)
