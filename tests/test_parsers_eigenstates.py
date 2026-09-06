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
    evecsv = rng.normal(size=(nstsv, nstsv)) + 1j * rng.normal(size=(nstsv, nstsv))
    pmat = rng.normal(size=(ncomp, nstsv, nstsv)) + 1j * rng.normal(
        size=(ncomp, nstsv, nstsv)
    )

    # src/elkpy_eigenstates.f90's MOMENTUM case writes nstsv, then nstsv
    # eigenvalues, then evecsv ("do b; do a", column-major -- exactly the
    # EIGENSTATES response's own leading block, added by patches/0009 so
    # the spin operators and pmat share one diagonalisation, see
    # docs/design.md #24), then "do comp; do b; do a" (a innermost) --
    # each Cartesian component's block column-major, the three blocks
    # consecutive. Unlike every other query here there is no band window
    # in the protocol at all (genpmatk's array is hard-dimensioned nstsv).
    tokens = [str(nstsv)] + [repr(float(e)) for e in energies]
    tokens += _complex_matrix_tokens(evecsv)
    for comp in range(ncomp):
        tokens += _complex_matrix_tokens(pmat[comp])
    parsed_energies, parsed_evecsv, parsed_pmat = parse_momentum_response(tokens)

    assert parsed_energies.shape == (nstsv,)
    assert parsed_energies == pytest.approx(energies, abs=1e-12)
    assert parsed_evecsv.shape == (nstsv, nstsv)
    assert parsed_evecsv == pytest.approx(evecsv, abs=1e-12)
    assert parsed_pmat.shape == (ncomp, nstsv, nstsv)
    assert parsed_pmat == pytest.approx(pmat, abs=1e-12)


def _upper_triangle_tokens(mat):
    """The token order elkpy_lapwexport writes H and O in: `do j; do i=1,j`,
    i.e. the UPPER triangle walked column by column. Elk fills nothing else
    -- olpistl/hmlistl and every muffin-tin contribution run `do i=1,j` --
    so the lower triangle of its array is uninitialised and deliberately not
    written."""
    tokens = []
    for j in range(mat.shape[1]):
        for i in range(j + 1):
            tokens.append("%.17g" % mat[i, j].real)
            tokens.append("%.17g" % mat[i, j].imag)
    return tokens


def test_upper_triangle_reconstruction_is_hermitian():
    """The column-major upper triangle must come back as the full Hermitian
    matrix. Reading it in numpy's own row-major triu order instead gives a
    matrix that is not Hermitian at all, which is how this was caught: the
    first real export failed scipy.linalg.eigh with "B is not positive
    definite" rather than returning a subtly wrong number."""
    from elkpy.parsers.eigenstates import _upper_triangle

    rng = np.random.default_rng(7)
    n = 6
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    hermitian = a + a.conj().T
    tokens = _upper_triangle_tokens(hermitian)
    parsed, pos = _upper_triangle(tokens, 0, n)
    assert pos == len(tokens)
    assert parsed == pytest.approx(hermitian, abs=1e-12)
    assert parsed == pytest.approx(parsed.conj().T, abs=1e-15)


def _lapw_tokens(reference):
    """Emit a LAPW response in exactly the order elkpy_lapwexport writes it."""
    tokens = [str(reference[key]) for key in (
        "ngp", "nlotot", "nmatp", "nstfv", "apwordmax", "lmmaxapw", "natmtot",
        "nspecies", "lmaxapw", "npapw", "nmatmax")]
    real = lambda x: "%.17g" % float(x)
    tokens.append(real(reference["omega"]))
    for name in ("avec", "bvec"):
        tokens += [real(v) for v in reference[name].flatten(order="F")]
    tokens += [real(v) for v in reference["vkc"]]
    tokens += [str(int(v)) for v in reference["idxis"]]
    tokens += [real(v) for v in reference["rmt"]]
    tokens += [str(int(v)) for v in reference["nrmt"]]
    tokens += [str(int(v)) for v in reference["apword"].flatten(order="F")]
    tokens += [real(v) for v in reference["atposc"].flatten(order="F")]
    tokens += [str(int(v)) for v in reference["igpig"]]
    for name in ("vgpl", "vgpc"):
        tokens += [real(v) for v in reference[name].flatten(order="F")]
    tokens += [real(v) for v in reference["gpc"]]
    for value in reference["apwalm"].flatten(order="F"):
        tokens += [real(value.real), real(value.imag)]
    for per_atom in reference["dmat"]:
        for matrix in per_atom:
            tokens.append(str(matrix.shape[0]))
            tokens += [real(v) for v in matrix.flatten(order="F")]
    for per_atom in reference["apwfr"]:
        for per_l in per_atom:
            for row in per_l:
                tokens += [real(v) for v in row]
    tokens += [real(v) for v in reference["rsp"].flatten()]
    for name in ("hmat", "omat", "hmat_istl", "omat_istl"):
        tokens += _upper_triangle_tokens(reference[name])
    tokens += [real(v) for v in reference["evalfv"]]
    for value in reference["evecfv"].flatten(order="F"):
        tokens += [real(value.real), real(value.imag)]
    return tokens


def test_parse_lapw_response_round_trip():
    """Every block of the LAPW export, round-tripped through the writer's own
    ordering. The nested dmat/apwfr lists are ragged by construction --
    apword varies with l -- which is why they are lists rather than arrays,
    and why the parser has to read each block's own order count first."""
    from elkpy.parsers.eigenstates import parse_lapw_response

    rng = np.random.default_rng(11)
    ngp, nlotot, lmaxapw, natmtot, nspecies, npapw = 5, 2, 1, 2, 2, 4
    nmatp, nstfv, apwordmax = ngp + nlotot, 3, 2
    lmmaxapw = (lmaxapw + 1) ** 2
    apword = np.array([[1, 2], [2, 1]])          # (lmaxapw+1, nspecies)
    idxis = np.array([1, 2])

    def hermitian(n):
        a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        return a + a.conj().T

    reference = dict(
        ngp=ngp, nlotot=nlotot, nmatp=nmatp, nstfv=nstfv,
        apwordmax=apwordmax, lmmaxapw=lmmaxapw, natmtot=natmtot,
        nspecies=nspecies, lmaxapw=lmaxapw, npapw=npapw, nmatmax=nmatp + 1,
        omega=270.0125,
        avec=rng.normal(size=(3, 3)), bvec=rng.normal(size=(3, 3)),
        vkc=rng.normal(size=3), idxis=idxis,
        rmt=rng.uniform(1, 3, size=nspecies),
        nrmt=np.array([300, 400]),
        apword=apword, atposc=rng.normal(size=(3, natmtot)),
        igpig=np.arange(1, ngp + 1),
        vgpl=rng.normal(size=(3, ngp)), vgpc=rng.normal(size=(3, ngp)),
        gpc=rng.uniform(0.1, 4, size=ngp),
        apwalm=(rng.normal(size=(ngp, apwordmax, lmmaxapw, natmtot))
                + 1j * rng.normal(size=(ngp, apwordmax, lmmaxapw, natmtot))),
        dmat=[[rng.normal(size=(apword[l, idxis[a] - 1],) * 2)
               for l in range(lmaxapw + 1)] for a in range(natmtot)],
        apwfr=[[rng.normal(size=(apword[l, idxis[a] - 1], npapw))
                for l in range(lmaxapw + 1)] for a in range(natmtot)],
        rsp=rng.uniform(1, 3, size=(nspecies, npapw)),
        hmat=hermitian(nmatp), omat=hermitian(nmatp),
        hmat_istl=hermitian(ngp), omat_istl=hermitian(ngp),
        evalfv=rng.normal(size=nstfv),
        evecfv=(rng.normal(size=(nmatp, nstfv))
                + 1j * rng.normal(size=(nmatp, nstfv))),
    )
    parsed = parse_lapw_response(_lapw_tokens(reference))

    for key, value in reference.items():
        if key in ("dmat", "apwfr"):
            for atom_ref, atom_got in zip(value, parsed[key]):
                for ref, got in zip(atom_ref, atom_got):
                    assert got == pytest.approx(ref, abs=1e-12)
        elif isinstance(value, np.ndarray):
            assert parsed[key] == pytest.approx(value, abs=1e-12), key
        else:
            assert parsed[key] == pytest.approx(value, abs=1e-12), key
