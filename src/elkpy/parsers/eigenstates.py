"""Parse responses from the elkpy interactive eigenstate/overlap session
(task 9002, src/elkpy_eigenstates.f90 -- patches/0003-eigenstate-session.patch).

Pure token-list parsing, no subprocess/file dependency, mirroring
parsers/berry.py's style -- session.py owns reading lines from the running
elk process and handing the accumulated tokens to the functions here, so
these are independently unit-testable (tests/test_parsers_eigenstates.py)
without needing a live Elk process.
"""

import numpy as np


def _take(tokens, pos, n, cast):
    vals = [cast(t) for t in tokens[pos : pos + n]]
    return vals, pos + n


def parse_eigenstates_response(tokens):
    """Parse the token stream of an EIGENSTATES response: nstsv, then nstsv
    eigenvalues (Hartree), then the nstsv x nstsv evecsv matrix as real/imag
    pairs.

    Returns (energies, evecsv): energies shape (nstsv,), evecsv shape
    (nstsv, nstsv) complex -- evecsv[:, i] is the i-th second-variational
    eigenvector's coefficients in the orthonormal spinor basis of this one
    diagonalisation. Valid for overlap-style inner products only among
    columns of this SAME array -- comparing evecsv across two different
    queries (different k, or a separate diagonalisation at the same k) is
    not meaningful this way; use EigenstateSession.overlap() for that (see
    docs/design.md #14).
    """
    pos = 0
    (nstsv,), pos = _take(tokens, pos, 1, int)
    energies, pos = _take(tokens, pos, nstsv, float)
    energies = np.array(energies)
    flat, pos = _take(tokens, pos, 2 * nstsv * nstsv, float)
    reim = np.array(flat).reshape(nstsv * nstsv, 2)
    values = reim[:, 0] + 1j * reim[:, 1]
    # Fortran wrote "do b; do a" with a innermost -- column-major, matching
    # evecsv(a, b)'s own natural storage order (same convention as
    # parsers/berry.py).
    evecsv = values.reshape(nstsv, nstsv, order="F")
    return energies, evecsv


def parse_overlap_response(tokens):
    """Parse the token stream of an OVERLAP response: nst, then the
    nst x nst overlap matrix M(a, b) = <psi_a(k_a)|psi_b(k_b)> as real/imag
    pairs (same column-major convention as parse_eigenstates_response)."""
    pos = 0
    (nst,), pos = _take(tokens, pos, 1, int)
    flat, pos = _take(tokens, pos, 2 * nst * nst, float)
    reim = np.array(flat).reshape(nst * nst, 2)
    values = reim[:, 0] + 1j * reim[:, 1]
    return values.reshape(nst, nst, order="F")


def parse_projection_response(tokens):
    """Parse the token stream of a PROJECTION response: nst, natmtot, then
    natmtot consecutive nst x nst atom-projection matrices (real/imag pairs,
    same column-major convention as parse_eigenstates_response/
    parse_overlap_response), one per atom, in Fortran's global 1-based atom
    order (species in declaration order, then atoms within each species in
    order -- see Calculation.get_forces()'s docstring for the same
    convention, and Calculation.get_atom_projection() for how to look up a
    particular atom's matrix by (species, index)).

    Returns an (natmtot, nst, nst) complex array; matrices[ias] is the
    atom-projection operator P_ias (0-based ias here, 1-based in the
    Fortran/protocol sense) -- see EigenstateSession.atom_projection() for
    what each matrix element means and its gauge caveat.
    """
    pos = 0
    (nst, natmtot), pos = _take(tokens, pos, 2, int)
    flat, pos = _take(tokens, pos, 2 * nst * nst * natmtot, float)
    reim = np.array(flat).reshape(nst * nst * natmtot, 2)
    values = reim[:, 0] + 1j * reim[:, 1]
    # Fortran wrote "do ias; do b; do a" with a innermost -- so within each
    # atom's block the layout is column-major (matching the other parsers
    # here), and the natmtot blocks themselves are consecutive.
    return values.reshape(nst, nst, natmtot, order="F").transpose(2, 0, 1)


def parse_orbital_projection_response(tokens):
    """Parse the token stream of an ORBITAL response: nst, natmtot, nl (always
    4: s, p, d, f), then natmtot*nl consecutive nst x nst orbital-projection
    matrices (real/imag pairs, same column-major convention as the other
    parsers here), in Fortran's "do ias; do lsel" nesting -- atom-major,
    l-minor (l=0..3 i.e. s,p,d,f, in that order) within each atom's block --
    same global 1-based atom order as parse_projection_response.

    Returns an (natmtot, 4, nst, nst) complex array; matrices[ias, l] is the
    l-resolved atom-projection operator P_{ias,l} (0-based ias/l here) --
    see EigenstateSession.orbital_projection() for what each matrix element
    means and its gauge caveat (same as atom_projection()'s).
    """
    pos = 0
    (nst, natmtot, nl), pos = _take(tokens, pos, 3, int)
    flat, pos = _take(tokens, pos, 2 * nst * nst * nl * natmtot, float)
    reim = np.array(flat).reshape(nst * nst * nl * natmtot, 2)
    values = reim[:, 0] + 1j * reim[:, 1]
    # column-major within each (ias, l) block, blocks consecutive with l
    # fastest-varying after (a, b) -- i.e. Fortran's "do ias; do lsel; do b;
    # do a" nesting, a innermost.
    return values.reshape(nst, nst, nl, natmtot, order="F").transpose(3, 2, 0, 1)


def parse_momentum_response(tokens):
    """Parse the token stream of a MOMENTUM response: nstsv, then nstsv
    eigenvalues (Hartree), then the nstsv x nstsv evecsv matrix, then the
    three nstsv x nstsv Cartesian momentum matrices p^x, p^y, p^z, all as
    real/imag pairs (same column-major convention as the other parsers
    here, Cartesian component slowest-varying).

    The leading nstsv/eigenvalues/evecsv block is deliberately byte-for-byte
    the shape parse_eigenstates_response() reads, with the momentum
    components appended -- a MOMENTUM response IS an EIGENSTATES response
    plus p^a. evecsv was added by patches/0009 for the reason docs/design.md
    #24 explains: S_a is built from evecsv alone (parsers.spin), so the spin
    current operator J^z_a = (1/2){S_z, v_a} is only well defined if that
    evecsv and this pmat come from the SAME diagonalisation. Taking evecsv
    from a separate EIGENSTATES query would silently mix two arbitrary
    resolutions of any degenerate multiplet (docs/design.md #14) while
    leaving every Hermiticity and unitarity check intact.

    Returns (energies, evecsv, pmat): energies shape (nstsv,) Hartree,
    evecsv shape (nstsv, nstsv) complex (evecsv[:, i] the i-th
    eigenvector, row index i = p + (ispn-1)*nstfv), pmat shape
    (3, nstsv, nstsv) complex, pmat[a] the a-th Cartesian component
    (a = 0, 1, 2 for x, y, z) in atomic units, indexed
    pmat[a][n, m] = <psi_n|p_a|psi_m> -- i.e. the CONJUGATED (bra) state
    on the row. That index convention is not a guess: genpmatk.f90's own
    header defines P_ij = integral Psi_i^*(-i grad + ...) Psi_j, and its
    accumulation is zgemv('C', ...) into pmat(1, jst, i), which places
    the conjugated factor on the first (row) index. It matters -- a
    transposed pmat flips both the Kubo Berry curvature and the circular
    polarization together, so no internal consistency check between them
    would catch it. Unlike the projection
    responses this carries NO band window -- genpmatk's array is
    hard-dimensioned nstsv (see elkpy_momentum), and the Kubo-style sums
    this feeds need states outside any window of interest anyway.

    The energies are returned alongside deliberately: they come from the
    same diagonalisation as pmat, which is what makes them safe to use as
    the energy denominators of those sums. Pairing pmat with a separate
    EIGENSTATES query's energies would reintroduce the degenerate-subspace
    basis ambiguity of docs/design.md #14.
    """
    pos = 0
    (nstsv,), pos = _take(tokens, pos, 1, int)
    energies, pos = _take(tokens, pos, nstsv, float)
    energies = np.array(energies)
    flat, pos = _take(tokens, pos, 2 * nstsv * nstsv, float)
    reim = np.array(flat).reshape(nstsv * nstsv, 2)
    # same "do b; do a" column-major block parse_eigenstates_response reads
    evecsv = (reim[:, 0] + 1j * reim[:, 1]).reshape(nstsv, nstsv, order="F")
    flat, pos = _take(tokens, pos, 2 * 3 * nstsv * nstsv, float)
    reim = np.array(flat).reshape(3 * nstsv * nstsv, 2)
    values = reim[:, 0] + 1j * reim[:, 1]
    # Fortran wrote "do comp; do b; do a" with a innermost -- column-major
    # within each Cartesian component's block, the three blocks consecutive.
    pmat = values.reshape(nstsv, nstsv, 3, order="F").transpose(2, 0, 1)
    return energies, evecsv, pmat


def parse_parity_response(tokens):
    """Parse the token stream of a PARITY response: nst, nstsv, then nstsv
    eigenvalues (Hartree), then the nst x nst inversion-operator matrix
    P(a, b) = <psi_a|I|psi_b> as real/imag pairs (same column-major
    convention as the other parsers here).

    Returns (energies, pmat): energies shape (nstsv,) Hartree -- ALL states,
    not just the window -- and pmat shape (nst, nst) complex over the
    requested band window. The full eigenvalue list travels with the
    windowed operator (same reasoning as parse_momentum_response) so the
    caller can see the degeneracy structure around the window and confirm
    its boundary does not cut a Kramers pair.

    P is Hermitian with P^2 = 1 and eigenvalues +-1 whenever the window is
    gapped from the rest of the spectrum -- see
    EigenstateSession.parity() for why that survives the truncation here
    when the analogous angular-momentum identities (docs/design.md #19) do
    not.
    """
    pos = 0
    (nst, nstsv), pos = _take(tokens, pos, 2, int)
    energies, pos = _take(tokens, pos, nstsv, float)
    energies = np.array(energies)
    flat, pos = _take(tokens, pos, 2 * nst * nst, float)
    reim = np.array(flat).reshape(nst * nst, 2)
    values = reim[:, 0] + 1j * reim[:, 1]
    return energies, values.reshape(nst, nst, order="F")


def parse_angular_momentum_response(tokens):
    """Parse the token stream of an ANGMOM response: nst, natmtot, nl (always
    4: s, p, d, f), ncomp (always 3: x, y, z), then natmtot*nl*ncomp
    consecutive nst x nst angular-momentum matrices (real/imag pairs, same
    column-major convention as the other parsers here), in Fortran's
    "do ias; do lsel; do comp" nesting -- atom-major, then l (l=0..3 i.e.
    s,p,d,f), then Cartesian component (x,y,z) minor -- same global 1-based
    atom order as parse_projection_response.

    Returns an (natmtot, 4, 3, nst, nst) complex array; matrices[ias, l, c]
    is the l-resolved angular-momentum operator (L_x, L_y or L_z, c=0,1,2)
    for atom ias -- see EigenstateSession.angular_momentum() for what each
    matrix element means.
    """
    pos = 0
    (nst, natmtot, nl, ncomp), pos = _take(tokens, pos, 4, int)
    flat, pos = _take(tokens, pos, 2 * nst * nst * ncomp * nl * natmtot, float)
    reim = np.array(flat).reshape(nst * nst * ncomp * nl * natmtot, 2)
    values = reim[:, 0] + 1j * reim[:, 1]
    # column-major within each (ias, l, comp) block, blocks consecutive with
    # comp fastest-varying after (a, b), then l, then ias -- i.e. Fortran's
    # "do ias; do lsel; do comp; do b; do a" nesting, a innermost.
    return values.reshape(nst, nst, ncomp, nl, natmtot, order="F").transpose(
        4, 3, 2, 0, 1
    )
