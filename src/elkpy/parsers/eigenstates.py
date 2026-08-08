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
