"""Rotation-eigenvalue symmetry indicators for 2D crystalline insulators.

The "arithmetic, not just parsing" module for the SYMMETRY query
(elkpy_symop, patches/0010), in the same spirit as parsers/symmetry.py is
for PARITY: the Fortran side returns only the operator matrix
S_mn = <psi_m|O|psi_n>, and every invariant built on it lives here in pure
NumPy, unit-testable on synthetic data (tests/test_parsers_indicators.py).

THE PHYSICS. A C_n rotation commutes with H, so at a k-point it leaves
invariant the occupied bands carry definite rotation eigenvalues

    Pi_p^(n) = exp(2 pi i (p - 1) / n),   p = 1 .. n

(spinless; a spinor would carry half-integer angular momentum and is not
supported -- see EigenstateSession.symmetry_operator()). Counting how many
occupied bands carry each eigenvalue at each high-symmetry point, RELATIVE
to Gamma, gives integer topological indices. Benalcazar, Li & Hughes,
"Quantization of fractional corner charge in C_n-symmetric higher-order
topological crystalline insulators", PRB 99, 245151 (2019),
arXiv:1809.02142, their Eq. 3:

    [Pi_p^(n)] = #Pi_p^(n) - #Gamma_p^(n)

with #Pi_p^(n) the number of occupied bands with eigenvalue Pi_p^(n) at the
high-symmetry point Pi. Their Eq. 4 gives the independent index sets, and
Eq. 14 the fractional corner charge:

    chi^(2) = ([X_1^(2)], [Y_1^(2)], [M_1^(2)])
    chi^(3) = ([K_1^(3)], [K_2^(3)])
    chi^(4) = ([X_1^(2)], [M_1^(4)], [M_2^(4)])
    chi^(6) = ([M_1^(2)], [K_1^(3)])

    Q^(2) = (e/4) (-[X_1^(2)] - [Y_1^(2)] + [M_1^(2)])          mod e
    Q^(3) = (e/3) [K_2^(3)]                                      mod e
    Q^(4) = (e/4) ([X_1^(2)] + 2[M_1^(4)] + 3[M_2^(4)])          mod e
    Q^(6) = (e/4) [M_1^(2)] + (e/6) [K_1^(3)]                    mod e

A nonzero Q is the hallmark of an obstructed atomic limit: the occupied
Wannier centers sit at a Wyckoff position other than the rotation centre,
which pins a fractional charge at a corner of a symmetric flake.

The general framework these sit inside -- symmetry indicators for all 230
space groups -- is Po, Vishwanath & Watanabe, Nat. Commun. 8, 50 (2017),
arXiv:1703.00911. Only the 2D C_n case is implemented here.

WHAT THE INDICES ARE RELATIVE TO. Q and chi are defined with respect to a
CHOSEN rotation centre (BLH fix a maximal Wyckoff position). Elk chooses its
own origin -- `tshift` moves the basis -- so the centre these indices
describe is whatever Elk put at the origin, which need not be the one a
paper assumed. This is the same trap parsers/symmetry.py documents for the
per-TRIM parity deltas, one level more consequential: there a wrong origin
permuted the deltas while leaving nu alone, whereas here it changes Q
itself. Always report the centre alongside the number.
"""

import numpy as np


def rotation_eigenvalues(smat, order, tol=5e-2):
    """The rotation eigenvalues of a C_n operator matrix, as an array of
    integers p in 1..n with Pi_p = exp(2 pi i (p-1)/n).

    `smat` is the (nst, nst) matrix from
    EigenstateSession.symmetry_operator() over a band window that must be
    gapped from the rest of the spectrum (use
    parsers.symmetry.check_window_gap). Since [O, H] = 0 the window is then
    O-invariant, O restricted to it is unitary with O^n = 1, and its
    eigenvalues are exactly n-th roots of unity.

    Eigenvalues come from diagonalising, never from the diagonal: at a
    high-symmetry point the spectrum is degenerate and the diagonalisation
    returns an arbitrary basis within each multiplet, in which the diagonal
    entries of O are not roots of unity at all.

    Raises ValueError if the matrix is not unitary to `tol`, or if any
    eigenvalue is further than `tol` from the nearest n-th root of unity --
    both indicate a window that is not an invariant band group, or a
    k-point not actually fixed by the operation.
    """
    smat = np.asarray(smat)
    n = int(order)
    if n < 1:
        raise ValueError(f"rotation order must be >= 1, got {order}")
    ident = np.eye(len(smat))
    dev = np.max(np.abs(smat.conj().T @ smat - ident)) if smat.size else 0.0
    if dev > tol:
        raise ValueError(
            f"symmetry operator is not unitary (max |O^dag O - 1| = {dev:.3e} > "
            f"{tol:.3e}) -- the band window is probably not invariant under this "
            f"operation, e.g. it cuts through a degenerate multiplet"
        )
    vals = np.linalg.eigvals(smat)
    roots = np.exp(2j * np.pi * np.arange(n) / n)
    p = np.argmin(np.abs(vals[:, None] - roots[None, :]), axis=1)
    off = np.abs(vals - roots[p])
    if np.max(off, initial=0.0) > tol:
        bad = vals[off > tol]
        raise ValueError(
            f"eigenvalues must be {n}-th roots of unity, got {bad} (tolerance "
            f"{tol:.3e}) -- this k-point is probably not invariant under the "
            f"operation, or the window is not an invariant band group"
        )
    return p + 1


def eigenvalue_counts(smat, order, tol=5e-2):
    """#Pi_p^(n) for p = 1..n: how many occupied bands carry each rotation
    eigenvalue. Returns an integer array of length `order`, which sums to
    the number of bands in the window (asserted -- every band must carry
    some eigenvalue, so a short sum means eigenvalues were mis-binned)."""
    p = rotation_eigenvalues(smat, order, tol=tol)
    counts = np.bincount(p - 1, minlength=int(order))
    if counts.sum() != len(p):
        raise ValueError("eigenvalue counts do not sum to the band count")
    return counts


def relative_indices(counts_at_point, counts_at_gamma):
    """BLH Eq. 3: [Pi_p^(n)] = #Pi_p^(n) - #Gamma_p^(n), as an integer
    array. Note [Gamma_p] is identically zero by construction, which is a
    useful sanity check rather than a result."""
    a = np.asarray(counts_at_point, dtype=int)
    b = np.asarray(counts_at_gamma, dtype=int)
    if a.shape != b.shape:
        raise ValueError(f"count arrays must match in length, got {a.shape} vs {b.shape}")
    return a - b


def corner_charge(indices, n):
    """The fractional corner charge in units of e, BLH Eq. 14, from the
    relative indices of the high-symmetry points that C_n needs.

    `indices` is a mapping from a label to the relative-index ARRAY at that
    point, as returned by relative_indices():

        n = 2: {"X": [...], "Y": [...], "M": [...]}   (all order 2)
        n = 3: {"K": [...]}                           (order 3)
        n = 4: {"X": [...], "M": [...]}               (X order 2, M order 4)
        n = 6: {"M": [...], "K": [...]}               (M order 2, K order 3)

    Indexing follows BLH's labels: [Pi_p] is `indices[label][p-1]`, so
    [K_2^(3)] is indices["K"][1].

    Returns the charge modulo e, in [0, 1).
    """
    n = int(n)
    if n == 2:
        q = (-indices["X"][0] - indices["Y"][0] + indices["M"][0]) / 4
    elif n == 3:
        q = indices["K"][1] / 3
    elif n == 4:
        q = (indices["X"][0] + 2 * indices["M"][0] + 3 * indices["M"][1]) / 4
    elif n == 6:
        q = indices["M"][0] / 4 + indices["K"][0] / 6
    else:
        raise ValueError(f"corner charge is defined for n = 2, 3, 4, 6; got {n}")
    return float(q % 1.0)


def find_rotation(ops, order, axis=2):
    """Pick a C_n rotation about a given axis out of `symmetries()`'s list.

    `ops` is EigenstateSession.symmetries()' output; `axis` is a 0-based
    lattice direction (2 = the third lattice vector, i.e. the out-of-plane
    axis of a 2D slab). Returns the matching op dict, or raises.

    A C_n about `axis` is recognised by acting as the identity on that axis
    and having order exactly n: R^n = 1 with no smaller power equal to 1.
    Only symmorphic (zero-translation) operations are considered, since
    those are the only ones symmetry_operator() accepts.
    """
    n = int(order)
    for op in ops:
        if not op["symmorphic"]:
            continue
        r = op["rotation"]
        if r[axis, axis] != 1 or np.any(r[axis, [i for i in range(3) if i != axis]] != 0):
            continue
        if np.any(r[[i for i in range(3) if i != axis], axis] != 0):
            continue
        power, m = 1, r.copy()
        while power <= n and not np.array_equal(m, np.eye(3, dtype=int)):
            m = m @ r
            power += 1
        if power == n:
            return op
    raise ValueError(
        f"no symmorphic C_{n} rotation about lattice axis {axis} among the "
        f"{len(ops)} crystal symmetries"
    )


def fixes_kpoint(op, k, tol=1e-8):
    """True if the operation leaves `k` invariant modulo a reciprocal
    lattice vector, i.e. R^T k == k (mod 1) in fractional coordinates.

    The rotation acts on reciprocal-space fractional coordinates by the
    TRANSPOSE of its real-space lattice matrix, which is why this is not
    simply `R @ k`. Integer arithmetic in lattice coordinates makes this
    exact up to the modulo, with no Cartesian tolerance.
    """
    kk = np.asarray(k, dtype=float)
    rk = np.asarray(op["rotation"], dtype=float).T @ kk
    diff = rk - kk
    return bool(np.all(np.abs(diff - np.round(diff)) < tol))
