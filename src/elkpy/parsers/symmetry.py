"""Parity eigenvalues at the time-reversal-invariant momenta, and the
Fu-Kane symmetry-indicator Z2 invariants built from them.

Like parsers/berry.py, parsers/wilson.py and parsers/optical.py this is
"arithmetic, not just parsing": the Fortran side (elkpy_parity,
patches/0008) returns only the inversion-operator matrix
P_mn = <psi_m|I|psi_n> and the eigenvalues of the same diagonalisation;
every invariant built on them lives here in pure NumPy, unit-testable on
synthetic data without an Elk run (tests/test_parsers_symmetry.py).

The physics (Fu & Kane, *Topological insulators with inversion symmetry*,
PRB 76, 045302 (2007)): for a crystal with an inversion centre, the Z2
invariants of a time-reversal-invariant insulator are fixed entirely by
parity eigenvalues at the 8 (3D) or 4 (2D) time-reversal-invariant momenta
(TRIM), with no Brillouin-zone integration at all --

    delta_i = prod_{m=1}^{N} xi_{2m}(Gamma_i),        (FKM eq. 1.2)
    (-1)^nu_0 = prod_{i=1}^{8} delta_i,               (FKM eq. 1.3)
    (-1)^nu_k = prod over the 4 TRIM with k_k = 1/2,

where xi = +-1 are parity eigenvalues and the product in delta_i runs over
ONE member of each Kramers pair. This is the cheap counterpart to the
Wannier-charge-center pumping of parsers/wilson.py (docs/design.md #20/#21):
a handful of k-points instead of a mesh sweep, at the cost of requiring the
inversion symmetry to actually be present.

See docs/design.md #23 and docs/physics.tex Part XII.

Sign immunity
-------------
A global sign error in P would flip every xi at once. Every invariant here
is a product over an EVEN number of TRIM (8 for nu_0, 4 for each nu_k and
for the 2D nu), so such an error cancels identically and cannot affect any
result. The checks that do have teeth are structural: P Hermitian, P^2 = 1,
eigenvalues +-1, an even occupied count (Kramers), and an even number of
-1 eigenvalues.
"""

import numpy as np

# The 8 TRIM in fractional coordinates: every component 0 or 1/2.
TRIM_3D = tuple(
    (a / 2, b / 2, c / 2) for a in (0, 1) for b in (0, 1) for c in (0, 1)
)
# The 4 TRIM of a 2D system in the first two reciprocal directions (the
# third is a vacuum/stacking direction and is held at 0).
TRIM_2D = tuple((a / 2, b / 2, 0.0) for a in (0, 1) for b in (0, 1))


def is_trim(k, tol=1e-8):
    """True if every fractional component of k is 0 or 1/2 modulo 1, i.e.
    -k == k modulo a reciprocal lattice vector -- the condition for the
    inversion operator to map the k-point to itself, and therefore for a
    parity eigenvalue to exist at all."""
    for x in k:
        frac = float(x) % 1.0
        if min(abs(frac), abs(frac - 0.5), abs(frac - 1.0)) > tol:
            return False
    return True


def parity_eigenvalues(pmat, tol=5e-2):
    """The parity eigenvalues xi = +-1 of the inversion operator restricted
    to a band window, as a sorted real array.

    `pmat` is the (nst, nst) matrix from EigenstateSession.parity(). Because
    [I, H] = 0, inversion preserves energy eigenspaces, so a window gapped
    from the rest of the spectrum is I-invariant, P restricted to it is
    Hermitian with P^2 = 1, and its eigenvalues are exactly +-1. The
    DIAGONAL entries are not the eigenvalues in general -- TRIM spectra are
    heavily degenerate, and within a degenerate multiplet the
    diagonalisation returns an arbitrary basis -- so this diagonalises
    rather than reading the diagonal.

    `tol` is generous (5e-2 by default) because genolpq carries a
    real-space truncation floor of order 1e-3 (docs/design.md #14); the
    eigenvalues are exactly +-1 in exact arithmetic, so anything landing
    between the two is a genuine problem (a window cutting through a
    degenerate multiplet, or a k-point that is not a TRIM), not noise.

    Raises ValueError if P is not Hermitian to `tol`, or if any eigenvalue
    is not within `tol` of +-1.
    """
    pmat = np.asarray(pmat)
    if pmat.ndim != 2 or pmat.shape[0] != pmat.shape[1]:
        raise ValueError(f"pmat must be square, got shape {pmat.shape}")
    asym = np.max(np.abs(pmat - pmat.conj().T)) if pmat.size else 0.0
    if asym > tol:
        raise ValueError(
            f"parity operator is not Hermitian (max |P - P^dagger| = {asym:.3e} > "
            f"{tol:.3e}) -- the band window is probably not inversion-invariant, e.g. "
            f"it cuts through a degenerate multiplet"
        )
    vals = np.linalg.eigvalsh((pmat + pmat.conj().T) / 2)
    off = np.abs(np.abs(vals) - 1.0)
    if np.max(off) > tol:
        bad = vals[off > tol]
        raise ValueError(
            f"parity eigenvalues must be +-1, got {bad} (tolerance {tol:.3e}) -- the "
            f"band window is not a well-separated, inversion-invariant group of bands, "
            f"or this k-point is not a time-reversal-invariant momentum"
        )
    return np.sort(np.sign(vals))


def trim_delta(pmat, tol=5e-2, require_kramers=True):
    """The Fu-Kane delta = prod_m xi_{2m} at one TRIM, as +1 or -1.

    delta is the product of parity eigenvalues over ONE member of each
    Kramers pair (FKM eq. 1.2). With spin-orbit coupling every level is
    two-fold degenerate and both partners share a parity eigenvalue, so
    the product over ALL occupied states is identically +1 and carries no
    information -- the pairing is essential, not a detail. Counting the
    -1 eigenvalues over all occupied states as N_minus, one member per
    pair gives

        delta = (-1)^(N_minus / 2).

    Raises ValueError if the window holds an odd number of states or if
    N_minus is odd; both violate Kramers degeneracy and mean the window
    boundary has split a pair (set `require_kramers=False` only to inspect
    such a case deliberately -- the returned delta is then not an FKM
    delta).
    """
    xi = parity_eigenvalues(pmat, tol=tol)
    n_minus = int(np.sum(xi < 0))
    if require_kramers:
        if len(xi) % 2 != 0:
            raise ValueError(
                f"band window holds an odd number of states ({len(xi)}) -- with "
                f"time-reversal and inversion every level is Kramers-degenerate, so an "
                f"odd count means the window boundary splits a pair"
            )
        if n_minus % 2 != 0:
            raise ValueError(
                f"odd number of negative parity eigenvalues ({n_minus}) -- Kramers "
                f"partners share a parity eigenvalue, so this count must be even; the "
                f"window boundary has split a degenerate pair"
            )
    return 1 if (n_minus // 2) % 2 == 0 else -1


def check_window_gap(energies, ist0, ist1, tol=1e-3):
    """Raise unless the band window [ist0, ist1] (inclusive, 1-based) is
    separated by at least `tol` Hartree from the states immediately below
    and above it.

    The Fu-Kane parity products are only defined for a band group that is
    gapped from the rest of the spectrum everywhere -- the same requirement
    parsers.berry.check_gap enforces for the Wilson-loop construction. This
    is NOT implied by the checks in trim_delta(): a window slicing through
    the middle of a band group can still be Hermitian with +-1 eigenvalues
    and an even Kramers count, and will then silently return a delta for a
    set of bands that is not a topological group at all.

    Found empirically on the [111]-dimerized diamond model, where windows
    starting inside a group (ist0 = 19 or 27, neither above a gap) passed
    every other check and returned nu0 = 1 and 0 respectively, while the
    two legitimate gapped starts (1 and 9, the latter above a 69 eV
    semicore gap) both give nu0 = 0.
    """
    energies = np.asarray(energies, dtype=float)
    n = len(energies)
    if not (1 <= ist0 <= ist1 <= n):
        raise ValueError(f"invalid band window (ist0={ist0}, ist1={ist1}, nstsv={n})")
    if ist0 > 1:
        gap = energies[ist0 - 1] - energies[ist0 - 2]
        if gap < tol:
            raise ValueError(
                f"band window starts at ist0={ist0}, which is not separated from the "
                f"state below it (gap {gap:.3e} Ha < {tol:.3e}) -- the window cuts "
                f"through a band group, so its parity product is not a topological "
                f"invariant of anything"
            )
    if ist1 < n:
        gap = energies[ist1] - energies[ist1 - 1]
        if gap < tol:
            raise ValueError(
                f"band window ends at ist1={ist1}, which is not separated from the "
                f"state above it (gap {gap:.3e} Ha < {tol:.3e}) -- the window is not "
                f"gapped from the conduction states at this k-point"
            )


def fu_kane_z2_2d(deltas):
    """The 2D Z2 invariant nu in {0, 1} from the 4 TRIM deltas.

    (-1)^nu = prod_i delta_i (FKM eq. 1.3 restricted to a plane).
    `deltas` is a mapping {k-point tuple: delta} or an iterable of deltas;
    exactly 4 are required.
    """
    values = list(deltas.values()) if isinstance(deltas, dict) else list(deltas)
    if len(values) != 4:
        raise ValueError(f"expected 4 TRIM deltas for a 2D system, got {len(values)}")
    return 0 if int(np.prod(values)) == 1 else 1


def fu_kane_z2_3d(deltas):
    """The full 3D classification (nu0; nu1 nu2 nu3) from the 8 TRIM deltas.

    `deltas` maps each TRIM (a tuple of three fractional coordinates, each
    0 or 1/2) to its delta = +-1. All 8 must be present.

        (-1)^nu_0 = prod over all 8 TRIM
        (-1)^nu_k = prod over the 4 TRIM with the k-th component = 1/2

    (Fu, Kane & Mele, PRL 98, 106803 (2007)); nu_0 = 1 is a strong
    topological insulator, nu_0 = 0 with some nu_k = 1 a weak one.

    Returns {"nu0": int, "nu": (nu1, nu2, nu3)}.
    """
    if len(deltas) != 8:
        raise ValueError(f"expected 8 TRIM deltas for a 3D system, got {len(deltas)}")
    keyed = {}
    for k, d in deltas.items():
        key = tuple(round(float(x) % 1.0, 6) for x in k)
        if any(abs(x) > 1e-6 and abs(x - 0.5) > 1e-6 for x in key):
            raise ValueError(f"{k} is not a TRIM (components must be 0 or 1/2)")
        keyed[key] = int(d)
    if len(keyed) != 8:
        raise ValueError("the 8 TRIM keys are not distinct modulo a lattice vector")
    nu0 = 0 if int(np.prod(list(keyed.values()))) == 1 else 1
    weak = []
    for axis in range(3):
        prod = int(np.prod([d for key, d in keyed.items() if abs(key[axis] - 0.5) < 1e-6]))
        weak.append(0 if prod == 1 else 1)
    return {"nu0": nu0, "nu": tuple(weak)}
