"""Quantum geometric tensor (quantum metric + Berry curvature) from
nearest-neighbour wavefunction overlaps -- the companion quantity to
parsers/berry.py's Wilson-loop Berry curvature, built from the *same* kind
of overlap matrix but needing one extra ingredient (see
_normalize_overlap() below).

Unlike berry.py, no new Fortran task backs this: every overlap consumed
here (including each corner's own self-overlap) is already exposed by the
task-9002 interactive session (session.py / patches/0003-eigenstate-session.patch),
so Calculation.get_quantum_geometry() drives this purely from
EigenstateSession.overlap() queries. See docs/design.md #15 and
docs/physics.tex Part IV for the physics writeup (quantum metric,
Fubini-Study distance, the Marzari-Vanderbilt/Resta projector-trace
derivation) and why the metric -- unlike curvature -- needs the
Loewdin-normalization step this module adds.
"""

import numpy as np

from .berry import _berry_phase, _link_variable


def _hermitian_inv_sqrt(mat, floor=1e-8):
    """S^{-1/2} for a Hermitian positive-definite self-overlap matrix via
    eigendecomposition.

    Raises ValueError if any eigenvalue is at or below `floor`, the same
    fail-loud choice parsers.berry._link_variable makes for a singular
    overlap determinant, rather than silently clipping: a self-overlap
    this close to singular means the requested band window is not well
    resolved at this k-point (e.g. too close to a degeneracy/band
    crossing for the corner's own diagonalisation to be trusted), and
    clipping would substitute a different matrix and return a
    plausible-looking but meaningless metric instead of surfacing that.
    genolpq's own real-space truncation error can push a self-overlap
    eigenvalue that should be exactly 1 down by ~1e-3 (docs/design.md
    #14) -- nowhere near this floor for a genuinely well-gapped window,
    so `floor` only guards against a real problem, not routine noise.
    """
    vals, vecs = np.linalg.eigh(mat)
    if np.any(vals <= floor):
        raise ValueError(
            f"self-overlap has eigenvalue(s) <= {floor} ({vals[vals <= floor]}) -- the "
            f"requested band window is not well resolved at this k-point (e.g. too close "
            f"to a degeneracy for this corner's diagonalisation to be trusted); Loewdin "
            f"normalization would be meaningless here, not just noisy"
        )
    return (vecs * (vals**-0.5)) @ vecs.conj().T


def _normalize_overlap(m, s_a, s_b):
    """Loewdin symmetric normalization M -> S_a^{-1/2} M S_b^{-1/2}.

    Elk's `genolpq` overlap has a real-space truncation floor of order
    1e-3 (CLAUDE.md: overlap(k,k,...) is the identity only to that
    tolerance, not exactly). Berry curvature (parsers.berry) is immune to
    this: it's built from -arg(det(.)) around a *closed* loop, where a
    common-mode modulus error in every link variable cancels exactly. The
    quantum metric is not -- quantum_distance() below is built directly
    from |M|^2, so an overlap deficient by a relative factor (1-eps)
    contributes ~2*eps*J to quantum_distance() (J = band-window size), a
    constant offset independent of the true metric. At a typical loop size
    dk~0.005-0.01 and eps~1e-3 this offset is many times the physical
    metric's own O(dk^2) signal -- unusable uncorrected. Loewdin
    normalization removes it to leading order because it is exactly the
    transformation that forces the normalized self-overlap
    S_a^{-1/2} S_a S_a^{-1/2} = 1 identically at every corner, not just
    approximately.
    """
    sa_inv = _hermitian_inv_sqrt(s_a)
    sb_inv = _hermitian_inv_sqrt(s_b)
    return sa_inv @ m @ sb_inv


def quantum_distance(m):
    """D(v) = J - Re Tr[M M^dagger] for a Loewdin-normalized window overlap
    M(k, k+v) between two J-dimensional occupied subspaces.

    This is the same bracket term as Marzari & Vanderbilt's gauge-invariant
    spread (PRB 56, 12847 (1997), eq. 27/30: Omega_I ~ sum_b w_b [J -
    Tr(M_b M_b^dagger)]), equivalently Resta's projector distance
    Tr[P(k) Q(k+v)], Q = 1-P. To leading order in the displacement v,
    D(v) = g_ab v^a v^b, the quantum metric evaluated on v -- see
    docs/physics.tex Part IV for the Tr[P dP dP] derivation.
    """
    j = m.shape[0]
    return j - float(np.real(np.trace(m @ m.conj().T)))


def compute_quantum_geometry(overlaps, v1, v2):
    """Quantum geometric tensor Q_ab = g_ab - (i/2) F_ab at one k-point,
    from raw overlaps on a 3x3 grid of corners centered at k: k, k +/- v1,
    k +/- v2, k + v1 +/- v2, k - v1 +/- v2.

    `overlaps`: dict of (nst, nst) complex arrays --
      "s0": self-overlap <psi(k)|psi(k)>.
      "s1p"/"s1m", "s2p"/"s2m": self-overlaps at k+v1/k-v1, k+v2/k-v2.
      "s12pp"/"s12pm"/"s12mp"/"s12mm": self-overlaps at the four diagonal
          corners k+v1+v2, k+v1-v2, k-v1+v2, k-v1-v2.
      "m1p"/"m1m", "m2p"/"m2m": <psi(k)|psi(k+/-v1)>, <psi(k)|psi(k+/-v2)>
          -- feed the metric's diagonal components.
      "m12pp"/"m12pm"/"m12mp"/"m12mm": <psi(k)|psi(k+v1+v2)>, etc. -- feed
          the metric's off-diagonal component via a centered mixed-partial
          stencil (see below); m12pp also feeds the curvature link
          variable, same role "m12" played before this was centered.
      "edge_b", "edge_c": <psi(k+v1)|psi(k+v1+v2)>, <psi(k+v2)|psi(k+v1+v2)>
          -- the remaining two edges of the closed Wilson loop (FHS eq. 8,
          same convention as parsers.berry.compute_berry_curvature's mesh
          anchor: corners k, k+e1, k+e1+e2, k+e2, not the centered
          convention compute_berry_curvature_path uses). Curvature is
          computed on this forward (non-centered) sub-loop deliberately --
          see below.

    `v1`, `v2`: the two Cartesian (Bohr^-1) displacement vectors used to
      reach the neighbouring corners from k -- their norms set the
      metric's dk normalization, their cross product the loop's area for
      curvature (same convention as parsers.berry's path mode).

    **Why the metric is centered but curvature isn't.** Expanding the
    quantum distance D(v) = J - Re Tr[M(v)M(v)^dagger] in v: the linear
    term vanishes (D(0)=0 is a minimum), the quadratic term is the metric
    g_ab v^a v^b, and the next (cubic) term is generically nonzero and,
    under time reversal, odd in k -- this is what made g12's forward-
    difference estimate (the polarization identity D(v1+v2)-D(v1)-D(v2))
    converge only as O(dk) instead of O(dk^2), and K/K' converge to a
    common value only in the dk->0 limit rather than agreeing tightly at
    one dk (see git log / docs/design.md #15 for the measured 1.83/0.97/0.49
    O(dk) signature this replaces). Using the +/-v corner pairs cancels
    that cubic term exactly, the same way a centered numerical derivative
    outperforms a forward one:
        g11 = [D(v1) + D(-v1)] / (2 dk1^2)
        g22 = [D(v2) + D(-v2)] / (2 dk2^2)
        g12 = [D(v1+v2) + D(-v1-v2) - D(v1-v2) - D(-v1+v2)] / (8 dk1 dk2)
    (g12's stencil is the standard centered mixed-partial-derivative
    formula applied to D; see docs/physics.tex Part IV for the full
    odd-term-cancellation derivation.) Curvature is built from -arg(det M)
    (_berry_phase, parsers.berry -- the sign convention is set there, once,
    for every consumer; see docs/design.md #22)
    around a *closed* loop (FHS eq. 8), which is already exact to O(dk^2)
    on the plain forward sub-loop k, k+v1, k+v1+v2, k+v2 -- centering it
    would cost more corners for no accuracy gain, so it's left as-is
    (also pinned by test_curvature_matches_berry_curvature_path_on_identical_corners's
    specific forward anchoring).

    Returns {"g": (2,2) real array [[g11,g12],[g12,g22]] (quantum metric,
    Bohr^2), "berry_curvature": float (Bohr^2, identical convention/value
    to parsers.berry.compute_berry_curvature_path), "Q": (2,2) complex
    Hermitian array, Q[0,0]=g11, Q[1,1]=g22, Q[0,1]=g12-(i/2)*F_12,
    Q[1,0]=conj(Q[0,1])}.
    """
    s0 = overlaps["s0"]
    m1p = _normalize_overlap(overlaps["m1p"], s0, overlaps["s1p"])
    m1m = _normalize_overlap(overlaps["m1m"], s0, overlaps["s1m"])
    m2p = _normalize_overlap(overlaps["m2p"], s0, overlaps["s2p"])
    m2m = _normalize_overlap(overlaps["m2m"], s0, overlaps["s2m"])
    m12pp = _normalize_overlap(overlaps["m12pp"], s0, overlaps["s12pp"])
    m12pm = _normalize_overlap(overlaps["m12pm"], s0, overlaps["s12pm"])
    m12mp = _normalize_overlap(overlaps["m12mp"], s0, overlaps["s12mp"])
    m12mm = _normalize_overlap(overlaps["m12mm"], s0, overlaps["s12mm"])
    edge_b = _normalize_overlap(overlaps["edge_b"], overlaps["s1p"], overlaps["s12pp"])
    edge_c = _normalize_overlap(overlaps["edge_c"], overlaps["s2p"], overlaps["s12pp"])

    dk1 = float(np.linalg.norm(v1))
    dk2 = float(np.linalg.norm(v2))
    d1p, d1m = quantum_distance(m1p), quantum_distance(m1m)
    d2p, d2m = quantum_distance(m2p), quantum_distance(m2m)
    d12pp, d12mm = quantum_distance(m12pp), quantum_distance(m12mm)
    d12pm, d12mp = quantum_distance(m12pm), quantum_distance(m12mp)

    g11 = (d1p + d1m) / (2 * dk1**2)
    g22 = (d2p + d2m) / (2 * dk2**2)
    g12 = (d12pp + d12mm - d12pm - d12mp) / (8 * dk1 * dk2)

    # FHS eq. 8, parsers.berry's own link-variable convention, corners
    # anchored at k (not centered): w = U1(k)*U2(k+e1) / (U1(k+e2)*U2(k)).
    # Loewdin normalization cannot change this: S_a^{-1/2}/S_b^{-1/2} are
    # Hermitian positive-definite, so det(S^{-1/2}) is real positive and
    # cannot shift arg(det M), and _berry_phase's negation is a global sign
    # that commutes with it -- normalizing here is for uniformity with
    # the metric computation above, not a correctness requirement.
    w = _link_variable(m1p) * _link_variable(edge_b) / (_link_variable(edge_c) * _link_variable(m2p))
    flux = _berry_phase(w)
    area = float(np.linalg.norm(np.cross(v1, v2)))
    curvature = flux / area

    g = np.array([[g11, g12], [g12, g22]])
    q = np.array(
        [[g11 + 0j, g12 - 0.5j * curvature], [g12 + 0.5j * curvature, g22 + 0j]]
    )
    return {"g": g, "berry_curvature": curvature, "Q": q}
