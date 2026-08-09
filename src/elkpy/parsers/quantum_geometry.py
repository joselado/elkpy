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

from .berry import _link_variable


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
    this: it's built from arg(det(.)) around a *closed* loop, where a
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
    from raw overlaps between the loop corners k, k+v1, k+v2, k+v1+v2.

    `overlaps`: dict of (nst, nst) complex arrays --
      "s0", "s1", "s2", "s12": self-overlaps <psi(c)|psi(c)> at corners
          k, k+v1, k+v2, k+v1+v2 respectively (needed only for Loewdin
          normalization, not physical inputs in their own right).
      "m1", "m2", "m12": <psi(k)|psi(k+v1)>, <psi(k)|psi(k+v2)>,
          <psi(k)|psi(k+v1+v2)> -- m1/m2 feed both the metric's diagonal
          components and (as FHS link variables) the curvature; m12 feeds
          only the metric's off-diagonal component via the polarization
          identity D(v1+v2) = D(v1) + D(v2) + 2*g(v1,v2).
      "edge_b", "edge_c": <psi(k+v1)|psi(k+v1+v2)>,
          <psi(k+v2)|psi(k+v1+v2)> -- the remaining two edges of the closed
          Wilson loop (FHS eq. 8, same convention as
          parsers.berry.compute_berry_curvature's mesh anchor: corners
          k, k+e1, k+e1+e2, k+e2, not the centered convention
          compute_berry_curvature_path uses).

    `v1`, `v2`: the two Cartesian (Bohr^-1) displacement vectors used to
      reach the neighbouring corners from k -- their norms set the
      metric's dk normalization, their cross product the loop's area for
      curvature (same convention as parsers.berry's path mode).

    Returns {"g": (2,2) real array [[g11,g12],[g12,g22]] (quantum metric,
    Bohr^2), "berry_curvature": float (Bohr^-2, identical convention/value
    to parsers.berry.compute_berry_curvature_path), "Q": (2,2) complex
    Hermitian array, Q[0,0]=g11, Q[1,1]=g22, Q[0,1]=g12-(i/2)*F_12,
    Q[1,0]=conj(Q[0,1])}.
    """
    m1 = _normalize_overlap(overlaps["m1"], overlaps["s0"], overlaps["s1"])
    m2 = _normalize_overlap(overlaps["m2"], overlaps["s0"], overlaps["s2"])
    m12 = _normalize_overlap(overlaps["m12"], overlaps["s0"], overlaps["s12"])
    edge_b = _normalize_overlap(overlaps["edge_b"], overlaps["s1"], overlaps["s12"])
    edge_c = _normalize_overlap(overlaps["edge_c"], overlaps["s2"], overlaps["s12"])

    dk1 = float(np.linalg.norm(v1))
    dk2 = float(np.linalg.norm(v2))
    d1 = quantum_distance(m1)
    d2 = quantum_distance(m2)
    d12 = quantum_distance(m12)
    g11 = d1 / dk1**2
    g22 = d2 / dk2**2
    g12 = (d12 - d1 - d2) / (2 * dk1 * dk2)

    # FHS eq. 8, parsers.berry's own link-variable convention, corners
    # anchored at k (not centered): w = U1(k)*U2(k+e1) / (U1(k+e2)*U2(k)).
    # Loewdin normalization cannot change this: S_a^{-1/2}/S_b^{-1/2} are
    # Hermitian positive-definite, so det(S^{-1/2}) is real positive and
    # cannot shift arg(det M) -- normalizing here is for uniformity with
    # the metric computation above, not a correctness requirement.
    w = _link_variable(m1) * _link_variable(edge_b) / (_link_variable(edge_c) * _link_variable(m2))
    flux = float(np.angle(w))
    area = float(np.linalg.norm(np.cross(v1, v2)))
    curvature = flux / area

    g = np.array([[g11, g12], [g12, g22]])
    q = np.array(
        [[g11 + 0j, g12 - 0.5j * curvature], [g12 + 0.5j * curvature, g22 + 0j]]
    )
    return {"g": g, "berry_curvature": curvature, "Q": q}
