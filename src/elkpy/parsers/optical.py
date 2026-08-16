"""Optical matrix elements and Kubo-form quantum geometry, from the
momentum (velocity) operator exported by the task-9002 MOMENTUM query.

Like parsers/berry.py and parsers/wilson.py this is "arithmetic, not just
parsing": the Fortran side (elkpy_momentum, patches/0007) does nothing but
hand back upstream genpmatk's matrix elements plus the eigenvalues of the
same diagonalisation, and every physical quantity built on them lives here
in Python, unit-testable against synthetic models without an Elk run
(tests/test_parsers_optical.py).

Two families live here:

  * `circular_polarization()` -- the degree of circular polarization of an
    interband transition, i.e. valley-selective circular dichroism.
  * `kubo_berry_curvature()` / `kubo_quantum_metric()` /
    `kubo_quantum_geometry()` -- the quantum geometric tensor in its
    sum-over-states (perturbation-theory) form, an INDEPENDENT code path
    for quantities parsers/berry.py and parsers/quantum_geometry.py
    already compute from finite-difference wavefunction overlaps.

`kubo_sum()`, the shared sum-over-states core, is public rather than
private because parsers/spin_hall.py reuses it with the spin current
operator J^s_a = (1/2){S_s, v_a} in place of one velocity factor -- the
spin Berry curvature is this same expression, so it should share this
same arithmetic and hence this same sign convention (docs/design.md #24).

See docs/design.md #22 and docs/physics.tex Part XI for the physics.

Units and conventions
---------------------
`pmat` is in Hartree atomic units, `energies` in Hartree. In atomic units
(hbar = m_e = 1) and for Elk's LOCAL Kohn-Sham potential, the velocity and
momentum operators coincide numerically, v = p -- this is what makes the
Kubo formulas below usable with genpmatk's output directly. genpmatk
includes the (1/4c^2)[sigma x grad V_s] spin-orbit correction that keeps
that identity true under spinorb=True (Rathgen and Katsnelson, Physica
Scripta T109, 170 (2004)).

Curvature and metric come out in Bohr^2, matching parsers/berry.py's and
parsers/quantum_geometry.py's own values.

`directions` here indexes CARTESIAN axes (1, 2, 3 = x, y, z), because
that is what genpmatk's three components are. This differs from the
identically-named argument of Calculation.get_berry_curvature() /
get_quantum_geometry(), which indexes RECIPROCAL LATTICE axes (b1, b2,
b3). The two coincide only when the relevant reciprocal vectors happen to
span the same plane as the chosen Cartesian pair -- true for the (1, 2)
default on a c-axis-out-of-plane 2D cell (graphene/h-BN/WSe2, where b1 and
b2 span xy, so both conventions mean "the z component of the curvature"),
not in general.
"""

import numpy as np


def _band_group(spec, nstsv):
    """Resolve a band specification to a 0-based index array.

    Accepts a 1-based integer (one band), a (lo, hi) tuple (inclusive
    1-based range, matching the ist0/ist1 convention used throughout
    elkpy), or an iterable of 1-based indices.
    """
    if isinstance(spec, (int, np.integer)):
        idx = [int(spec)]
    elif isinstance(spec, tuple) and len(spec) == 2:
        lo, hi = int(spec[0]), int(spec[1])
        if hi < lo:
            raise ValueError(f"empty band range {spec}")
        idx = list(range(lo, hi + 1))
    else:
        idx = [int(i) for i in spec]
    for i in idx:
        if i < 1 or i > nstsv:
            raise ValueError(f"band index {i} outside 1..{nstsv}")
    return np.array(idx) - 1


def circular_polarization(pmat, valence, conduction, directions=(1, 2)):
    """Degree of circular polarization (optical selectivity) eta of the
    interband transition from `valence` to `conduction` at one k-point.

    With a = directions[0], b = directions[1] the two in-plane Cartesian
    axes, the circular-basis interband matrix elements are

        P_pm = p^a_cv +- i p^b_cv,     p^a_cv = <psi_c|p_a|psi_v>

    and

        eta = (|P_+|^2 - |P_-|^2) / (|P_+|^2 + |P_-|^2),

    the normalized difference in absorption strength between left- and
    right-circularly polarized light. eta = +1 means the transition
    couples exclusively to sigma+ light, eta = -1 exclusively to sigma-.
    In a gapped honeycomb lattice (gapped graphene, h-BN, a monolayer
    group-VI transition-metal dichalcogenide) the three-fold rotation
    symmetry at the zone corner forces |eta| = 1 there, with opposite
    sign at the two inequivalent valleys K and K' -- valley-selective
    circular dichroism (Yao, Xiao and Niu, PRB 77, 235406 (2008); Xiao,
    Liu, Feng, Xu and Yao, PRL 108, 196802 (2012); Cao et al., Nat.
    Commun. 3, 887 (2012)).

    `valence`/`conduction` each accept a 1-based band index, an inclusive
    (lo, hi) range tuple, or an iterable of indices. When either is a
    group, |P_pm|^2 is SUMMED over all (c, v) pairs in it -- the correct
    handling for a degenerate manifold, where no individual pair's matrix
    element is separately meaningful (the same degenerate-group rule
    docs/design.md #13/#14 already document for band windows). Note the
    converse trap: lumping two NON-degenerate bands together (e.g. both
    partners of a spin-orbit-split valence-band top) averages their
    separate selectivities and dilutes eta, which is a physically
    different question from the band-edge transition's own selectivity.

    Returns {"eta": float, "i_plus": float, "i_minus": float} -- the
    latter two being |P_+|^2 and |P_-|^2 (atomic units squared), whose
    absolute scale is the transition's oscillator strength.

    Raises ValueError if both intensities vanish (a dipole-forbidden
    transition, where eta is undefined rather than zero).
    """
    a, b = int(directions[0]), int(directions[1])
    if not (1 <= a <= 3 and 1 <= b <= 3) or a == b:
        raise ValueError(f"directions must be two distinct Cartesian axes in 1..3, got {directions}")
    nstsv = pmat.shape[-1]
    iv = _band_group(valence, nstsv)
    ic = _band_group(conduction, nstsv)
    # p^a_cv for every (c, v) pair in the two groups
    pa = pmat[a - 1][np.ix_(ic, iv)]
    pb = pmat[b - 1][np.ix_(ic, iv)]
    p_plus = pa + 1j * pb
    p_minus = pa - 1j * pb
    i_plus = float(np.sum(np.abs(p_plus) ** 2))
    i_minus = float(np.sum(np.abs(p_minus) ** 2))
    total = i_plus + i_minus
    if total <= 0.0:
        raise ValueError(
            "both circular intensities vanish -- this transition is dipole-forbidden "
            "in the chosen plane, so its degree of circular polarization is undefined"
        )
    return {"eta": (i_plus - i_minus) / total, "i_plus": i_plus, "i_minus": i_minus}


def kubo_sum(energies, op_a, op_b, ist0, ist1, degeneracy_tol=1e-4):
    """Shared core of the Kubo-form geometric quantities: the complex
    sum-over-states tensor

        T_ab = sum_{n in W} sum_{m not in W}
                   <n|A|m><m|B|n> / (eps_n - eps_m)^2

    over the band window W = [ist0, ist1] (inclusive, 1-based), for any
    two Hermitian operators A, B given as (nstsv, nstsv) matrices in the
    same eigenbasis the energies belong to. With A = v_a and B = v_b this
    is the quantum geometric tensor (kubo_quantum_geometry); with
    A = J^z_a = (1/2){S_z, v_a} and B = v_b it is the spin Berry
    curvature (parsers.spin_hall), which is why this takes operator
    matrices rather than a `pmat` plus two axis indices.

    Pairs with BOTH states inside the window do not appear -- and, for a
    window-summed quantity, dropping them is EXACT rather than a
    convention, for ANY Hermitian A and B: writing X = <n|A|m> and
    Y = <m|B|n> for one such pair, Hermiticity makes the reversed pair's
    numerator <m|A|n><n|B|m> = X* Y* = (XY)*, over the identical
    denominator (eps_n - eps_m)^2, so the two imaginary parts cancel and
    the two real parts are equal-but-already-counted. (For the metric
    that pairing is why the window's tensor is a property of the
    projector onto W alone, blind to how W is internally resolved -- the
    same multi-band, non-Abelian-trace quantity parsers/berry.py's Wilson
    loop and parsers/quantum_geometry.py's overlap stencil compute.) A
    per-band Omega_n, by contrast, keeps its intra-window terms and can
    diverge on a degenerate occupied pair; that is one reason only the
    window-summed quantity is exposed.

    Raises ValueError if any window/outside pair is closer in energy than
    `degeneracy_tol` (Hartree). That is a genuine failure, not noise: the
    window would not be a well-separated group of bands, its projector
    would be discontinuous in k, and the 1/(eps_n - eps_m)^2 weight would
    be dominated by an arbitrarily-split near-degenerate pair. Fails loud
    rather than clipping, matching parsers.quantum_geometry's
    _hermitian_inv_sqrt.
    """
    energies = np.asarray(energies, dtype=float)
    nstsv = len(energies)
    if not (1 <= ist0 <= ist1 <= nstsv):
        raise ValueError(f"invalid band window (ist0={ist0}, ist1={ist1}, nstsv={nstsv})")
    inside = np.zeros(nstsv, dtype=bool)
    inside[ist0 - 1 : ist1] = True
    outside = ~inside
    if not outside.any():
        raise ValueError(
            "band window covers every available state -- the sum over states outside it "
            "is empty, so the Kubo geometric tensor cannot be formed; a window must leave "
            "unoccupied states to sum over (raise nempty)"
        )
    e_in = energies[inside]
    e_out = energies[outside]
    denom = e_in[:, None] - e_out[None, :]
    if np.min(np.abs(denom)) < degeneracy_tol:
        raise ValueError(
            f"band window [{ist0}, {ist1}] is not separated from the states outside it: "
            f"smallest |eps_n - eps_m| across the window boundary is "
            f"{float(np.min(np.abs(denom))):.3e} Ha < degeneracy_tol={degeneracy_tol:.3e}. "
            f"Widen the window to enclose the whole degenerate group (the same rule "
            f"docs/design.md #13 documents for Berry-curvature band windows)"
        )
    va = np.asarray(op_a)[np.ix_(inside, outside)]
    vb = np.asarray(op_b)[np.ix_(outside, inside)]
    # sum_n sum_m va[n,m] * vb[m,n] / denom[n,m]^2
    return complex(np.sum(va * vb.T / denom**2))


def kubo_quantum_geometry(
    energies, pmat, ist0, ist1, directions=(1, 2), degeneracy_tol=1e-4
):
    """The quantum geometric tensor of the band window [ist0, ist1]
    (inclusive, 1-based) in its Kubo / sum-over-states form, at the one
    k-point `energies` and `pmat` were computed at.

    Second-order perturbation theory in k turns the derivative
    |partial_a u> into a sum over the other states,

        <u_m|partial_a u_n> = <m|v_a|n> / (eps_n - eps_m),   m != n,

    which converts the geometric tensor's definition into matrix elements
    of the velocity operator alone -- no k-derivative, no finite
    difference, no dk. Writing
    T_ab = sum_{n in W, m not in W} <n|v_a|m><m|v_b|n>/(eps_n - eps_m)^2,

        g_ab = Re T_ab                (quantum metric, Bohr^2)
        F_ab = -2 Im T_ab             (Berry curvature, Bohr^2)
        Q_ab = g_ab - (i/2) F_ab      (quantum geometric tensor)

    This is the same object Calculation.get_quantum_geometry() builds from
    finite-difference overlaps (docs/design.md #15), reached along a
    completely different route -- which is the point: agreement between
    the two is a real cross-check, since they share no arithmetic and,
    on the Fortran side, not even the same wavefunction machinery
    (genpmatk's gradient/quadrature vs genolpq's real-space overlap).

    Truncation, and why agreement is a few-percent claim rather than an
    exact one:

      * The m-sum runs only over states Elk actually computed, i.e. up to
        nstsv, which `nempty` sets. Elk's default leaves few empty states;
        raise it (Calculation(extra_blocks={"nempty": ...})) and confirm
        the result is stable between two values before trusting it.
      * Core states are not in nstsv at all and are therefore omitted
        entirely. Their contribution is suppressed by 1/(eps_n-eps_m)^2
        but is not zero.

    For the metric's DIAGONAL components each term is |v^a_nm|^2/(delta
    eps)^2 >= 0, so an under-converged nempty shows up there as a clean
    underestimate. That monotonicity does NOT extend to g_ab (a != b) or
    to the curvature, whose terms carry either sign -- treat those as
    "converge and check", not "approach from below".

    Sign convention: A = i<u|grad_k u>, Omega = curl A (Xiao, Chang &
    Niu, RMP 82, 1959 (2010)), pinned against direct numerical
    differentiation in tests/test_parsers_optical.py. This is elkpy's
    single convention everywhere: parsers.berry's Wilson-loop curvature
    agrees, via parsers.berry._berry_phase(), so the two routes are
    directly comparable with no sign bookkeeping. That agreement is not
    free -- this module's arrival is what exposed a missing
    King-Smith--Vanderbilt/Resta negation in parsers.berry, since fixed;
    see docs/design.md #22.

    Returns {"g": (2,2) real array [[g_aa, g_ab], [g_ab, g_bb]],
    "berry_curvature": float (F_ab, same sign convention and units as
    parsers.berry.compute_berry_curvature_path), "Q": (2,2) complex
    Hermitian array} -- the same keys and shapes
    parsers.quantum_geometry.compute_quantum_geometry returns, so the two
    are directly comparable.
    """
    a, b = int(directions[0]), int(directions[1])
    if not (1 <= a <= 3 and 1 <= b <= 3) or a == b:
        raise ValueError(f"directions must be two distinct Cartesian axes in 1..3, got {directions}")
    energies = np.asarray(energies, dtype=float)
    t_aa = kubo_sum(energies, pmat[a - 1], pmat[a - 1], ist0, ist1, degeneracy_tol)
    t_bb = kubo_sum(energies, pmat[b - 1], pmat[b - 1], ist0, ist1, degeneracy_tol)
    t_ab = kubo_sum(energies, pmat[a - 1], pmat[b - 1], ist0, ist1, degeneracy_tol)
    g_aa, g_bb, g_ab = t_aa.real, t_bb.real, t_ab.real
    curvature = -2.0 * t_ab.imag
    g = np.array([[g_aa, g_ab], [g_ab, g_bb]])
    q = np.array(
        [
            [g_aa + 0j, g_ab - 0.5j * curvature],
            [g_ab + 0.5j * curvature, g_bb + 0j],
        ]
    )
    return {"g": g, "berry_curvature": float(curvature), "Q": q}


def kubo_berry_curvature(
    energies, pmat, ist0, ist1, directions=(1, 2), degeneracy_tol=1e-4
):
    """Berry curvature F_ab (Bohr^2) of the band window [ist0, ist1] in
    Kubo form -- kubo_quantum_geometry()'s "berry_curvature" entry. See
    that function for the formula, conventions and truncation caveats."""
    return kubo_quantum_geometry(
        energies, pmat, ist0, ist1, directions, degeneracy_tol
    )["berry_curvature"]


def kubo_quantum_metric(
    energies, pmat, ist0, ist1, directions=(1, 2), degeneracy_tol=1e-4
):
    """Quantum metric g_ab (Bohr^2, a (2,2) real symmetric array) of the
    band window [ist0, ist1] in Kubo form -- kubo_quantum_geometry()'s
    "g" entry. See that function for the formula, conventions and
    truncation caveats."""
    return kubo_quantum_geometry(
        energies, pmat, ist0, ist1, directions, degeneracy_tol
    )["g"]


def band_velocity(pmat, ist):
    """The band velocity v_n = <psi_n|v|psi_n> (atomic units, Cartesian
    x/y/z) of one 1-based band index -- the diagonal of the momentum
    matrix, which for a local Kohn-Sham potential is exactly the band
    slope dE_n/dk by the Hellmann-Feynman theorem (hbar = m_e = 1).

    That identity is what makes this the sharpest cheap check on the
    whole export path: the same number is obtainable from a finite
    difference of eigenvalues alone, which shares no code with the
    matrix-element machinery (see tests/test_calculation_momentum.py).
    Meaningful for a non-degenerate band; within a degenerate group the
    individual diagonal entries depend on how that group happens to be
    resolved.
    """
    nstsv = pmat.shape[-1]
    if not (1 <= ist <= nstsv):
        raise ValueError(f"band index {ist} outside 1..{nstsv}")
    return np.array([pmat[a, ist - 1, ist - 1].real for a in range(3)])
