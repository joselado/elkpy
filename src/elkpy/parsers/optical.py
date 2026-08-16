"""Optical matrix elements and Kubo-form quantum geometry, from the
momentum (velocity) operator exported by the task-9002 MOMENTUM query.

Like parsers/berry.py and parsers/wilson.py this is "arithmetic, not just
parsing": the Fortran side (elkpy_momentum, patches/0007) does nothing but
hand back upstream genpmatk's matrix elements plus the eigenvalues of the
same diagonalisation, and every physical quantity built on them lives here
in Python, unit-testable against synthetic models without an Elk run
(tests/test_parsers_optical.py).

Four families live here:

  * `circular_polarization()` -- the degree of circular polarization of an
    interband transition, i.e. valley-selective circular dichroism.
  * `kubo_berry_curvature()` / `kubo_quantum_metric()` /
    `kubo_quantum_geometry()` -- the quantum geometric tensor in its
    sum-over-states (perturbation-theory) form, an INDEPENDENT code path
    for quantities parsers/berry.py and parsers/quantum_geometry.py
    already compute from finite-difference wavefunction overlaps.
  * `circular_absorption()` -- the polarization-resolved interband
    absorption spectrum, summed over a k-mesh; its polarization-SUMMED
    total is what Elk's own task 121 (src/dielectric.f90, wrapped as
    Calculation.get_dielectric_function()) computes for the linear
    Cartesian components, giving an independent-Fortran cross-check.
  * `effective_mass_tensor()` / `oscillator_strength_sum()` -- the k.p
    sum rule for the inverse effective mass, likewise an INDEPENDENT code
    path for a quantity Calculation.get_effective_mass() (Elk task 25)
    already computes by finite-differencing EIGENVALUES.

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


def _as_energies_pmat(point):
    """Accept either a Momentum namedtuple (EigenstateSession.momentum()),
    the dict Calculation.get_momentum_matrix(kpoints=...) returns, or a
    bare (energies, pmat) pair."""
    if hasattr(point, "energies") and hasattr(point, "pmat"):
        return np.asarray(point.energies, dtype=float), np.asarray(point.pmat)
    if isinstance(point, dict):
        return np.asarray(point["energies"], dtype=float), np.asarray(point["pmat"])
    energies, pmat = point
    return np.asarray(energies, dtype=float), np.asarray(pmat)


def circular_transitions(point, occupations, occmax=2.0, directions=(1, 2)):
    """Every allowed interband transition at ONE k-point, as
    (delta_e, i_plus, i_minus, factor) flat arrays over the (v, c) pairs.

    `delta_e` is the transition energy eps_c - eps_v (Hartree),
    `i_plus`/`i_minus` are (1/2)|P_pm|^2 with
    P_pm = p^a_cv +- i p^b_cv (a, b = `directions`, CARTESIAN axes) --
    the same circular-basis matrix elements circular_polarization() forms,
    with the same convention that the CONDUCTION state carries the bra
    (<c|p|v>, absorption of a photon promoting v to c), and `factor` is
    Elk's own occupation weight f_v (1 - f_c/f_max).

    The 1/2 in the intensities is what makes the two circular channels
    add up to the two linear ones: |P_+|^2 + |P_-|^2 = 2(|p^a|^2 +
    |p^b|^2), so (i_plus + i_minus) = |p^a_cv|^2 + |p^b_cv|^2 exactly.
    That identity is the basis of the cross-check in
    circular_absorption()'s docstring.

    Only UPWARD pairs (delta_e > 0) are kept, where dielectric.f90 sums
    over every ordered pair with |e_ji| > 1e-8. The two agree for a gapped
    system: a pair carrying nonzero f_v (1 - f_c/f_max) there always has
    the empty state above the occupied one. They would differ for a metal,
    which parsers.eigval.occupations_if_uniform() refuses anyway.
    """
    a, b = int(directions[0]), int(directions[1])
    if not (1 <= a <= 3 and 1 <= b <= 3) or a == b:
        raise ValueError(f"directions must be two distinct Cartesian axes in 1..3, got {directions}")
    energies, pmat = _as_energies_pmat(point)
    occupations = np.asarray(occupations, dtype=float)
    if occupations.shape != energies.shape:
        raise ValueError(
            f"occupations shape {occupations.shape} does not match the "
            f"{energies.shape} states at this k-point"
        )
    # Elk's own weight (src/dielectric.f90): occupied initial state, and an
    # empty share of the final one. Formed as an outer product over all
    # (v, c) pairs and then masked, rather than assuming a band count.
    factor = occupations[:, None] * (1.0 - occupations[None, :] / occmax)
    delta_e = energies[None, :] - energies[:, None]
    allowed = (factor > 1e-10) & (delta_e > 1e-8)
    iv, ic = np.nonzero(allowed)
    pa = pmat[a - 1][ic, iv]  # <c|p_a|v>
    pb = pmat[b - 1][ic, iv]
    p_plus = pa + 1j * pb
    p_minus = pa - 1j * pb
    return (
        delta_e[iv, ic],
        0.5 * np.abs(p_plus) ** 2,
        0.5 * np.abs(p_minus) ** 2,
        factor[iv, ic],
    )


def _broadened_delta(x, width, shape):
    if shape == "lorentzian":
        return (width / np.pi) / (x**2 + width**2)
    if shape == "gaussian":
        return np.exp(-((x / width) ** 2) / 2.0) / (width * np.sqrt(2.0 * np.pi))
    raise ValueError(f"broadening must be 'lorentzian' or 'gaussian', got {shape!r}")


def _transition_weights(omega, delta_e, swidth, broadening):
    """The (nw, ntransitions) array each transition's occupation-weighted
    intensity is multiplied by, i.e. everything in Im eps except 1/Omega
    and the matrix elements themselves.

    Two lineshapes, differing only in how the finite broadening is taken:

    "lorentzian"/"gaussian" -- the delta-function form,

        4 pi^2 / w^2 * d(w - Delta),

    the s -> 0 limit of the response, with a normalized broadened delta of
    width `swidth` standing in for the true one.

    "elk" -- src/dielectric.f90's own finite-s response, with no delta
    approximation at all,

        4 pi s / (Delta (w^2 + s^2)) *
          [ (2w - Delta)/((w - Delta)^2 + s^2)
          + (2w + Delta)/((w + Delta)^2 + s^2) ],

    obtained by putting Im eps = 4 pi Re[sigma/(w + i s)] together with
    dielectric.f90's sigma (see Calculation.get_dielectric_function()) for
    a real intensity, keeping BOTH the resonant and the anti-resonant term.

    The two agree exactly at resonance, w = Delta, and their ratio away
    from it is (2w - Delta)/Delta * w^2/(w^2 + s^2) -- a first-order
    O((w - Delta)/Delta) difference, i.e. a few percent across a linewidth
    for s/Delta ~ 0.02, not a small correction. The delta form also carries
    a spurious 1/w^2 blow-up at small w, where a Lorentzian's fat tail is
    multiplied by a diverging prefactor; the "elk" form's w^2 + s^2 has no
    such pole and its two terms cancel exactly at w = 0.
    """
    omega = omega[:, None]
    delta_e = delta_e[None, :]
    if broadening == "elk":
        common = swidth / (delta_e * (omega**2 + swidth**2))
        return 4.0 * np.pi * common * (
            (2.0 * omega - delta_e) / ((omega - delta_e) ** 2 + swidth**2)
            + (2.0 * omega + delta_e) / ((omega + delta_e) ** 2 + swidth**2)
        )
    positive = omega > 0.0
    inv_w2 = np.where(positive, 1.0 / np.where(positive, omega, 1.0) ** 2, 0.0)
    return (
        4.0
        * np.pi**2
        * inv_w2
        * _broadened_delta(omega - delta_e, swidth, broadening)
    )


def circular_absorption(
    kdata, omega, occupations, volume, occmax=2.0, swidth=0.001,
    directions=(1, 2), weights=None, broadening="elk",
):
    """Polarization-resolved interband absorption spectrum: the imaginary
    part of the dielectric function for left- and right-circularly
    polarized light, summed over a k-mesh.

        Im eps_pm(w) = (4 pi^2 / (Omega w^2)) sum_k W_k sum_{v,c}
            f_v (1 - f_c/f_max) (1/2)|P_pm(k)|^2 d(w - (eps_c - eps_v)),

    with P_pm = p^a_cv +- i p^b_cv, Omega the cell volume (Bohr^3), W_k the
    k-point weights and d() a normalized broadened delta function replacing
    the true delta at finite mesh/broadening. This is the
    independent-particle (random-phase, no local-field, no excitonic)
    spectrum: it is the imaginary part of the SAME Kubo-Greenwood response
    src/dielectric.f90 (Elk task 121) evaluates for the LINEAR Cartesian
    components, in the s -> 0 limit of its Lorentzian denominators -- see
    the derivation in Calculation.get_dielectric_function()'s docstring.

    `broadening="elk"` drops the delta approximation and evaluates
    dielectric.f90's finite-s response itself (see _transition_weights),
    which is what makes the comparison below exact rather than approximate:
    at a finite s the delta form differs from it by a factor
    (2w - Delta)/Delta * w^2/(w^2 + s^2), first order in (w - Delta)/Delta
    and therefore a few percent across a linewidth -- and it diverges as
    1/w^2 at small w, where the "elk" form correctly gives zero.

    What is new relative to task 121: Elk builds sigma_xx, sigma_xy and so
    on, never sigma_+ / sigma_-. The circular channels are not recoverable
    from those two alone -- sigma_pm mixes the symmetric and antisymmetric
    parts, and in a non-magnetic crystal the k-RESOLVED dichroism (the
    valley physics) survives even though the zone-integrated one cancels.

    The polarization-summed total is the cross-check, and it is an
    algebraic identity rather than a symmetry assumption:

        Im eps_+ + Im eps_- = Im eps_aa + Im eps_bb

    because |P_+|^2 + |P_-|^2 = 2(|p^a|^2 + |p^b|^2). So a task-121 run
    over the same mesh, the same `nempty` and the same `swidth`, asked for
    the (a, a) and (b, b) components, reproduces this sum -- an entirely
    independent Fortran implementation of the same physics
    (tests/test_calculation_absorption.py).

    Time reversal makes the ZONE-INTEGRATED circular channels equal for a
    non-magnetic crystal: k -> -k exchanges sigma+ and sigma-, and a
    Gamma-centred mesh maps onto itself under that. So eps_+ = eps_- for a
    full-mesh sum is a check on the arithmetic, not a null result -- the
    dichroism is exposed by restricting the sum, which is what `weights`
    is for (pass a mask selecting one valley, and the two channels
    separate; see docs/design.md #24).

    Arguments:
      * `kdata` -- one entry per k-point, each a Momentum namedtuple, a
        dict with "energies"/"pmat" keys (what
        Calculation.get_momentum_matrix(kpoints=...) returns), or a bare
        (energies, pmat) pair. The k-points should be the FULL non-reduced
        mesh: |P_pm|^2 is not invariant under the crystal symmetries that
        fold it (that is exactly the dichroism), so a symmetry-reduced sum
        with the usual weights is wrong here even though it is right for
        the linear components.
      * `omega` -- photon energies (Hartree).
      * `occupations` -- either one (nstsv,) vector shared by every
        k-point (a gapped system, see parsers.eigval.occupations_if_uniform)
        or an (nk, nstsv) array.
      * `volume` -- cell volume in Bohr^3. For a 2D slab this includes the
        vacuum, so the absolute scale of eps - 1 falls like 1/L_z; only
        ratios and comparisons at fixed geometry are meaningful.
      * `occmax` -- Elk's `occmax`: 2 for a spin-degenerate calculation, 1
        when nspinor = 2 (spinpol or spinorb).
      * `swidth` -- broadening width (Hartree), Elk's own `swidth`, whose
        reciprocal is the relaxation time entering the response.
      * `broadening` -- "lorentzian" (default) or "gaussian" for the
        delta-function form, or "elk" for dielectric.f90's own finite-s
        response. Reproducing task 121 point by point needs "elk"; the
        delta forms are the textbook spectrum and agree with it in the
        s -> 0 limit and, at any s, on the integrated oscillator strength.
      * `weights` -- k-point weights, default uniform 1/nk (Elk's
        `wkptnr`). Also the hook for a valley- or region-restricted sum.

    Returns {"omega", "eps2_plus", "eps2_minus", "eps2_total", "eta"},
    with eps2_total = eps2_plus + eps2_minus (= Im eps_aa + Im eps_bb) and
    eta = (eps2_plus - eps2_minus)/eps2_total the frequency-resolved degree
    of circular polarization (NaN where the total vanishes). In the delta
    forms any non-positive `omega` entry is returned as zero rather than as
    the 1/w^2 divergence.
    """
    omega = np.asarray(omega, dtype=float)
    occupations = np.asarray(occupations, dtype=float)
    nk = len(kdata)
    if nk == 0:
        raise ValueError("kdata is empty")
    if occupations.ndim == 1:
        occupations = np.repeat(occupations[None, :], nk, axis=0)
    if len(occupations) != nk:
        raise ValueError(f"occupations has {len(occupations)} rows for {nk} k-points")
    if weights is None:
        weights = np.full(nk, 1.0 / nk)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (nk,):
        raise ValueError(f"weights shape {weights.shape} does not match {nk} k-points")

    delta_e, i_plus, i_minus, factor = [], [], [], []
    for point, occ, weight in zip(kdata, occupations, weights):
        d, ip, im, f = circular_transitions(point, occ, occmax, directions)
        delta_e.append(d)
        i_plus.append(ip)
        i_minus.append(im)
        factor.append(f * weight)
    delta_e = np.concatenate(delta_e)
    i_plus = np.concatenate(i_plus)
    i_minus = np.concatenate(i_minus)
    factor = np.concatenate(factor)

    lineshape = _transition_weights(omega, delta_e, swidth, broadening) / float(volume)
    eps2_plus = lineshape @ (factor * i_plus)
    eps2_minus = lineshape @ (factor * i_minus)
    total = eps2_plus + eps2_minus
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(total > 0.0, (eps2_plus - eps2_minus) / np.where(total > 0.0, total, 1.0), np.nan)
    return {
        "omega": omega,
        "eps2_plus": eps2_plus,
        "eps2_minus": eps2_minus,
        "eps2_total": total,
        "eta": eta,
    }


def _kp_contributions(energies, pmat, ist, degeneracy_tol, nstates=None):
    """Per-intermediate-state terms of the k.p sum rule for band `ist`
    (1-based),

        C^{ab}_m = 2 Re[p^a_nm p^b_mn] / (eps_n - eps_m),    m != n,

    as an (nstates, 3, 3) real array indexed by m. Summing over m and
    adding delta_ab gives the inverse effective mass tensor; see
    effective_mass_tensor() for the physics, the truncation caveats and
    the degeneracy guard this shares.
    """
    energies = np.asarray(energies, dtype=float)
    pmat = np.asarray(pmat)
    nstsv = len(energies)
    if pmat.shape != (3, nstsv, nstsv):
        raise ValueError(
            f"pmat shape {pmat.shape} does not match (3, nstsv, nstsv) with "
            f"nstsv={nstsv} from energies"
        )
    if not (1 <= ist <= nstsv):
        raise ValueError(f"band index {ist} outside 1..{nstsv}")
    if nstates is None:
        nstates = nstsv
    nstates = int(nstates)
    if not (1 <= nstates <= nstsv):
        raise ValueError(f"nstates={nstates} outside 1..{nstsv}")
    if ist > nstates:
        raise ValueError(
            f"band ist={ist} is itself above the truncation nstates={nstates}; "
            f"the state whose mass is asked for must be inside the retained set"
        )
    n = ist - 1
    other = np.array([m for m in range(nstates) if m != n])
    if other.size == 0:
        raise ValueError(
            "only one state available -- the k.p sum over intermediate states is "
            "empty, so the interband contribution to the effective mass cannot be "
            "formed at all (raise nempty)"
        )
    denom = energies[n] - energies[other]
    closest = float(np.min(np.abs(denom)))
    if closest < degeneracy_tol:
        m_closest = int(other[np.argmin(np.abs(denom))]) + 1
        raise ValueError(
            f"band {ist} is degenerate (or near-degenerate) with band {m_closest}: "
            f"|eps_n - eps_m| = {closest:.3e} Ha < degeneracy_tol={degeneracy_tol:.3e}. "
            f"The k.p sum rule as implemented here is a NON-degenerate second-order "
            f"perturbation theory result: with a degenerate partner the single-band "
            f"curvature is not even well defined (the two partners' dispersions cross "
            f"and the 1/(eps_n - eps_m) weight is dominated by an arbitrarily-split "
            f"pair), and Elk's own finite-difference task 25 is equally meaningless "
            f"there. Pick a non-degenerate band, or a k-point where this one is"
        )
    pa = pmat[:, n, other]           # (3, nother): p^a_nm
    pb = pmat[:, other, n]           # (3, nother): p^b_mn
    terms = 2.0 * np.real(pa[:, None, :] * pb[None, :, :]) / denom[None, None, :]
    out = np.zeros((nstates, 3, 3))
    out[other] = np.moveaxis(terms, 2, 0)
    return out


def effective_mass_tensor(
    energies, pmat, ist, degeneracy_tol=1e-4, free_electron_term=True,
    nstates=None, decompose=False,
):
    """The effective mass tensor of band `ist` (1-based) at the one
    k-point `energies` and `pmat` were computed at, from the k.p sum rule
    -- i.e. from momentum matrix elements alone, with no k-derivative and
    no finite difference anywhere.

    In the Bloch Hamiltonian H(k) = e^{-ik.r} H e^{ik.r} = (p + k)^2/2 +
    V(r) (Hartree atomic units, hbar = m_e = 1) the k-dependence is
    explicit and elementary:

        dH/dk_a = p_a + k_a = v_a,      d^2H/dk_a dk_b = delta_ab,

    so ordinary non-degenerate second-order perturbation theory in k gives
    the band curvature exactly:

        (1/m*)^{ab}_n = d^2 eps_n / dk_a dk_b
                      = delta_ab + 2 sum_{m != n}
                            Re[p^a_nm p^b_mn] / (eps_n - eps_m).

    The delta_ab is the bare free-electron term (an electron with no
    interband coupling at all has m* = m_e); every deviation from unit
    mass is interband repulsion, with states BELOW band n (eps_m < eps_n,
    positive denominator) pushing the curvature up and states above
    pushing it down. That decomposition -- which coupling to which band
    produces the mass -- is the physics a finite-difference mass cannot
    give, and is what `decompose=True` returns.

    This is the same tensor Calculation.get_effective_mass() (Elk task 25,
    src/effmass.f90) obtains by fitting a polynomial to EIGENVALUES on a
    small k-mesh around the point and differentiating it. The two share no
    arithmetic and, on the Fortran side, no machinery beyond the
    diagonalisation itself, so agreement is a real cross-check -- the same
    idiom #22's Kubo curvature vs. #13's Wilson loop already uses. Note
    which of task 25's two printed matrices to compare against: its
    "matrix of eigenvalue derivatives" IS this inverse-mass tensor
    (parsers.effmass's "derivative_tensor"), while its "effective mass
    tensor" ("tensor") is that matrix INVERTED, i.e. this function's
    "mass".

    Convergence, and why it is slower here than for #22's Kubo geometry.
    The energy denominator appears to the FIRST power, not squared as in
    kubo_quantum_geometry()'s T_ab. High-lying intermediate states are
    therefore suppressed only as 1/(delta eps), and the sum converges
    correspondingly more slowly in the number of states retained:

      * The m-sum runs only over states Elk computed, i.e. up to nstsv,
        which `nempty` sets. Raise it well above Elk's default
        (Calculation(extra_blocks={"nempty": ...})) and check stability;
        `nstates` here truncates the same sum from Python, which is the
        cheap way to trace out that convergence from a single run.
      * Core states are not among nstsv's valence states at all and are
        omitted entirely. Their denominators are large but their momentum
        matrix elements are not small, and unlike the 1/(delta eps)^2
        geometric sums they are not comfortably negligible here.
      * Even with every computed state retained, the LAPW basis is finite
        (`rgkmax`) and its radial functions are linearized about fixed
        energies, so the high-lying states it does produce are themselves
        a poor representation of the true continuum. The sum rule is exact
        only for a complete set of eigenstates of the same Hamiltonian;
        expect agreement with the finite-difference mass at the tens of
        percent level, improving with nempty, not machine precision.

    Signs help read an under-converged result: omitted states lie ABOVE
    band n (they are the ones truncation removes), so their omitted terms
    have eps_n - eps_m < 0 and each diagonal term -|p^a_nm|^2 x 2/|delta
    eps| is negative. A truncated (1/m*)^{aa} is therefore an
    OVERESTIMATE, and falls monotonically as nempty rises. That
    monotonicity does not extend to the off-diagonal components, whose
    terms carry either sign.

    `free_electron_term=False` drops the delta_ab. Use it for a model
    Hamiltonian written directly in a finite band basis -- a k.p or
    tight-binding H(k), where d^2H/dk_a dk_b is not delta_ab (usually it
    is zero) and the sum over the model's own bands is the whole answer.
    For anything built from Elk's momentum matrix elements the default
    True is the physical choice.

    Raises ValueError if band `ist` is within `degeneracy_tol` (Hartree)
    of any other retained state: the sum rule above is non-degenerate
    perturbation theory, and within a degenerate multiplet a single band's
    curvature is not well defined at all (the partners' dispersions cross,
    and their arbitrary splitting dominates the 1/(eps_n - eps_m) weight).
    Fails loud rather than clipping, matching _kubo_sum's own guard.

    Returns {"inverse_mass": (3,3) real symmetric array (1/m*)^{ab}, in
    units of 1/m_e, "mass": its matrix inverse (units of m_e), "energy":
    eps_n (Hartree), "nstates": how many states the sum retained}, plus,
    with decompose=True, "contributions": an (nstates, 3, 3) array whose
    m-th entry is band m's own contribution, so that
    contributions.sum(axis=0) + delta_ab reproduces "inverse_mass".

    "mass" is None -- not a huge number, and not an exception -- when the
    inverse-mass tensor is singular, i.e. when the band has a direction of
    vanishing curvature and hence a genuinely infinite mass along it (the
    out-of-plane direction of a strictly two-dimensional model
    Hamiltonian, for instance). "inverse_mass" is the primitive here and
    is always well defined; the mass tensor is the derived quantity that
    may not exist.
    """
    energies = np.asarray(energies, dtype=float)
    contributions = _kp_contributions(energies, pmat, ist, degeneracy_tol, nstates)
    inverse_mass = contributions.sum(axis=0)
    if free_electron_term:
        inverse_mass = inverse_mass + np.eye(3)
    singular_values = np.linalg.svd(inverse_mass, compute_uv=False)
    invertible = singular_values[-1] > 1e-10 * max(singular_values[0], 1.0)
    result = {
        "inverse_mass": inverse_mass,
        "mass": np.linalg.inv(inverse_mass) if invertible else None,
        "energy": float(energies[ist - 1]),
        "nstates": int(contributions.shape[0]),
    }
    if decompose:
        result["contributions"] = contributions
    return result


def oscillator_strength_sum(energies, pmat, ist, degeneracy_tol=1e-4, nstates=None):
    """The Thomas-Reiche-Kuhn oscillator-strength sum of band `ist`
    (1-based) at one k-point,

        f^{ab}_n = sum_{m != n} 2 Re[p^a_nm p^b_mn] / (eps_m - eps_n),

    a (3,3) real symmetric array (dimensionless). Each term is the usual
    velocity-form oscillator strength of the n -> m transition, positive
    for an absorption to a higher state.

    What this is and is not a check on. It is the SAME arithmetic as
    effective_mass_tensor(), with the energy denominator reversed in sign:
    identically, at every k-point,

        f^{ab}_n = delta_ab - (1/m*)^{ab}_n.

    So "f = 1 per electron" is NOT a pointwise statement in a crystal, and
    reading it as one would just be asserting that every band is flat.
    What is true is the Brillouin-zone-averaged version: for a filled band
    the average of d^2 eps_n/dk_a dk_b over the zone vanishes (it is the
    second derivative of a periodic function integrated over a period), so

        <f^{ab}_n>_BZ = delta_ab,

    one unit of oscillator strength per electron per Cartesian direction
    -- the f-sum rule behind the optical conductivity's spectral weight,
    sum_int sigma_1(w) dw = pi n / 2 in these units.

    Its practical value here is as a truncation diagnostic with a known
    exact target rather than as an independent physics check: the trace of
    f, averaged over a filled band's k-points, must approach 3, and how
    far short it falls measures directly how much oscillator strength the
    missing core states and the truncated/linearized high-lying LAPW
    states carry. All of effective_mass_tensor()'s caveats apply verbatim.
    """
    contributions = _kp_contributions(energies, pmat, ist, degeneracy_tol, nstates)
    return -contributions.sum(axis=0)
