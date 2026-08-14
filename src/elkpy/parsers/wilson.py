"""Non-Abelian Wilson loop (hybrid Wannier charge centers) and the Z2
topological invariant via Wannier-center pumping.

Yu, Qi, Bernevig, Fang, Dai, "Equivalent expression of Z2 topological
invariant for band insulators using the non-Abelian Berry connection",
Phys. Rev. B 84, 075119 (2011) (arXiv:1101.2011) -- the multi-band Wilson
loop D(k_pump) (their eq. 5) built from a closed loop of nearest-neighbour
overlap matrices, whose eigenvalue phases are the hybrid Wannier charge
centers (WCCs, eq. 6), and the reference-line crossing-counting argument
for Z2 (their Sec. II).

Soluyanov & Vanderbilt, "Computing topological invariants without
inversion symmetry", Phys. Rev. B 83, 235401 (2011) (arXiv:1102.5600) --
the "largest gap" tracking method (their eqs. 16-18) used here instead of
an arbitrary fixed reference line: robust to WCC branch-labelling/
band-crossing ambiguity, since it only ever asks whether a WCC crossed the
*largest empty arc* between two pumping steps, not which individual band
index it belongs to.

See docs/design.md #20 and docs/physics.tex Part IX for the physics
writeup (why only half the Brillouin zone needs to be pumped, the Kramers
pairing at the two ends, the additivity of Z2 across independently-gapped
band groups) this implements. `Calculation.get_z2_invariant()` drives this
purely from task 9000's existing mesh-neighbour overlap export
(parsers/berry.py's parse_berry_overlaps -- already relied on, via its
Chern-number arithmetic, to handle the Brillouin-zone-boundary periodic
gauge closure correctly) -- no new Fortran, and all Wilson-loop/WCC/Z2
arithmetic happens here in pure numpy, independently unit-testable against
synthetic overlap matrices (tests/test_z2_gauge_invariance.py).
"""

import numpy as np


def _unitarize(m, floor=1e-8):
    """SVD (polar-decomposition) unitarization of a possibly non-unitary
    overlap matrix: M = U S V^dagger -> U V^dagger, the multi-band
    generalization of parsers.berry._link_variable's det(M)/|det(M)| (which
    is exactly this construction's 1x1 special case: for a scalar, U and V
    are unit-modulus phases and S=|M|, so U V^dagger = M/|M| = det(M)/|det(M)|).

    Raises ValueError if any singular value is at or below `floor`, the
    same fail-loud choice parsers.berry._link_variable and
    parsers.quantum_geometry._hermitian_inv_sqrt make for a near-singular
    overlap -- a small singular value means the requested band window is
    not well resolved at this link, and unitarizing anyway would silently
    substitute a very different (spuriously "closest unitary") matrix for
    what should be a small, well-defined correction.
    """
    u, s, vh = np.linalg.svd(m)
    if np.any(s <= floor):
        raise ValueError(
            f"near-singular link overlap (smallest singular value {s.min():.2e} <= "
            f"{floor:.2e}) -- the requested band window is not well resolved at this "
            f"k-point pair (e.g. too close to a degeneracy/band crossing), or the mesh "
            f"is too coarse; increase nkx/nt or check the band window"
        )
    return u @ vh


def wilson_loop_wannier_centers(link_overlaps):
    """Hybrid Wannier charge centers from one closed Wilson loop.

    `link_overlaps`: ordered list [F_0, F_1, ..., F_{N-1}] of (nst, nst)
    complex matrices, F_i = <psi(k_i)|psi(k_{i+1})>, walking a CLOSED loop
    (k_N == k_0 modulo a reciprocal lattice vector -- F_{N-1} is the link
    that closes the loop back to the start).

    arXiv:1101.2011 eq. 5: D = U(F_0) U(F_1) ... U(F_{N-1}), each link
    unitarized first (_unitarize -- the standard multi-band Wilson-loop
    practice; this project's parsers.berry.compute_berry_curvature already
    uses this construction's single-band/Abelian det-phase special case).

    Returns sorted Wannier-center angles theta_m in (-pi, pi] (eq. 6:
    theta_m = Im(log(lambda_m)), the phases of D's eigenvalues lambda_m --
    |lambda_m| = 1 exactly since D is a product of unitary matrices).
    x_m = theta_m / (2*pi) mod 1 gives the actual hybrid Wannier center
    position, in units of the loop direction's lattice vector spacing.
    """
    d = None
    for f in link_overlaps:
        u = _unitarize(f)
        d = u if d is None else d @ u
    eigvals = np.linalg.eigvals(d)
    return np.sort(np.angle(eigvals))


def _largest_gap_center(thetas):
    """The angle (in (-pi, pi]) at the center of the largest empty arc
    between adjacent Wannier-center angles on the unit circle --
    Soluyanov & Vanderbilt's "largest gap" reference curve (arXiv:1102.5600
    Sec. III.B, the z^(m) used in eq. 16-18): robust to how individual WCC
    branches are labelled/sorted, unlike tracking a fixed reference line or
    individual band trajectories.
    """
    thetas = np.sort(thetas)
    extended = np.concatenate([thetas, [thetas[0] + 2 * np.pi]])
    gaps = np.diff(extended)
    i = int(np.argmax(gaps))
    center = thetas[i] + gaps[i] / 2
    return ((center + np.pi) % (2 * np.pi)) - np.pi


def _orientation(a, b, c):
    """Signed circular-orientation test: whether angle c lies on the
    counter-clockwise arc from a to b.

    arXiv:1102.5600 eq. 17: g(phi1,phi2,phi3) = sin(phi2-phi1) +
    sin(phi3-phi2) + sin(phi1-phi3) -- factors as
    -4*sin((a-b)/2)*sin((b-c)/2)*sin((c-a)/2), vanishing iff any two of
    a,b,c coincide (mod 2*pi) and changing sign according to the cyclic
    order of the three angles on the unit circle.
    """
    return np.sin(b - a) + np.sin(c - b) + np.sin(a - c)


def z2_from_wannier_centers(theta_by_step):
    """Z2 invariant (0 trivial, 1 topological) from a sequence of WCC
    spectra sampled across the pumping direction's time-reversal-invariant
    half Brillouin zone (arXiv:1101.2011 Sec. II; arXiv:1102.5600
    eqs. 16-18 for the largest-gap crossing count used here).

    `theta_by_step`: list of arrays (one per pumping step m=0..M, in
    order), each the sorted WCC-angle array (radians) returned by
    wilson_loop_wannier_centers() at that pumping-direction k-value. The
    first and last steps should be the two time-reversal-invariant momenta
    (k_pump=0 and k_pump=0.5) -- the WCC spectrum there is Kramers-paired,
    though this function does not itself check that (see
    Calculation.get_z2_invariant()'s docstring for the mesh construction
    that guarantees it).

    arXiv:1102.5600 eq. 16: Delta = sum_m Delta_m mod 2, where eq. 18 gives
    each step's contribution from the sign of the orientation test between
    the largest-gap centers z^(m), z^(m+1) and every WCC x_n^(m+1) at the
    LATER step of the pair:
        (-1)^Delta_m = prod_n sgn[g(2*pi*z^(m), 2*pi*z^(m+1), 2*pi*x_n^(m+1))]
    An odd total (Delta mod 2 == 1) is the topological (QSH) case.
    """
    gap_centers = [_largest_gap_center(thetas) for thetas in theta_by_step]
    parity = 1
    for m in range(len(theta_by_step) - 1):
        z_m, z_m1 = gap_centers[m], gap_centers[m + 1]
        for theta in theta_by_step[m + 1]:
            s = np.sign(_orientation(z_m, z_m1, theta))
            if s < 0:
                parity = -parity
            # s == 0 (a WCC landing exactly on a gap center, or z_m == z_m1)
            # is an unbroken-tie edge case eq. 18's own text notes is
            # harmless to assign either way, since it appears identically
            # in every step's product -- left as a no-op (+1) here.
    return 0 if parity > 0 else 1


def combine_3d_invariants(z_by_axis_offset):
    """The 3D strong/weak Z2 classification (nu0; nu1, nu2, nu3), Fu, Kane
    & Mele, "Topological Insulators in Three Dimensions", Phys. Rev. Lett.
    98, 106803 (2007) (arXiv:cond-mat/0607699), from the six per-plane 2D
    Z2 invariants of a 3D time-reversal-invariant insulator's six
    time-reversal-invariant (TRI) planes (k_i=0 and k_i=pi for each
    reciprocal-lattice direction i=1,2,3 -- each such plane is itself a
    genuine 2D time-reversal-invariant system, since the other two
    directions on the plane still map k -> -k onto the same plane; see
    docs/design.md #21).

    `z_by_axis_offset`: dict {(axis, offset): z2}, axis in (1, 2, 3),
    offset in (0.0, 0.5), z2 in (0, 1) -- the six 2D Z2 invariants (e.g.
    Calculation.get_z2_invariant_3d()'s six get_z2_invariant() calls, one
    per (axis, offset) pair).

    FKM eq. 2 (strong index): the product of the parity eigenvalues at all
    8 TRIM factors exactly into the product over the 4 TRIM of the k_i=0
    plane times the product over the 4 TRIM of the k_i=pi plane, for ANY
    choice of axis i -- so in mod-2/XOR form, nu0 = z(k_i=0) XOR z(k_i=pi)
    for i=1, 2, OR 3, and these three must all agree (FKM state this
    explicitly: "nu0 is independent of the choice of b_k"). A disagreement
    here is therefore a bug (mesh/plumbing), not a physical ambiguity, and
    raises ValueError rather than silently picking one axis's answer.

    FKM eq. 3 (weak indices): nu_i = z(k_i=pi), the k_i=pi plane's OWN 2D
    Z2 invariant (not the k_i=0 plane -- the two conventions are not
    interchangeable, this is the one FKM's eq. 3 defines). Unlike nu0,
    (nu1,nu2,nu3) are basis-dependent -- FKM note they combine into a
    reciprocal-lattice vector G_nu = sum_i nu_i b_i, so their individual
    values depend on the primitive reciprocal vectors b_i used (here,
    whatever Calculation.structure.avec's own reciprocal vectors are) --
    report them as computed, not as an assumed universal (0,0,0)/(1,1,1).

    Returns a dict: {"nu0": 0 or 1, "nu": (nu1, nu2, nu3), "nu0_by_axis":
    (nu0 computed via axis 1, via axis 2, via axis 3) -- all three equal to
    "nu0" if this function returned at all}.
    """
    nu0_by_axis = tuple(
        z_by_axis_offset[(axis, 0.0)] ^ z_by_axis_offset[(axis, 0.5)] for axis in (1, 2, 3)
    )
    if len(set(nu0_by_axis)) != 1:
        raise ValueError(
            f"strong index nu0 disagrees across axes: {nu0_by_axis} -- FKM eq. 2 guarantees "
            f"these must agree algebraically, so this is a bug (mesh construction, band "
            f"window, or a genuinely unresolved gap on one axis' mesh -- see "
            f"Calculation.get_z2_invariant()'s resolvability warning), not a physical result"
        )
    nu = tuple(z_by_axis_offset[(axis, 0.5)] for axis in (1, 2, 3))
    return {"nu0": nu0_by_axis[0], "nu": nu, "nu0_by_axis": nu0_by_axis}
