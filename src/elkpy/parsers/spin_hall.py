"""Spin Berry curvature and the intrinsic spin Hall conductivity, from the
spin current operator J^s_a = (1/2){S_s, v_a}.

This is the join between two things elkpy already had separately and could
not legally multiply together:

  * the velocity matrix elements v_a = p_a of the MOMENTUM query
    (parsers.optical, docs/design.md #22), and
  * the spin operators S_x, S_y, S_z built from evecsv
    (parsers.spin, docs/design.md #17).

Before patches/0009 those came from two SEPARATE diagonalisations -- the
MOMENTUM query returned energies and pmat but not evecsv, so S_s had to be
fetched by a second EIGENSTATES query. Two independent diagonalisations at
the same k are free to resolve a degenerate multiplet in different, equally
valid bases (docs/design.md #14), which makes the product S_s v_a
meaningless while leaving every Hermiticity and unitarity check on both
factors perfectly intact. Patch 0009 makes MOMENTUM return evecsv from its
own diagonalisation, which is the entire content of that patch and the
reason this module can exist. See docs/design.md #24.

Like parsers/optical.py, parsers/berry.py and parsers/wilson.py this is
"arithmetic, not just parsing": no new Fortran, and every function here is
unit-testable on synthetic (energies, pmat, S) triples with no Elk run
(tests/test_parsers_spin_hall.py).

Physics
-------
The conventional spin current operator (Guo, Yao & Niu, PRL 94, 226601
(2005), arXiv:cond-mat/0505146 -- the ancestor Kubo formula; Yao & Fang,
PRL 95, 156601 (2005), arXiv:cond-mat/0502351; Guo, Murakami, Chen &
Nagaosa, PRL 100, 096401 (2008), arXiv:0705.0409) is the symmetrized
product

    J^s_a = (1/2) {S_s, v_a} = (1/2) (S_s v_a + v_a S_s),

Hermitian by construction even though S_s and v_a need not commute. In
atomic units this IS the papers' own operator: they write it as
j^z_x = (hbar/4)(sigma_z v_x + v_x sigma_z), i.e. (1/2){s_z, v_x} with
s_z of eigenvalues +-hbar/2, which is elkpy's S_z at hbar = 1.

Its Kubo response to an electric field along b gives the spin Berry
curvature of an occupied band window W,

    Omega^s_ab = -2 Im sum_{n in W} sum_{m not in W}
                        <n|J^s_a|m><m|v_b|n> / (eps_n - eps_m)^2,

i.e. exactly parsers.optical's Kubo geometric tensor with the first velocity
factor replaced by the spin current. The intrinsic spin Hall conductivity is
its Brillouin-zone integral over occupied states,

    sigma^s_ab = int [dk] Omega^s_ab(k).

ON THE OVERALL SIGN, which the literature does not settle: Yao & Fang print
this expression with -2 Im, Guo/Murakami/Chen/Nagaosa with +2 Im, everything
else (operator, matrix-element ordering <n|j_x|m><m|v_y|n>, squared
denominator) identical -- and both nevertheless report a POSITIVE SHC of
comparable size for hole-doped GaAs, so the printed difference is absorbed
somewhere in the charge vertex (-e v), not a physical disagreement. elkpy
therefore fixes the sign by INTERNAL consistency rather than by picking a
paper: -2 Im is what makes S -> 1 reproduce elkpy's own charge Berry
curvature, which is pinned to A = i<u|grad_k u>, Omega = curl A (Xiao,
Chang & Niu, RMP 82, 1959 (2010)) -- see docs/design.md #22, where that
convention was fixed once for the whole project. It coincides with
Yao & Fang's printed form. The physical statement of the sign, in words
(Yao & Fang): positive sigma^{s_z}_xy means the s_z = +1/2 component flows
along +x.

CAVEAT (a real one, not a formality): (1/2){S_s, v} is the CONVENTIONAL
spin current, which is not conserved once spin-orbit coupling breaks spin
conservation. Because [S_z, H] != 0 the continuity equation acquires a
torque density, dS_z/dt + div J_s = T_z, so J_s has no conjugate
thermodynamic force and obeys no Onsager relation; Shi, Zhang, Xiao & Niu,
PRL 96, 076604 (2006) (arXiv:cond-mat/0503505) argue for a "proper" current
d(r S_z)/dt = J_s + P_tau, differing by a torque-dipole term this module
does not compute. The conventional definition is nonetheless what
essentially all first-principles SHC numbers in the literature use, so it
is what is implemented here, named for what it is. Review covering both the
Kubo conventions and this caveat: Sinova, Valenzuela, Wunderlich, Back &
Jungwirth, RMP 87, 1213 (2015), arXiv:1411.3249.

Units and conventions
---------------------
Everything is Hartree atomic units, inherited from parsers.optical: pmat in
atomic units, energies in Hartree, so Omega^s comes out in Bohr^2 times
whatever units S carries.

S here is elkpy's own spin operator (parsers.spin), whose eigenvalues are
+-1/2, i.e. hbar = 1 with S = sigma/2 -- which matches the source papers'
own s_z = (hbar/2) sigma_z exactly at hbar = 1, so J^s here is literally
their j^z. The consequence worth stating, because it is exactly a factor of
2: when S_z is conserved and the bands split into two decoupled sectors,
s_z acts as the scalar +-1/2 within each, so

    Omega^s = sum_sigma s_sigma Omega^sigma = (1/2)(Omega^up - Omega^down),

not the bare difference. (Kane & Mele, PRL 95, 226801 (2005) state the
current-level version, J_s = (hbar/2e)(J_up - J_down); PRL 95, 146802
(2005) the invariant-level one, that each spin sector carries its own
Chern number whose DIFFERENCE is the quantized spin Hall conductivity --
the same reduction elkpy's own parsers/wilson.py cross-check already uses
mod 2, docs/design.md #20.)

`spin_berry_curvature()` returns that (1/2)-scaled quantity, i.e. spin
conductivity in units where hbar = 1. Papers quote SHC either in
(hbar/e)(Ohm cm)^-1 or converted to charge-conductivity units by
"multiplying a factor 2|e|/hbar" (Yao & Fang's own words) -- so multiply
by 2 for that comparison.

`directions` indexes CARTESIAN axes (1, 2, 3 = x, y, z), the same as
parsers.optical and for the same reason (genpmatk's components), NOT the
reciprocal-lattice convention of Calculation.get_berry_curvature().
"""

import numpy as np

from .optical import kubo_sum


def spin_current_operator(pmat, s, direction=1):
    """The conventional spin current operator

        J^s_a = (1/2) {S_s, v_a} = (1/2) (S_s v_a + v_a S_s)

    as an (nstsv, nstsv) complex Hermitian matrix, for Cartesian axis
    `direction` (1-based: 1, 2, 3 = x, y, z).

    `pmat`: (3, nstsv, nstsv) momentum/velocity matrix from a MOMENTUM
    query. `s`: (nstsv, nstsv) spin operator for the chosen spin
    projection (parsers.spin.compute_spin_operator over the FULL band
    range), in the SAME eigenbasis -- which in practice means built from
    the evecsv of the same MOMENTUM response (see this module's
    docstring; EigenstateSession.spin_current_operator() does this
    pairing for you).

    The symmetrization is not cosmetic: S_s v_a alone is not Hermitian
    when [S_s, v_a] != 0, which is precisely the spin-orbit-coupled case
    of interest, and a non-Hermitian J would break the pairwise
    cancellation of intra-window terms that makes the window-summed
    curvature below well defined (see parsers.optical.kubo_sum).
    """
    a = int(direction)
    if not 1 <= a <= 3:
        raise ValueError(f"direction must be a Cartesian axis in 1..3, got {direction}")
    v = np.asarray(pmat[a - 1])
    s = np.asarray(s)
    if s.shape != v.shape:
        raise ValueError(
            f"spin operator shape {s.shape} does not match the momentum matrix's "
            f"{v.shape} -- the spin operator must span the FULL nstsv range, in the "
            f"same eigenbasis as pmat (build it from the MOMENTUM response's own "
            f"evecsv, not a separate EIGENSTATES query -- see docs/design.md #24)"
        )
    return 0.5 * (s @ v + v @ s)


def spin_berry_curvature(
    energies, pmat, s, ist0, ist1, directions=(1, 2), degeneracy_tol=1e-4
):
    """The spin Berry curvature Omega^s_ab (Bohr^2) of the band window
    [ist0, ist1] (inclusive, 1-based) at the one k-point `energies`,
    `pmat` and `s` were computed at:

        Omega^s_ab = -2 Im sum_{n in W} sum_{m not in W}
                            <n|J^s_a|m><m|v_b|n> / (eps_n - eps_m)^2,

    with J^s_a = (1/2){S_s, v_a} the conventional spin current operator
    (Guo, Yao & Niu, PRL 94, 226601 (2005)). `directions` is
    (a, b), 1-based Cartesian; the default (1, 2) is the usual
    "spin current along x driven by a field along y", i.e. sigma^s_xy.

    Which spin projection this is comes entirely from `s`: pass sz for
    the standard sigma^{s_z}_xy, sx/sy for the others.

    Sign convention: identical to parsers.optical's ordinary Kubo Berry
    curvature (A = i<u|grad_k u>, Omega = curl A -- Xiao, Chang & Niu,
    RMP 82, 1959 (2010)), since this IS that expression with v_a -> J^s_a
    and shares its arithmetic (parsers.optical.kubo_sum). Setting s = the
    identity recovers the ordinary charge Berry curvature exactly, which
    is the cheapest available pin on that shared sign -- and is how the
    sign is fixed here at all, since the source papers print it both ways
    (see this module's docstring).

    Scale: S has eigenvalues +-1/2 here, so for a system with conserved
    S_z this equals sum_sigma s_sigma Omega^sigma = (Omega^up -
    Omega^down)/2, half of what a paper working in hbar/2 units would
    quote. See this module's docstring.

    Truncation and the degeneracy guard are inherited unchanged from
    parsers.optical.kubo_sum: the m-sum runs only to nstsv (raise
    `nempty` and check stability), and a window not separated from the
    states outside it raises ValueError rather than being clipped. Note
    the truncation is LESS benign here than for the ordinary metric --
    no term has a definite sign, so an under-converged nempty is not a
    one-sided error.

    Only the window-SUMMED curvature is exposed, deliberately: pairs with
    both states inside the window cancel exactly in that sum for any
    Hermitian J and v (parsers.optical.kubo_sum's docstring derives it),
    so the result never touches an intra-window energy denominator, while
    a per-band Omega^s_n would diverge on any degenerate occupied pair.
    """
    a, b = int(directions[0]), int(directions[1])
    if not (1 <= a <= 3 and 1 <= b <= 3) or a == b:
        raise ValueError(
            f"directions must be two distinct Cartesian axes in 1..3, got {directions}"
        )
    j_a = spin_current_operator(pmat, s, direction=a)
    t = kubo_sum(energies, j_a, np.asarray(pmat[b - 1]), ist0, ist1, degeneracy_tol)
    return -2.0 * t.imag


def spin_hall_conductivity(curvatures, cell_volume, weights=None):
    """The intrinsic spin Hall conductivity

        sigma^s_ab = int_BZ [d^dk/(2 pi)^d] Omega^s_ab(k)

    in Hartree atomic units, from spin Berry curvatures sampled on a
    Brillouin-zone mesh.

    `curvatures`: the per-k-point Omega^s_ab values (Bohr^2) from
    spin_berry_curvature(), each already summed over the occupied window.
    `cell_volume`: the real-space unit cell volume in Bohr^d (d = 3 for a
    bulk crystal; for a slab, using the supercell volume gives the
    conductivity of the slab-plus-vacuum stack, so multiply by the
    supercell height to get the 2D sheet conductance instead).
    `weights`: optional per-point weights, defaulting to a uniform mesh.
    They are normalized internally, so only their ratios matter.

    The mesh sum uses BZ volume = (2 pi)^d / V_cell, which turns
    int [dk]/(2 pi)^d into (1/(N_k V_cell)) sum_k for a uniform mesh --
    so this is simply the mean curvature divided by the cell volume.
    Units are then Bohr^(2-d); for d = 3 that is 1/Bohr, the atomic unit
    of (spin) conductivity, e^2/(hbar a_0) with S dimensionless. Multiply
    by 2 for the hbar/2e convention most SHC papers quote (see this
    module's docstring), and note that a converged number needs a genuinely
    dense mesh: the spin Berry curvature is sharply peaked near avoided
    crossings, so this integral converges far more slowly than a total
    energy on the same mesh.

    This helper does no k-point generation or symmetry unfolding of its
    own -- feed it the curvature you sampled, on whatever mesh covers the
    full zone (a symmetry-reduced mesh needs its own multiplicity weights,
    which is what `weights` is for).
    """
    c = np.asarray(curvatures, dtype=float).ravel()
    if c.size == 0:
        raise ValueError("no curvature samples given")
    if weights is None:
        w = np.full(c.size, 1.0 / c.size)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape != c.shape:
            raise ValueError(f"weights shape {w.shape} does not match curvatures {c.shape}")
        total = w.sum()
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        w = w / total
    if cell_volume <= 0:
        raise ValueError(f"cell_volume must be positive, got {cell_volume}")
    return float(np.sum(w * c) / cell_volume)
