"""Four-state energy mapping: the full 3x3 exchange tensor J_ij^{ab} of a
magnetic pair, from the total energies of constrained non-collinear DFT
states.

This is the same "arithmetic, not just parsing" exception parsers.berry and
parsers.wilson are: every formula below is pure Python over four total
energies, so it is unit-testable against synthetic data without an Elk run
(tests/test_parsers_exchange.py), while the expensive part -- getting those
four energies out of constrained SCF runs -- lives in elkpy.exchange.

The Hamiltonian convention here is Sabani, Bacaksiz & Milosevic,
PRB 102, 014457 (2020), arXiv:2002.10861, Eqs. 1-2:

    H = sum_{i<j} S_i . J_ij . S_j  +  sum_i S_i . A_ii . S_i

with classical spin vectors of magnitude |S_i| = S, each pair counted ONCE,
and a PLUS sign -- so J < 0 is ferromagnetic. J_ij is a general 3x3 matrix;
in particular J^{ab} != J^{ba} in general, which is what makes a
Dzyaloshinskii-Moriya vector possible.

Two convention traps this module exists to avoid:

* Xiang et al.'s original four-state paper (PRB 84, 224429 (2011),
  arXiv:1106.5549) derives the DM components from a Hamiltonian that has
  already ASSUMED J^{xy} = -J^{yx}, and so carries an extra minus sign that
  is wrong for a general tensor. Sabani et al. document this and show it
  manufactures a spurious DM vector on monolayer CrI3, where an inversion
  centre at the bond midpoint forbids one. Here every one of the nine
  components comes from the SAME formula with no extra sign, and D is
  extracted afterwards from the antisymmetric part.

* The multiplicity m_ij -- the number of bonds this pair actually
  contributes in the supercell, INCLUDING periodic images -- is implicitly 1
  in Xiang/Sabani, and is only 1 if the supercell is big enough. See
  `bond_multiplicity`/`check_supercell` (criterion from arXiv:2512.08471).

Other conventions in this literature differ in sign, in whether the pair sum
double-counts, and in whether S is normalized to 1 (TB2J, arXiv:2009.01910,
differs on all three). Convert explicitly at the boundary; see
docs/design.md #29 and docs/physics.tex Part XVI.
"""

import numpy as np

HARTREE_TO_MEV = 27211.386245988

AXES = ("x", "y", "z")


def _axis_index(a):
    if isinstance(a, str):
        try:
            return AXES.index(a)
        except ValueError:
            raise ValueError(f"axis must be one of {AXES} or 0/1/2, got {a!r}")
    if a not in (0, 1, 2):
        raise ValueError(f"axis must be one of {AXES} or 0/1/2, got {a!r}")
    return a


def axis_index(a):
    """Public spelling of the axis normalizer: "x"/"y"/"z" or 0/1/2 -> 0/1/2."""
    return _axis_index(a)


def spectator_axis(a, b):
    """The Cartesian axis every spin OTHER than the two being flipped points
    along, for the component J^{ab}.

    Sabani et al. require the spectators to be identical in all four states
    and perpendicular to the flipped pair -- that is what makes the
    spectator-spectator terms cancel by being untouched, and the
    pair-spectator cross terms cancel by being linear in one flipped spin.
    For a != b the choice is forced (the remaining axis); for a == b any
    perpendicular axis works and we take the next one cyclically, so the
    choice is deterministic and reproducible.
    """
    a, b = _axis_index(a), _axis_index(b)
    if a != b:
        return 3 - a - b
    return (a + 1) % 3


def four_state_directions(a, b, spin=0.5):
    """The four (S_i, S_j, S_spectator) configurations for component J^{ab},
    in the order (E1, E2, E3, E4) the formula below expects: the pair's signs
    run (+,+), (+,-), (-,+), (-,-).

    Returns a list of 4 dicts with keys "i", "j", "spectator", each a length-3
    vector of magnitude `spin`.
    """
    a, b = _axis_index(a), _axis_index(b)
    c = spectator_axis(a, b)
    ea, eb, ec = (np.eye(3)[k] * spin for k in (a, b, c))
    return [
        {"i": si * ea, "j": sj * eb, "spectator": ec}
        for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1))
    ]


def four_state_component(e1, e2, e3, e4, spin=0.5, multiplicity=1):
    """One tensor component (Sabani Eq. 9/27):

        J^{ab} = (E1 + E4 - E2 - E3) / (4 S^2 m)

    identical for all nine components -- diagonal and off-diagonal alike,
    with no extra sign in any case.

    Everything except the target term cancels: spectator-spectator pairs are
    untouched and appear identically in all four energies; pair-spectator
    cross terms are linear in one flipped spin and so carry sign patterns
    (+,+,-,-) or (+,-,+,-), both annihilated by (+,-,-,+); and single-ion
    anisotropy is quadratic in spin, hence invariant under S -> -S, so it too
    appears identically in all four. The target term alone carries (+,-,-,+)
    and survives with weight 4 S^2.

    Energies in Hartree in, meV out.
    """
    if multiplicity < 1:
        raise ValueError(f"multiplicity must be >= 1, got {multiplicity}")
    return (e1 + e4 - e2 - e3) * HARTREE_TO_MEV / (4.0 * spin**2 * multiplicity)


def assemble_tensor(energies, spin=0.5, multiplicity=1):
    """Build the 3x3 tensor (meV) from a {(a, b): (E1, E2, E3, E4)} mapping,
    energies in Hartree. Missing components are left as NaN, so a partial
    (e.g. symmetric-only) extraction stays visibly partial rather than
    silently reading as zero.
    """
    tensor = np.full((3, 3), np.nan)
    for (a, b), quad in energies.items():
        if len(quad) != 4:
            raise ValueError(f"component ({a}, {b}) needs 4 energies, got {len(quad)}")
        tensor[_axis_index(a), _axis_index(b)] = four_state_component(
            *quad, spin=spin, multiplicity=multiplicity
        )
    return tensor


def decompose(tensor):
    """Split a 3x3 exchange tensor into the three physically distinct pieces:

        J_ij^{ab} = J_iso delta_{ab} + Gamma^{ab} + epsilon_{abc} D^c

    * `isotropic`  J_iso = Tr(J)/3, the Heisenberg part.
    * `dm`         D, from the antisymmetric part: the Dzyaloshinskii-Moriya
                   vector, defined by S_i . A . S_j = D . (S_i x S_j), giving
                   D_x = (J^{yz} - J^{zy})/2 and cyclic. (Identical to the
                   KKR convention, Mankovsky & Ebert arXiv:2206.09969 Eq. 55.)
    * `symmetric`  the traceless symmetric remainder -- the anisotropic
                   exchange proper, which for a honeycomb Kitaev magnet
                   carries K, Gamma and Gamma'.

    A bond whose midpoint is an inversion centre has D = 0 identically
    (Moriya's theorem), which makes `dm` a symmetry-forced null test of the
    whole pipeline rather than merely a reported number.
    """
    tensor = np.asarray(tensor, dtype=float)
    if tensor.shape != (3, 3):
        raise ValueError(f"exchange tensor must be 3x3, got {tensor.shape}")
    isotropic = np.trace(tensor) / 3.0
    antisym = 0.5 * (tensor - tensor.T)
    symmetric = 0.5 * (tensor + tensor.T) - isotropic * np.eye(3)
    dm = np.array([antisym[1, 2], antisym[2, 0], antisym[0, 1]])
    return {"isotropic": isotropic, "dm": dm, "symmetric": symmetric}


def rotate_tensor(tensor, rotation):
    """Express a tensor given in one orthonormal frame in another: with
    `rotation` the matrix whose ROWS are the new frame's axes written in the
    old frame's components, J_new = R J_old R^T.
    """
    rotation = np.asarray(rotation, dtype=float)
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-8):
        raise ValueError("rotation must be orthonormal (rows an orthonormal frame)")
    return rotation @ np.asarray(tensor, dtype=float) @ rotation.T


def honeycomb_cubic_frame():
    """The rotation taking a tensor from crystal Cartesian axes (a, b, c*) to
    the cubic octahedral axes (x, y, z) of a honeycomb Kitaev magnet, for the
    bond whose Kitaev label is z (the bond parallel to b).

    The standard convention has the crystal axes, written in cubic
    components, as
        a  ~ (1, 1, -2)/sqrt(6),   b ~ (-1, 1, 0)/sqrt(2),   c* ~ (1, 1, 1)/sqrt(3)
    so the matrix U with those as ROWS maps cubic components to crystal
    components; its transpose is what we want here (rows = cubic axes in
    crystal components).
    """
    u = np.array([
        [1.0, 1.0, -2.0] ,
        [-1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ])
    u /= np.linalg.norm(u, axis=1)[:, None]
    return u.T


def kitaev_parameters(tensor):
    """J, K, Gamma, Gamma' from a bond tensor ALREADY expressed in the cubic
    octahedral frame of a z-labelled Kitaev bond, where the ideal form is

        M = [[J,     Gamma,  Gamma'],
             [Gamma,  J,     Gamma'],
             [Gamma', Gamma', J + K]]

    (Rau-Lee-Kee convention, as used by Winter et al. arXiv:1603.02548 and
    Hou, Xiang & Gong arXiv:1612.00761). Only bonds whose midpoint has the
    full C2h site symmetry are forced into exactly that shape; on a lower-
    symmetry bond the two Gamma' entries are independent, so both are
    returned separately rather than averaged away, and `residual` reports how
    far the tensor is from the ideal form.
    """
    m = np.asarray(tensor, dtype=float)
    if m.shape != (3, 3):
        raise ValueError(f"bond tensor must be 3x3, got {m.shape}")
    j = 0.5 * (m[0, 0] + m[1, 1])
    k = m[2, 2] - j
    gamma = 0.5 * (m[0, 1] + m[1, 0])
    gamma_p1 = 0.5 * (m[0, 2] + m[2, 0])
    gamma_p2 = 0.5 * (m[1, 2] + m[2, 1])
    ideal = np.array([
        [j, gamma, 0.5 * (gamma_p1 + gamma_p2)],
        [gamma, j, 0.5 * (gamma_p1 + gamma_p2)],
        [0.5 * (gamma_p1 + gamma_p2), 0.5 * (gamma_p1 + gamma_p2), j + k],
    ])
    return {
        "J": j,
        "K": k,
        "Gamma": gamma,
        "Gamma_p1": gamma_p1,
        "Gamma_p2": gamma_p2,
        "residual": float(np.max(np.abs(0.5 * (m + m.T) - ideal))),
    }


# ---------------------------------------------------------------------------
# Single-ion anisotropy
#
# The four-state exchange formula cancels single-ion anisotropy by
# construction (it is even under S -> -S), so the SIA tensor needs its own
# configurations -- and its own formulas, which differ from the exchange one
# in BOTH the sign pattern and the denominator. Sabani et al. Eqs. 15 and 21.
# ---------------------------------------------------------------------------


def single_ion_offdiagonal(e1, e2, e3, e4, spin=0.5):
    """A_ii^{ab} = A_ii^{ab} = (E1 + E4 - E2 - E3) / (4 S^2) for a != b, from
    the four states with S_i at 45 degrees in the ab plane,
    (+-S/sqrt2) a-hat + (+-S/sqrt2) b-hat, all other spins along the third
    axis. Same shape as the exchange formula; the 45-degree placement is what
    makes the cross term A^{ab} S_i^a S_i^b flip sign between the states.

    Energies in Hartree in, meV out.
    """
    return (e1 + e4 - e2 - e3) * HARTREE_TO_MEV / (4.0 * spin**2)


def single_ion_difference(e1, e2, e3, e4, spin=0.5):
    """A_ii^{bb} - A_ii^{aa} = (E1 + E2 - E3 - E4) / (2 S^2), from S_i along
    +-b (states 1, 2) and +-a (states 3, 4), all other spins along the
    complementary axis.

    Note the different sign pattern and denominator from the exchange
    formula. Only DIFFERENCES of the diagonal are extractable at all: with a
    classical spin of fixed length, (S^x)^2 + (S^y)^2 + (S^z)^2 = S^2 makes
    any common part of the diagonal an additive constant, invisible to every
    energy difference. Pairing the two signs of S_i in each half is what
    removes the exchange terms, which are linear in S_i.

    Energies in Hartree in, meV out.
    """
    return (e1 + e2 - e3 - e4) * HARTREE_TO_MEV / (2.0 * spin**2)


# ---------------------------------------------------------------------------
# Supercell bookkeeping
#
# Flipping site j in a periodic cell flips ALL of its periodic images, so the
# energy difference measures sum_T J(i, j + T), not J(i, j). Xiang and Sabani
# both implicitly assume that sum has exactly one term. The criterion below
# is from arXiv:2512.08471: for every pair within the n-th neighbour
# distance, no periodic image may sit closer than that distance.
# ---------------------------------------------------------------------------


def image_distances(avec, r_i, r_j, nmax=3):
    """Sorted distances |r_j + T - r_i| over supercell translations T, with
    `avec` the (3, 3) lattice vectors as rows and positions Cartesian, in
    whatever length unit is used consistently. T = 0 is excluded when the two
    positions coincide (an atom does not bond to itself).
    """
    avec = np.asarray(avec, dtype=float)
    delta = np.asarray(r_j, dtype=float) - np.asarray(r_i, dtype=float)
    rng = range(-nmax, nmax + 1)
    shifts = np.array([(n1, n2, n3) for n1 in rng for n2 in rng for n3 in rng])
    vectors = delta + shifts @ avec
    distances = np.linalg.norm(vectors, axis=1)
    return np.sort(distances[distances > 1e-6])


def bond_multiplicity(avec, r_i, r_j, tol=1e-3, nmax=3):
    """m_ij: how many periodic images of j sit at the SAME distance as the
    nearest one -- i.e. how many copies of this bond a single flip of j
    actually turns over, and therefore what the four-state energy difference
    must be divided by.
    """
    distances = image_distances(avec, r_i, r_j, nmax=nmax)
    return int(np.count_nonzero(np.abs(distances - distances[0]) < tol))


def check_supercell(avec, r_i, r_j, shell_cutoff, tol=1e-3, nmax=3):
    """Raise unless the (i, j) bond can be extracted cleanly: every image of j
    within `shell_cutoff` of i must sit at the target distance itself.

    An image at a DIFFERENT distance inside the retained interaction range
    means the measured energy difference mixes two physically distinct bonds,
    and no choice of multiplicity can unmix them -- the supercell is simply
    too small. Returns (multiplicity, target_distance).
    """
    distances = image_distances(avec, r_i, r_j, nmax=nmax)
    target = distances[0]
    if shell_cutoff < target - tol:
        raise ValueError(
            f"shell_cutoff {shell_cutoff:.4f} is shorter than the bond itself "
            f"({target:.4f}) -- nothing to extract"
        )
    contaminating = distances[
        (distances < shell_cutoff + tol) & (np.abs(distances - target) >= tol)
    ]
    if contaminating.size:
        raise ValueError(
            f"supercell too small for this pair: bond length {target:.4f}, but "
            f"periodic image(s) of the same atom at {np.unique(np.round(contaminating, 4))} "
            f"also fall within the retained interaction range ({shell_cutoff:.4f}). "
            f"The four-state energy difference would mix distinct neighbour shells; "
            f"enlarge the supercell (see docs/design.md #29)."
        )
    return bond_multiplicity(avec, r_i, r_j, tol=tol, nmax=nmax), float(target)
