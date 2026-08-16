"""Integration tests for Calculation.get_parity() / get_fu_kane_invariant()
(the inversion operator at the TRIM and the Fu-Kane symmetry-indicator Z2,
task 9002's PARITY query, patches/0008 -- see docs/design.md #23), against
a real compiled elk binary.

The point of this feature is that it reaches the same invariant as
get_z2_invariant()/get_z2_invariant_3d() (docs/design.md #20/#21) from a
handful of k-points instead of a Wannier-charge-center mesh sweep. So the
tests here are deliberately run on the two systems those methods were
already validated on, and must agree with them:

  * monolayer graphene with soc_scale=3000 -- Kane & Mele's quantum spin
    Hall prediction, Z2 = 1 (PRL 95, 146802 (2005))
  * the [111]-dimerized diamond ("cesium") model -- Fu, Kane & Mele's own
    introductory example for (nu0; nu1nu2nu3), nu0 = 1 (PRB 76, 045302
    (2007))

Agreement between a symmetry-indicator count at 4-8 k-points and a
gauge-based mesh pumping calculation is a strong cross-check: the two share
no arithmetic at all, and the parity route additionally exercises a Fortran
path (the symmetry transformation of the first-variational coefficients)
that nothing else in elkpy touches.

Skipped if the elk binary hasn't been built, same as test_calculation_si.py.
"""

import re

import pytest

from elkpy import config
from elkpy.parsers import symmetry
from elkpy.structure import Structure

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)

_BOHR_PER_ANGSTROM = 1.0 / 0.529177210903

# monolayer graphene, same fixture as tests/test_calculation_z2.py
A = 2.46 * _BOHR_PER_ANGSTROM
VACUUM = 20.0
GRAPHENE_AVEC = [(A, 0.0, 0.0), (-A / 2, A * 3**0.5 / 2, 0.0), (0.0, 0.0, VACUUM)]
GRAPHENE_SPECIES = {"C": [(0.0, 0.0, 0.5), (1 / 3, 2 / 3, 0.5)]}

# monolayer h-BN -- two DIFFERENT species on the two sublattices, so the
# honeycomb's inversion centre is destroyed; used here as the no-inversion
# error case
HBN_A = 4.743210000
HBN_AVEC = [(HBN_A, 0.0, 0.0), (-HBN_A / 2, HBN_A * 3**0.5 / 2, 0.0), (0.0, 0.0, 20.0)]
HBN_SPECIES = {"B": [(1 / 3, 2 / 3, 0.5)], "N": [(2 / 3, 1 / 3, 0.5)]}

# freestanding buckled-honeycomb monolayer Bi ("bismuthene"), same fixture as
# tests/test_calculation_z2.py -- a QSH insulator (Murakami, PRL 97, 236805
# (2006)) where get_z2_invariant() measures Z2 = 1, used here as a SECOND
# independent 2D cross-check of the parity route (graphene is the first), on a
# real material with its own atomic spin-orbit coupling rather than an
# artificially scaled one
BI_A = 4.34 * _BOHR_PER_ANGSTROM
BI_BUCKLING = 1.73 * _BOHR_PER_ANGSTROM
BI_VACUUM = 28.0
BI_AVEC = [(BI_A, 0.0, 0.0), (-BI_A / 2, BI_A * 3**0.5 / 2, 0.0), (0.0, 0.0, BI_VACUUM)]
_bi_half_delta = (BI_BUCKLING / 2) / BI_VACUUM
BI_SPECIES = {"Bi": [(0.0, 0.0, 0.5 + _bi_half_delta), (1 / 3, 2 / 3, 0.5 - _bi_half_delta)]}

# [111]-dimerized diamond, same fixture as tests/test_calculation_z2_3d.py
CS_A = 16.0
DIAMOND_AVEC = [
    (0.0, CS_A / 2, CS_A / 2),
    (CS_A / 2, 0.0, CS_A / 2),
    (CS_A / 2, CS_A / 2, 0.0),
]
DELTA = 0.03
CS_SPECIES = {"Cs": [(0.0, 0.0, 0.0), (0.25 - DELTA,) * 3]}


def _occupied_count(calc):
    """Occupied-band count from EIGVAL.OUT's own occupation numbers, not an
    assumed electron count -- the same pitfall/fix as every other spinorb
    fixture in this suite (nspinor=2 makes Elk's occmax 1.0 per state)."""
    first_block = (calc.workdir / "EIGVAL.OUT").read_text().split("k-point")[1]
    state_line = re.compile(r"^\s*\d+\s+(\S+)\s+(\S+)\s*$")
    occ = [float(m.group(2)) for m in map(state_line.match, first_block.splitlines()) if m]
    return sum(o > 0.5 for o in occ)


@pytest.fixture(scope="module")
def graphene_soc(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("parity")
    calc = Structure(GRAPHENE_AVEC, GRAPHENE_SPECIES).get_calculation(
        workdir / "graphene_soc", xc="PW", ngridk=(6, 6, 1), rgkmax=7.0,
        spinorb=True, soc_scale={"C": 3000.0},
    )
    calc.get_energy()
    return calc, _occupied_count(calc)


@pytest.fixture(scope="module")
def bismuthene(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("parity_bi")
    calc = Structure(BI_AVEC, BI_SPECIES).get_calculation(
        workdir / "bi", xc="PW", ngridk=(6, 6, 1), rgkmax=7.0, spinorb=True,
    )
    calc.get_energy()
    return calc, _occupied_count(calc)


@pytest.fixture(scope="module")
def cs_dimerized(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("parity3d")
    calc = Structure(DIAMOND_AVEC, CS_SPECIES).get_calculation(
        workdir / "cs_dimerized", xc="PW", ngridk=(4, 4, 4), rgkmax=7.0,
        spinorb=True, soc_scale={"Cs": 3000.0},
    )
    calc.get_energy()
    return calc, _occupied_count(calc)


# --- error paths (cheap, and they guard the two ways this can be misused) ---


def test_non_trim_k_point_is_rejected(graphene_soc):
    """Inversion maps a generic k to -k, a different point, so no parity
    matrix exists there. Caught in Python before the query is sent; the
    Fortran side independently detects it as a rotated G+k vector with no
    partner in the same basis (elkpy_parity's ierr=2)."""
    calc, ist1 = graphene_soc
    with calc.eigenstate_session(label="nontrim") as session:
        with pytest.raises(ValueError, match="time-reversal-invariant momentum"):
            session.parity((1 / 3, 1 / 3, 0), 1, ist1)


def test_structure_without_inversion_is_rejected(tmp_path):
    """h-BN's two sublattices carry different species, so the honeycomb's
    inversion centre is gone (Elk's tsyminv is false) and the parity
    operator is undefined. Reported by the Fortran side as a session error
    that leaves the session alive, not a crash."""
    calc = Structure(HBN_AVEC, HBN_SPECIES).get_calculation(
        tmp_path / "hbn", xc="PW", ngridk=(4, 4, 1), rgkmax=7.0
    )
    calc.get_energy()
    with calc.eigenstate_session(label="noinv") as session:
        with pytest.raises(ValueError, match="no inversion symmetry"):
            session.parity((0, 0, 0), 1, 4)
        # the session must still be usable afterwards
        assert session.get_eigenstates((0, 0, 0)).energies.size > 0


# --- the operator's own structure ---


def test_parity_operator_is_an_involution_on_a_gapped_window(graphene_soc):
    """P is Hermitian with P^2 = 1 and eigenvalues exactly +-1 on an
    inversion-invariant window. Unlike the su(2)/Casimir identities of
    docs/design.md #19, this survives the band-window truncation, because
    [I, H] = 0 makes a gapped window invariant rather than a mere slice."""
    calc, ist1 = graphene_soc
    result = calc.get_parity(k=(0, 0, 0), ist0=1, ist1=ist1)
    p = result.pmat
    import numpy as np

    assert np.allclose(p, p.conj().T, atol=5e-2)
    assert np.allclose(p @ p, np.eye(len(p)), atol=1e-1)
    xi = symmetry.parity_eigenvalues(p)
    assert set(np.round(xi).astype(int)) <= {-1, 1}
    # Kramers: with time reversal and inversion every level is two-fold
    # degenerate, so both the state count and the -1 count must be even
    assert len(xi) % 2 == 0
    assert int(np.sum(xi < 0)) % 2 == 0


# --- the invariants, against the already-validated WCC results ---


def test_graphene_fu_kane_matches_wcc_z2(graphene_soc):
    """Kane & Mele's QSH prediction for graphene with intrinsic SOC, now
    from 4 k-points instead of a mesh pumping sweep -- and it must agree
    with get_z2_invariant()'s answer for the same system (docs/design.md
    #20, which measured Z2 = 1)."""
    calc, ist1 = graphene_soc
    result = calc.get_fu_kane_invariant(1, ist1, dimension=2)
    assert result["nu"] == 1
    assert set(result["deltas"].values()) <= {-1, 1}
    # exactly one TRIM must carry the odd parity product, which is what
    # makes the total product -1
    assert list(result["deltas"].values()).count(-1) % 2 == 1


def test_bismuthene_fu_kane_matches_wcc_z2(bismuthene):
    """Murakami's QSH prediction for freestanding buckled-honeycomb monolayer
    Bi (PRL 97, 236805 (2006)), from 4 k-points -- and it must agree with
    get_z2_invariant()'s answer for the same system, which
    tests/test_calculation_z2.py measures as Z2 = 1.

    The SECOND independent 2D cross-check of the parity route (graphene is the
    first), and the more informative one: bismuthene needs no soc_scale
    enhancement at all, and its band inversion is an s-p inversion at Gamma
    (HgTe/CdTe-style) rather than graphene's Dirac-point-at-K mechanism.

    Note what is NOT asserted: that the three M-point deltas are equal. They
    are not, and need not be. The honeycomb's inversion centres come in two
    classes -- the hexagon centre, which lies on the 3-fold axis, and the three
    bond midpoints, which do not -- and Elk's findsymcrys picks a bond midpoint
    here (GEOMETRY.OUT puts the atoms at +-(1/3, 1/6, z); SYMCRYS.OUT reports
    both C3 elements carrying non-lattice translations, a symmorphic group in a
    non-standard origin). Shifting the inversion origin by a half-lattice
    vector t multiplies delta(k) by (exp(2 pi i k.2t))^N with N the occupied
    Kramers-pair count, so with N = 15 here the deltas at two of the three M
    points are flipped relative to the on-axis convention. Transported to the
    on-axis origin they read delta(Gamma) = -1, delta(M) = +1 for all three M
    -- C3-symmetric, and the odd delta sitting at Gamma alone is exactly the
    Gamma-centred band inversion test_bi_gap_minimum_is_at_gamma_not_k
    measures. The four-TRIM product, and hence nu, is origin-independent.
    """
    calc, ist1 = bismuthene
    result = calc.get_fu_kane_invariant(1, ist1, dimension=2)
    assert result["nu"] == 1
    assert set(result["deltas"].values()) <= {-1, 1}
    assert list(result["deltas"].values()).count(-1) % 2 == 1


def test_bismuthene_parity_is_window_independent(bismuthene):
    """Z2 is additive mod 2 over independently-gapped band groups, so dropping
    a gapped semicore block must not change the invariant. ist0 = 21 and 25 sit
    above ~8.6 eV and ~6.8 eV gaps at Gamma respectively."""
    calc, ist1 = bismuthene
    for ist0 in (21, 25):
        assert calc.get_fu_kane_invariant(
            ist0, ist1, dimension=2, label=f"fk_bi_{ist0}"
        )["nu"] == 1


def test_cs_dimerized_is_trivial_and_corrects_the_wcc_result(cs_dimerized):
    """The [111]-dimerized diamond ("cesium") structure is topologically
    TRIVIAL, (0; 000) -- which corrects docs/design.md #21, where a single
    six-plane WCC run at nkx=12, nt=7 reported nu0 = 1.

    Why the parity answer is the trustworthy one here:

      * It is exact. There is no mesh: 8 diagonalisations and a product of
        parity eigenvalues. Mesh convergence cannot be an issue.
      * The WCC number on the disputed planes is not converged. Refining
        the same plane (axis 3, offset 0, where the two disagree) gives
        z = 1, 0, 1, 0 for (nkx, nt) = (12,7), (18,9), (24,13), (32,17) --
        it oscillates rather than settling, so the original nkx=12 sample
        carried no information.
      * It is robust to the band window. Every legitimate gapped window
        (ist0 = 1, 9, 21, 25, 29, 37, each sitting above a 6-69 eV gap at
        Gamma) gives nu0 = 0 and nu = (0,0,0). Between some of those
        windows all eight deltas flip together, which cannot change nu0 --
        the even-TRIM sign immunity documented in parsers.symmetry.
      * The plane is genuinely gapped (minimum direct gap 0.19 eV over a
        13x13 scan of the k3=0 plane), so this is not a case of an
        ill-defined invariant, nor the mesh-aliasing failure mode
        docs/design.md #20 hit on graphene at soc_scale=100.

    The physical reading: this hypothetical Cs diamond structure with SOC
    scaled 3000x simply does not realize the phase of Fu, Kane & Mele's
    single-orbital tight-binding model (PRB 76, 045302 (2007) section IV.3).
    Section 21's agreement with FKM's prediction was coincidental -- an
    unconverged number that happened to land on the hoped-for answer.

    This does not impugn the WCC implementation itself, which agrees with
    the parity route on graphene (test_graphene_fu_kane_matches_wcc_z2) and
    is separately validated in 2D on bismuthene; it impugns the cesium
    result and shows the 3D six-plane sweep was run at too coarse a mesh.
    """
    calc, ist1 = cs_dimerized
    result = calc.get_fu_kane_invariant(1, ist1, dimension=3)
    assert len(result["deltas"]) == 8
    assert result["nu0"] == 0
    assert result["nu"] == (0, 0, 0)


def test_cs_dimerized_parity_is_window_independent(cs_dimerized):
    """The same invariant from a window that drops the deep semicore group
    (ist0 = 9, above a ~69 eV gap at Gamma). Z2 is additive mod 2 over
    independently-gapped band groups, so a gapped semicore block must
    contribute nothing -- the check that gives the nu0 = 0 result above its
    credibility, independently of where the window starts."""
    calc, ist1 = cs_dimerized
    full = calc.get_fu_kane_invariant(1, ist1, dimension=3, label="fk_full")
    trimmed = calc.get_fu_kane_invariant(9, ist1, dimension=3, label="fk_trim")
    assert trimmed["nu0"] == full["nu0"] == 0
    assert trimmed["nu"] == full["nu"] == (0, 0, 0)


def test_window_cutting_a_band_group_is_rejected(cs_dimerized):
    """A window whose boundary sits inside a band group is not a
    topological group at all, and its parity product means nothing. Such a
    window can still pass every check in trim_delta() (Hermitian, +-1
    eigenvalues, even Kramers counts) -- found the hard way: ist0 = 19 on
    this structure passed all of those and returned a confident nu0 = 1.
    check_window_gap() is what rejects it."""
    calc, ist1 = cs_dimerized
    with pytest.raises(ValueError, match="cuts|not separated"):
        calc.get_fu_kane_invariant(19, ist1, dimension=3, label="fk_bad")
