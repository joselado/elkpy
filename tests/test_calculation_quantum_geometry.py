"""Integration tests for Calculation.get_quantum_geometry() (quantum metric
+ Berry curvature, task 9002 EigenstateSession.overlap() queries -- no new
Fortran task, see docs/design.md #15), against a real compiled elk binary.

Skipped if the elk binary hasn't been built, same as test_calculation_si.py.
"""

import re

import numpy as np
import pytest

from elkpy import config, spec
from elkpy.parsers import berry, quantum_geometry as qg
from elkpy.structure import Structure

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)

SI_AVEC = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
SI_SPECIES = {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}

A, VACUUM = 4.743210000, 20.0  # Bohr, same hBN slab as notebooks/05_berry_curvature.ipynb
HBN_AVEC = [(A, 0.0, 0.0), (-A / 2, A * 3**0.5 / 2, 0.0), (0.0, 0.0, VACUUM)]
HBN_SPECIES = {"B": [(1 / 3, 2 / 3, 0.5)], "N": [(2 / 3, 1 / 3, 0.5)]}


@pytest.fixture
def si_calculation(tmp_path):
    s = Structure(SI_AVEC, SI_SPECIES)
    return s.get_calculation(tmp_path / "si", xc="PW", ngridk=(4, 4, 4))


@pytest.fixture
def hbn_calculation(tmp_path):
    calc = Structure(HBN_AVEC, HBN_SPECIES).get_calculation(
        tmp_path / "hbn", xc="PW", ngridk=(6, 6, 1), rgkmax=7.0
    )
    calc.get_energy()
    # occupied-band count from EIGVAL.OUT's own occupation numbers, not an
    # assumed electron count -- core electrons aren't among the valence
    # bands nstsv indexes at all (same pitfall noted in docs/design.md #13
    # for get_berry_curvature_path on this same structure).
    first_block = (calc.workdir / "EIGVAL.OUT").read_text().split("k-point")[1]
    state_line = re.compile(r"^\s*\d+\s+(\S+)\s+(\S+)\s*$")
    occ = [float(m.group(2)) for m in map(state_line.match, first_block.splitlines()) if m]
    ist0, ist1 = 1, sum(o > 1.0 for o in occ)
    return calc, ist0, ist1


def test_normalization_prevents_truncation_divergence(si_calculation):
    """The failure mode the Loewdin-normalization fix (parsers.quantum_geometry
    ._normalize_overlap) targets, demonstrated on real genolpq output rather
    than the synthetic eps in test_quantum_geometry_gauge_invariance.py:
    raw (unnormalized) quantum_distance()/dk^2 diverges as dk shrinks (the
    truncation-deficiency offset ~2*eps*J/dk^2 blows up), while the
    Loewdin-normalized g11 converges. Both are computed from the exact same
    session queries at each dk, so this isolates the normalization step as
    the only difference."""
    kpoint = (0.05, 0.11, 0.03)
    bvec = si_calculation._reciprocal_vectors()
    ist0, ist1 = 1, 4

    raw, normalized = [], []
    with si_calculation.eigenstate_session(label="raw_vs_norm") as session:
        s0 = session.overlap(kpoint, kpoint, ist0, ist1)
        for dk in (0.02, 0.01, 0.005):
            v1 = dk * bvec[0]
            dk1 = np.linalg.norm(v1)
            k1 = (kpoint[0] + dk, kpoint[1], kpoint[2])
            s1 = session.overlap(k1, k1, ist0, ist1)
            m1 = session.overlap(kpoint, k1, ist0, ist1)
            raw.append(qg.quantum_distance(m1) / dk1**2)
            normalized.append(qg.quantum_distance(qg._normalize_overlap(m1, s0, s1)) / dk1**2)

    # normalized g11 must stabilize (successive differences shrink)...
    assert abs(normalized[2] - normalized[1]) < abs(normalized[1] - normalized[0])
    # ...while the raw quantity's successive differences must grow instead
    # (diverging as dk -> 0), the opposite trend -- this is the concrete
    # signature of the uncorrected 1/dk^2 truncation artifact.
    assert abs(raw[2] - raw[1]) > abs(raw[1] - raw[0])


def test_curvature_matches_berry_curvature_path_on_identical_corners(si_calculation):
    """Cross-checks get_quantum_geometry()'s curvature against
    get_berry_curvature_path() (task 9001) on literally the same four
    corners, the same reasoning as
    test_calculation_berry.py::test_path_and_mesh_conventions_agree:
    get_berry_curvature_path's centered loop at k0 with half-width dk_c
    visits corners k0-d1-d2, k0+d1-d2, k0+d1+d2, k0-d1+d2; anchoring
    get_quantum_geometry's own (non-centered) loop at
    k_anchor = k0-d1-d2 with a full step dk_o = 2*dk_c visits the identical
    four points in the identical cyclic order. Both ultimately reuse the
    same Fortran fresh-diagonalisation routine (elkpy_wfcorner, task 9001
    directly and task 9002's OVERLAP query both) -- agreement here confirms
    get_quantum_geometry's Python-level loop construction (session.py +
    parsers/quantum_geometry.py) didn't introduce a sign/corner-order bug
    of its own, not bit-for-bit identity (independent diagonalisations)."""
    ist0, ist1 = 1, 4
    dk_c = 0.02
    k0 = (0.05, 0.11, 0.03)
    path_result = si_calculation.get_berry_curvature_path(
        [k0], ist0, ist1, directions=(1, 2), dk=dk_c, label="path_ref"
    )[0]

    dk_o = 2 * dk_c
    k_anchor = (k0[0] - dk_c, k0[1] - dk_c, k0[2])
    qg_result = si_calculation.get_quantum_geometry(
        [k_anchor], ist0, ist1, directions=(1, 2), dk=dk_o, label="qg_ref"
    )[0]

    assert qg_result["berry_curvature"] == pytest.approx(path_result["curvature"], rel=0.05)


def test_reciprocal_vectors_match_elks_own_bvec(hbn_calculation):
    """_reciprocal_vectors() replicates Elk's src/reciplat.f90 formula
    directly in Python from self.structure.avec (so get_quantum_geometry()
    needs no new Fortran export for it) -- checked here against the real
    bvec Elk itself computes and exports via task 9000
    (parsers.berry.parse_berry_overlaps), on h-BN's hexagonal
    (non-orthogonal) reciprocal lattice specifically: a row-order or
    cross-product-argument-order mistake could easily go unnoticed on a
    cubic cell like Si (whose bvec rows are related by a symmetry the
    mistake might respect) but not here."""
    calc, ist0, ist1 = hbn_calculation
    calc.get_berry_curvature(ist0, ist1, directions=(1, 2))
    parsed = berry.parse_berry_overlaps(calc.workdir / "berry" / spec.OUTPUT_FILES["berry"])
    np.testing.assert_allclose(calc._reciprocal_vectors(), parsed["bvec"], atol=1e-8)


def test_hbn_gkm_valley_quantum_geometry(hbn_calculation):
    """Benchmark on monolayer h-BN's occupied valence manifold along the
    Gamma-K-M-K' set of high-symmetry points (see notebooks/07_quantum_geometry.ipynb
    for the full path plot), checking properties required by symmetry --
    not just plausible-looking numbers:

    - The quantum metric must be positive semi-definite everywhere (it's a
      Gram-matrix-like quantity, Re Tr[dP dP] up to normalization).
    - Time-reversal symmetry requires Omega(-k)=-Omega(k) and g_ab(-k)=g_ab(k)
      pointwise -- K'=-K (mod a reciprocal lattice vector) makes this a
      sharp, checkable prediction for K/K' (same argument already verified
      for curvature alone in docs/design.md #13/test_calculation_berry.py;
      here it's extended to the metric's diagonal). Gamma and M are
      time-reversal-invariant momenta, forcing curvature to vanish there.
    - det(g) >= (Omega/2)^2 at every point -- not a loose plausibility
      band but an exact theorem: Q_ab = g_ab - (i/2)*Omega_ab is
      Tr[da_P Q db_P], a sum over band indices of Gram-matrix-like
      <.|Q|.> terms, hence positive semi-definite as a 2x2 Hermitian
      matrix in the direction indices (a, b) -- and det(Q) >= 0 for a PSD
      2x2 Hermitian matrix is exactly det(g) - (Omega/2)^2 >= 0. This ties
      the (new) metric's absolute scale to the (already-verified)
      curvature value at the same point without dividing by a curvature
      that can be near zero (Gamma, M): a normalization bug in the metric
      alone (but not curvature, which is insensitive to the Loewdin fix)
      would violate this exact inequality, not just look like an odd
      number.
    """
    calc, ist0, ist1 = hbn_calculation
    points = {"Gamma": (0, 0, 0), "K": (1 / 3, 1 / 3, 0), "Kprime": (-1 / 3, -1 / 3, 0), "M": (0.5, 0, 0)}
    result = calc.get_quantum_geometry(
        list(points.values()), ist0, ist1, directions=(1, 2), dk=0.01, label="qg_gkm"
    )
    by_name = dict(zip(points, result))

    for name, r in by_name.items():
        eigvals = np.linalg.eigvalsh(r["g"])
        assert (eigvals > -1e-6).all(), f"metric not PSD at {name}: eigenvalues {eigvals}"

    # Gamma/M are time-reversal-invariant momenta, forcing curvature to
    # vanish there exactly in the continuum -- the residual here is pure
    # dk=0.01 discretization noise (M's is larger than Gamma's, ~0.08 vs
    # ~0.0001, but both are <1% of the genuine ~8 Bohr^-2 valley signal at
    # K/K' checked below, not a comparable-magnitude effect).
    assert by_name["Gamma"]["berry_curvature"] == pytest.approx(0.0, abs=0.1)
    assert by_name["M"]["berry_curvature"] == pytest.approx(0.0, abs=0.1)

    k_curv = by_name["K"]["berry_curvature"]
    kp_curv = by_name["Kprime"]["berry_curvature"]
    assert k_curv == pytest.approx(-kp_curv, rel=0.02)
    assert abs(k_curv) > 5  # sanity: a real, non-vanishing valley curvature (~8 Bohr^-2 expected)

    k_g, kp_g = by_name["K"]["g"], by_name["Kprime"]["g"]
    assert k_g[0, 0] == pytest.approx(kp_g[0, 0], rel=1e-2)
    assert k_g[1, 1] == pytest.approx(kp_g[1, 1], rel=1e-2)

    # exact theorem (see docstring), checked at all four points, not just K --
    # M/Gamma have curvature near zero, so this is the version of the earlier
    # sqrt(det g)/(|Omega|/2) idea that doesn't divide by (near-)zero there
    for name, r in by_name.items():
        det_g = np.linalg.det(r["g"])
        assert det_g >= (r["berry_curvature"] / 2) ** 2 - 1e-6, f"PSD bound violated at {name}"


def test_hbn_metric_offdiagonal_parity_improves_with_smaller_dk(hbn_calculation):
    """Companion to test_hbn_gkm_valley_quantum_geometry: the off-diagonal
    metric component g12 converges more slowly with dk than g11/g22 or the
    curvature (it's built from a difference of three comparable-magnitude
    quantum_distance() values, D(v1+v2)-D(v1)-D(v2), so its own O(dk)
    discretization error is amplified relative to its O(dk^2) signal) --
    worth pinning explicitly rather than silently relying on a single dk's
    K/K' agreement being tight, unlike the diagonal case above. The
    discriminating check: the K vs K' gap in g12 must shrink as dk shrinks,
    not just happen to be small at one particular dk -- checked over three
    successive halvings (dk = 0.02, 0.01, 0.005), which empirically shrink
    the gap by very close to a factor of 2 each time (1.83, 0.97, 0.49 --
    the textbook signature of a clean O(dk) discretization error going to
    zero, not noise), i.e. K and K' really do converge to a common value as
    the time-reversal theorem (g_ab(-k)=g_ab(k)) requires, not to two
    different values (which would mean a bug in the polarization identity,
    not just slow discretization convergence)."""
    calc, ist0, ist1 = hbn_calculation
    points = [(1 / 3, 1 / 3, 0), (-1 / 3, -1 / 3, 0)]

    gaps = []
    for dk in (0.02, 0.01, 0.005):
        result = calc.get_quantum_geometry(points, ist0, ist1, directions=(1, 2), dk=dk, label=f"qg_g12_{dk}")
        gaps.append(abs(result[0]["g"][0, 1] - result[1]["g"][0, 1]))

    assert gaps[1] < gaps[0]
    assert gaps[2] < gaps[1]
