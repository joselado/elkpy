"""Gauge-invariance test for the Wilson-loop / Fukui-Hatsugai-Suzuki (FHS)
arithmetic in elkpy.parsers.berry, on synthetic data -- no Elk run needed.

Per docs/design.md #13: the whole point of the FHS lattice construction
(cond-mat/0503172) is that the Chern number/plaquette flux is exactly
gauge-invariant -- unaffected by the arbitrary per-k, per-band phase freedom
of the underlying wavefunctions. We can't easily change Elk's own gauge, but
we can apply a synthetic gauge transformation directly to parsed overlap
matrices (M(k, k') -> D(k)^dagger @ M(k, k') @ D(k'), D(k) a random diagonal
unitary) and confirm the computed flux/Chern number is unchanged. This
validates the Wilson-loop arithmetic itself, independent of whether the
overlap matrices came from a real (correct) Elk run.
"""

import numpy as np
import pytest

from elkpy.parsers.berry import compute_berry_curvature, compute_berry_curvature_path


def _random_unitary_diag(rng, n):
    phases = rng.uniform(0, 2 * np.pi, size=n)
    return np.diag(np.exp(1j * phases))


def _synthetic_parsed(rng, ngridk, directions, nst):
    n1, n2, n3 = ngridk
    overlaps = {}
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                # well-conditioned random matrices (identity + small random
                # perturbation) so det() stays comfortably away from zero
                m1 = np.eye(nst) + 0.3 * (rng.normal(size=(nst, nst)) + 1j * rng.normal(size=(nst, nst)))
                m2 = np.eye(nst) + 0.3 * (rng.normal(size=(nst, nst)) + 1j * rng.normal(size=(nst, nst)))
                overlaps[(i1, i2, i3)] = (m1, m2)
    return {"ngridk": ngridk, "directions": directions, "overlaps": overlaps}


def _apply_gauge(parsed, gauge):
    """gauge: {(i1,i2,i3): diagonal unitary matrix D(k)}"""
    ngridk = parsed["ngridk"]
    dir1, dir2 = parsed["directions"]
    ns = ngridk

    def neighbor(k, direction):
        idx = list(k)
        idx[direction - 1] = (idx[direction - 1] + 1) % ns[direction - 1]
        return tuple(idx)

    new_overlaps = {}
    for k, (m1, m2) in parsed["overlaps"].items():
        d_k = gauge[k]
        d_k1 = gauge[neighbor(k, dir1)]
        d_k2 = gauge[neighbor(k, dir2)]
        new_overlaps[k] = (d_k.conj().T @ m1 @ d_k1, d_k.conj().T @ m2 @ d_k2)
    return {**parsed, "overlaps": new_overlaps}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_flux_and_chern_number_are_gauge_invariant(seed):
    rng = np.random.default_rng(seed)
    ngridk = (3, 4, 2)
    directions = (1, 2)
    nst = 3
    parsed = _synthetic_parsed(rng, ngridk, directions, nst)

    gauge = {
        (i1, i2, i3): _random_unitary_diag(rng, nst)
        for i1 in range(ngridk[0])
        for i2 in range(ngridk[1])
        for i3 in range(ngridk[2])
    }
    gauged = _apply_gauge(parsed, gauge)

    before = compute_berry_curvature(parsed)
    after = compute_berry_curvature(gauged)

    np.testing.assert_allclose(before["flux"], after["flux"], atol=1e-10)
    np.testing.assert_allclose(before["chern_number"], after["chern_number"], atol=1e-10)
    assert before["max_flux"] == pytest.approx(after["max_flux"], abs=1e-10)


def test_trivial_identity_overlaps_give_zero_flux():
    """Sanity check independent of gauge invariance: if every link is
    exactly the identity matrix (perfectly adiabatic, zero curvature by
    construction), the flux must be exactly zero everywhere."""
    ngridk = (2, 2, 1)
    directions = (1, 2)
    nst = 2
    overlaps = {
        (i1, i2, i3): (np.eye(nst), np.eye(nst))
        for i1 in range(ngridk[0])
        for i2 in range(ngridk[1])
        for i3 in range(ngridk[2])
    }
    parsed = {"ngridk": ngridk, "directions": directions, "overlaps": overlaps}
    result = compute_berry_curvature(parsed)
    np.testing.assert_allclose(result["flux"], 0.0, atol=1e-12)
    np.testing.assert_allclose(result["chern_number"], 0.0, atol=1e-12)


def _single_band_mesh(ngridk):
    return {
        (i1, i2, i3): (np.array([[1.0 + 0j]]), np.array([[1.0 + 0j]]))
        for i1 in range(ngridk[0])
        for i2 in range(ngridk[1])
        for i3 in range(ngridk[2])
    }


def test_flux_sign_matches_fhs_eq8_numerator_role():
    """Pins the *sign* of the Python-side Wilson-loop arithmetic -- something
    gauge invariance alone cannot do (conj(M) is exactly as gauge-invariant
    as M, since a global sign flip of theta commutes with the gauge-phase
    cancellation in test_flux_and_chern_number_are_gauge_invariant above).

    FHS eq. 8's link product is
    w = U1(k) U2(k+e1) U1(k+e2)^-1 U2(k)^-1, and elkpy's reported flux is
    the BERRY PHASE of that loop, gamma = -arg(w) (parsers.berry._berry_phase
    -- the King-Smith--Vanderbilt/Resta negation, which puts curvature and
    Chern numbers in the standard Xiao-Chang-Niu convention; see
    docs/design.md #22). So a +theta0 phase on U1(k), which enters w's
    numerator, must give -theta0 flux.

    This pins the elkpy.parsers.berry convention given overlap matrices in
    the M(a,b) = <psi_a(k)|psi_b(k+e)> convention; that the Fortran side
    (elkpy_berry.f90) actually produces matrices in that convention is
    confirmed separately and empirically by
    tests/test_calculation_momentum.py's Kubo cross-check, which agrees
    end-to-end against a real binary."""
    ngridk = (2, 2, 1)
    directions = (1, 2)
    theta0 = 0.7
    overlaps = _single_band_mesh(ngridk)
    # U1(k=(0,0,0)) enters eq. 8's product in the numerator (no inverse),
    # and flux = -arg(product), so -> -theta0
    overlaps[(0, 0, 0)] = (np.array([[np.exp(1j * theta0)]]), overlaps[(0, 0, 0)][1])
    parsed = {"ngridk": ngridk, "directions": directions, "overlaps": overlaps}
    flux = compute_berry_curvature(parsed)["flux"]
    assert flux[0, 0, 0] == pytest.approx(-theta0, abs=1e-12)


def test_flux_sign_matches_fhs_eq8_denominator_role():
    """Companion to the numerator-role test above: U1(k+e2) enters eq. 8's
    product inverted (denominator), so with flux = -arg(product) a +theta0
    link phase there must give +theta0 flux at k, not -theta0."""
    ngridk = (2, 2, 1)
    directions = (1, 2)
    theta0 = 0.7
    overlaps = _single_band_mesh(ngridk)
    # k+e2 for k=(0,0,0) is (0,1,0); its U1 enters eq. 8 as U1(k+e2)^-1
    overlaps[(0, 1, 0)] = (np.array([[np.exp(1j * theta0)]]), overlaps[(0, 1, 0)][1])
    parsed = {"ngridk": ngridk, "directions": directions, "overlaps": overlaps}
    flux = compute_berry_curvature(parsed)["flux"]
    assert flux[0, 0, 0] == pytest.approx(theta0, abs=1e-12)


# --- path mode (task 9001, small Wilson loop per arbitrary point) ---


def _synthetic_path_point(rng, nst):
    """One point's four cyclic edge matrices (corner1->2->3->4->1), well
    conditioned (identity + small random perturbation) like the mesh helper
    above."""
    edges = []
    for _ in range(4):
        m = np.eye(nst) + 0.3 * (rng.normal(size=(nst, nst)) + 1j * rng.normal(size=(nst, nst)))
        edges.append(m)
    return tuple(edges)


def _apply_edge_gauge(edges, corner_gauges):
    """Gauge transform for the path/single-plaquette case: each of the 4
    corners gets its own independent diagonal unitary D_i, and edge i->j
    transforms as M -> D_i^dagger @ M @ D_j (same rule as the mesh case,
    applied around the 4-corner cycle instead of a periodic mesh)."""
    d1, d2, d3, d4 = corner_gauges
    m12, m23, m34, m41 = edges
    return (
        d1.conj().T @ m12 @ d2,
        d2.conj().T @ m23 @ d3,
        d3.conj().T @ m34 @ d4,
        d4.conj().T @ m41 @ d1,
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_path_flux_and_curvature_are_gauge_invariant(seed):
    rng = np.random.default_rng(seed)
    nst = 3
    edges = _synthetic_path_point(rng, nst)
    corner_gauges = [_random_unitary_diag(rng, nst) for _ in range(4)]
    gauged_edges = _apply_edge_gauge(edges, corner_gauges)

    parsed = {
        "directions": (1, 2),
        "dk": 0.01,
        "bvec": np.eye(3),
        "points": [((0.1, 0.2, 0.0), edges)],
    }
    gauged_parsed = {**parsed, "points": [((0.1, 0.2, 0.0), gauged_edges)]}

    before = compute_berry_curvature_path(parsed)[0]
    after = compute_berry_curvature_path(gauged_parsed)[0]

    assert before["flux"] == pytest.approx(after["flux"], abs=1e-10)
    assert before["curvature"] == pytest.approx(after["curvature"], abs=1e-10)


def test_path_flux_matches_product_of_link_variables():
    """Pins the sign/formula of compute_berry_curvature_path: with a single
    band and only one edge carrying a nonzero phase (others exactly the
    identity), the flux must equal exactly minus that phase: the path
    formula is a plain product of all four edge link variables around the
    loop (so every edge enters the product identically, unlike the mesh
    formula's two numerator/two denominator terms), and the reported flux
    is the Berry phase, gamma = -arg(product) -- parsers.berry._berry_phase,
    the King-Smith--Vanderbilt/Resta negation putting curvature in the
    standard Xiao-Chang-Niu convention (docs/design.md #22)."""
    theta0 = 0.6
    identity = np.array([[1.0 + 0j]])
    phased = np.array([[np.exp(1j * theta0)]])
    for edge_index in range(4):
        edges = [identity, identity, identity, identity]
        edges[edge_index] = phased
        parsed = {
            "directions": (1, 2),
            "dk": 0.01,
            "bvec": np.eye(3),
            "points": [((0.0, 0.0, 0.0), tuple(edges))],
        }
        result = compute_berry_curvature_path(parsed)[0]
        assert result["flux"] == pytest.approx(-theta0, abs=1e-12)


def test_path_curvature_area_normalization():
    """The curvature is flux divided by the loop's actual Cartesian area
    (2*dk*b1) x (2*dk*b2), not a bare dk*dk -- exercised here with a
    non-orthogonal bvec (hexagonal-like) so an area bug that only shows up
    for a skewed reciprocal lattice wouldn't be masked by an identity bvec."""
    theta0 = 0.5
    dk = 0.02
    identity = np.array([[1.0 + 0j]])
    phased = np.array([[np.exp(1j * theta0)]])
    bvec = np.array([[1.0, 0.0, 0.0], [-0.5, 3**0.5 / 2, 0.0], [0.0, 0.0, 5.0]])
    parsed = {
        "directions": (1, 2),
        "dk": dk,
        "bvec": bvec,
        "points": [((0.0, 0.0, 0.0), (phased, identity, identity, identity))],
    }
    result = compute_berry_curvature_path(parsed)[0]
    edge1 = 2 * dk * bvec[0]
    edge2 = 2 * dk * bvec[1]
    expected_area = np.linalg.norm(np.cross(edge1, edge2))
    assert result["curvature"] == pytest.approx(-theta0 / expected_area, rel=1e-10)
