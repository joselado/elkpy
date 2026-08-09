"""Unit tests for elkpy.parsers.quantum_geometry, on synthetic data -- no
Elk run needed (mirrors tests/test_berry_gauge_invariance.py's style and
role for the Wilson-loop Berry curvature arithmetic).

Three independent things are pinned here, none of which the Berry-curvature
tests already cover:

1. Gauge invariance under a general (non-diagonal) unitary mixing of the
   occupied subspace at each loop corner -- the non-Abelian analogue of
   test_berry_gauge_invariance.py's diagonal-gauge check.
2. That Loewdin normalization actually removes the truncation-offset
   failure mode identified during design (see
   parsers.quantum_geometry._normalize_overlap()'s docstring): a uniform
   overlap deficiency that would otherwise swamp the metric with a spurious
   constant offset.
3. The *absolute* normalization (not just invariance) against an
   analytically known case -- the spin-1/2 / CP^1 quantum geometry worked
   out in Provost & Vallee, "Riemannian structure on manifolds of quantum
   states," Commun. Math. Phys. 76, 289 (1980): for the spin coherent state
   |n(theta,phi)> = (cos(theta/2), e^{i phi} sin(theta/2)), the quantum
   metric is the round-sphere metric over 4, g = (1/4) diag(1, sin^2(theta)).
"""

import numpy as np
import pytest

from elkpy.parsers import quantum_geometry as qg


def _random_orthonormal_basis(rng, dim, nst):
    m = rng.normal(size=(dim, nst)) + 1j * rng.normal(size=(dim, nst))
    q, _ = np.linalg.qr(m)
    return q[:, :nst]


def _random_unitary(rng, n):
    m = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(m)
    phases = np.diagonal(r) / np.abs(np.diagonal(r))
    return q * phases


def _corner_bases(rng, dim, nst):
    return {name: _random_orthonormal_basis(rng, dim, nst) for name in ("0", "1", "2", "12")}


def _overlaps_from_bases(bases):
    def ov(a, b):
        return bases[a].conj().T @ bases[b]

    return {
        "s0": ov("0", "0"),
        "s1": ov("1", "1"),
        "s2": ov("2", "2"),
        "s12": ov("12", "12"),
        "m1": ov("0", "1"),
        "m2": ov("0", "2"),
        "m12": ov("0", "12"),
        "edge_b": ov("1", "12"),
        "edge_c": ov("2", "12"),
    }


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gauge_invariance_under_random_unitary_mixing(seed):
    rng = np.random.default_rng(seed)
    dim, nst = 8, 3
    bases = _corner_bases(rng, dim, nst)
    v1 = np.array([0.01, 0.0, 0.0])
    v2 = np.array([0.0, 0.01, 0.0])

    before = qg.compute_quantum_geometry(_overlaps_from_bases(bases), v1, v2)
    gauged = {name: basis @ _random_unitary(rng, nst) for name, basis in bases.items()}
    after = qg.compute_quantum_geometry(_overlaps_from_bases(gauged), v1, v2)

    np.testing.assert_allclose(before["g"], after["g"], atol=1e-10)
    assert before["berry_curvature"] == pytest.approx(after["berry_curvature"], abs=1e-10)
    np.testing.assert_allclose(before["Q"], after["Q"], atol=1e-10)


def test_normalization_removes_truncation_offset():
    """The critical failure mode identified during design: a uniform raw
    overlap deficiency (mimicking genolpq's ~1e-3 real-space truncation
    floor -- an overlap(k, k, ...) that isn't exactly the identity) would,
    left unnormalized, swamp the O(dk^2) metric signal with an O(eps)
    constant offset (~2*eps*J, J = band-window size). Here every corner is
    *exactly* the same state (the true dk->0 metric is exactly zero) but
    every raw overlap -- including the self-overlaps -- is scaled by
    (1-eps), simulating that deficiency. The Loewdin-normalized result must
    still recover exactly zero; the raw (unnormalized) quantity it's built
    from would not.
    """
    nst = 3
    eps = 1e-3
    shrink = 1 - eps
    identity = np.eye(nst)
    overlaps = {name: shrink * identity for name in ("s0", "s1", "s2", "s12", "m1", "m2", "m12", "edge_b", "edge_c")}

    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    result = qg.compute_quantum_geometry(overlaps, v1, v2)
    np.testing.assert_allclose(result["g"], 0.0, atol=1e-12)

    raw_offset = qg.quantum_distance(shrink * identity)
    assert raw_offset > eps * nst  # order 2*eps*J -- the offset normalization must remove


def _spin_state(theta, phi):
    return np.array([[np.cos(theta / 2)], [np.exp(1j * phi) * np.sin(theta / 2)]])


def test_matches_analytic_bloch_sphere_quantum_geometry():
    """Pins the *absolute* normalization (not just gauge invariance)
    against the Provost & Vallee (1980) spin-1/2 example -- see module
    docstring. Single-band (nst=1), so there is no gauge freedom and
    Loewdin normalization is a no-op (every self-overlap is exactly 1);
    this instead exercises quantum_distance()/the polarization identity
    against a genuine analytic target."""
    theta0, phi0 = np.pi / 3, 0.4
    dk = 1e-4

    def ov(a, b):
        return a.conj().T @ b

    u0 = _spin_state(theta0, phi0)
    u1 = _spin_state(theta0 + dk, phi0)
    u2 = _spin_state(theta0, phi0 + dk)
    u12 = _spin_state(theta0 + dk, phi0 + dk)
    overlaps = {
        "s0": ov(u0, u0),
        "s1": ov(u1, u1),
        "s2": ov(u2, u2),
        "s12": ov(u12, u12),
        "m1": ov(u0, u1),
        "m2": ov(u0, u2),
        "m12": ov(u0, u12),
        "edge_b": ov(u1, u12),
        "edge_c": ov(u2, u12),
    }
    v1 = np.array([dk, 0.0, 0.0])
    v2 = np.array([0.0, dk, 0.0])
    result = qg.compute_quantum_geometry(overlaps, v1, v2)

    assert result["g"][0, 0] == pytest.approx(0.25, abs=1e-5)
    assert result["g"][1, 1] == pytest.approx(0.25 * np.sin(theta0) ** 2, abs=1e-5)
    assert result["g"][0, 1] == pytest.approx(0.0, abs=1e-5)
    assert result["berry_curvature"] == pytest.approx(0.5 * np.sin(theta0), abs=2e-4)
