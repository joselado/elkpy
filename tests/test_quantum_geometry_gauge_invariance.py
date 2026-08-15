"""Unit tests for elkpy.parsers.quantum_geometry, on synthetic data -- no
Elk run needed (mirrors tests/test_berry_gauge_invariance.py's style and
role for the Wilson-loop Berry curvature arithmetic).

Four independent things are pinned here, none of which the Berry-curvature
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
4. That the centered mixed-partial stencil compute_quantum_geometry() uses
   for g12 (see its docstring) genuinely converges as O(dk^2), not O(dk)
   like the plain forward polarization identity -- on a skewed
   reparametrization of the same CP^1 state where g12 is analytically
   nonzero, so there is something nontrivial to converge to.
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


CORNER_NAMES = ("0", "1p", "1m", "2p", "2m", "12pp", "12pm", "12mp", "12mm")


def _corner_bases(rng, dim, nst):
    return {name: _random_orthonormal_basis(rng, dim, nst) for name in CORNER_NAMES}


def _overlaps_from_bases(bases):
    def ov(a, b):
        return bases[a].conj().T @ bases[b]

    return {
        "s0": ov("0", "0"),
        "s1p": ov("1p", "1p"),
        "s1m": ov("1m", "1m"),
        "s2p": ov("2p", "2p"),
        "s2m": ov("2m", "2m"),
        "s12pp": ov("12pp", "12pp"),
        "s12pm": ov("12pm", "12pm"),
        "s12mp": ov("12mp", "12mp"),
        "s12mm": ov("12mm", "12mm"),
        "m1p": ov("0", "1p"),
        "m1m": ov("0", "1m"),
        "m2p": ov("0", "2p"),
        "m2m": ov("0", "2m"),
        "m12pp": ov("0", "12pp"),
        "m12pm": ov("0", "12pm"),
        "m12mp": ov("0", "12mp"),
        "m12mm": ov("0", "12mm"),
        "edge_b": ov("1p", "12pp"),
        "edge_c": ov("2p", "12pp"),
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
    keys = (
        "s0", "s1p", "s1m", "s2p", "s2m", "s12pp", "s12pm", "s12mp", "s12mm",
        "m1p", "m1m", "m2p", "m2m", "m12pp", "m12pm", "m12mp", "m12mm",
        "edge_b", "edge_c",
    )
    overlaps = {name: shrink * identity for name in keys}

    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    result = qg.compute_quantum_geometry(overlaps, v1, v2)
    np.testing.assert_allclose(result["g"], 0.0, atol=1e-12)

    raw_offset = qg.quantum_distance(shrink * identity)
    assert raw_offset > eps * nst  # order 2*eps*J -- the offset normalization must remove


def _spin_state(theta, phi):
    return np.array([[np.cos(theta / 2)], [np.exp(1j * phi) * np.sin(theta / 2)]])


def _overlaps_on_grid(state_fn, a0, b0, dk):
    """Build a full compute_quantum_geometry() overlaps dict for a
    single-band (nst=1) state_fn(a, b) on the centered 3x3 grid around
    (a0, b0), step dk in each coordinate -- the synthetic-data analogue of
    Calculation.get_quantum_geometry()'s own corner construction."""

    def ov(x, y):
        return x.conj().T @ y

    offsets = {
        "0": (0, 0), "1p": (1, 0), "1m": (-1, 0), "2p": (0, 1), "2m": (0, -1),
        "12pp": (1, 1), "12pm": (1, -1), "12mp": (-1, 1), "12mm": (-1, -1),
    }
    states = {name: state_fn(a0 + n1 * dk, b0 + n2 * dk) for name, (n1, n2) in offsets.items()}
    return states, {
        "s0": ov(states["0"], states["0"]),
        "s1p": ov(states["1p"], states["1p"]),
        "s1m": ov(states["1m"], states["1m"]),
        "s2p": ov(states["2p"], states["2p"]),
        "s2m": ov(states["2m"], states["2m"]),
        "s12pp": ov(states["12pp"], states["12pp"]),
        "s12pm": ov(states["12pm"], states["12pm"]),
        "s12mp": ov(states["12mp"], states["12mp"]),
        "s12mm": ov(states["12mm"], states["12mm"]),
        "m1p": ov(states["0"], states["1p"]),
        "m1m": ov(states["0"], states["1m"]),
        "m2p": ov(states["0"], states["2p"]),
        "m2m": ov(states["0"], states["2m"]),
        "m12pp": ov(states["0"], states["12pp"]),
        "m12pm": ov(states["0"], states["12pm"]),
        "m12mp": ov(states["0"], states["12mp"]),
        "m12mm": ov(states["0"], states["12mm"]),
        "edge_b": ov(states["1p"], states["12pp"]),
        "edge_c": ov(states["2p"], states["12pp"]),
    }


def test_matches_analytic_bloch_sphere_quantum_geometry():
    """Pins the *absolute* normalization (not just gauge invariance)
    against the Provost & Vallee (1980) spin-1/2 example -- see module
    docstring. Single-band (nst=1), so there is no gauge freedom and
    Loewdin normalization is a no-op (every self-overlap is exactly 1);
    this instead exercises quantum_distance()/the centered stencil against
    a genuine analytic target.

    The curvature is an independent absolute pin on elkpy's Berry-phase
    SIGN convention (parsers.berry._berry_phase), derived here rather than
    calibrated to the code: with A = i<u|grad u> (Xiao, Chang & Niu),
    A_phi = i<u|d_phi u> = -sin^2(theta/2) and A_theta = 0, so

        Omega_{theta,phi} = d_theta A_phi - d_phi A_theta = -(1/2) sin(theta),

    which integrates over the sphere to -2*pi -- the spin-1/2 monopole
    charge, Berry phase = -(1/2) * solid angle. Note the minus: this test
    asserted +(1/2)sin(theta) while parsers.berry omitted the
    King-Smith--Vanderbilt/Resta negation, i.e. it had been calibrated to
    the code rather than to Provost & Vallee. See docs/design.md #22."""
    theta0, phi0 = np.pi / 3, 0.4
    dk = 1e-4

    _, overlaps = _overlaps_on_grid(_spin_state, theta0, phi0, dk)
    v1 = np.array([dk, 0.0, 0.0])
    v2 = np.array([0.0, dk, 0.0])
    result = qg.compute_quantum_geometry(overlaps, v1, v2)

    assert result["g"][0, 0] == pytest.approx(0.25, abs=1e-5)
    assert result["g"][1, 1] == pytest.approx(0.25 * np.sin(theta0) ** 2, abs=1e-5)
    assert result["g"][0, 1] == pytest.approx(0.0, abs=1e-5)
    assert result["berry_curvature"] == pytest.approx(-0.5 * np.sin(theta0), abs=2e-4)


def test_offdiagonal_metric_centered_stencil_converges_quadratically():
    """The centered mixed-partial stencil compute_quantum_geometry() uses
    for g12 (see its docstring) must converge as O(dk^2), unlike the plain
    forward polarization identity D(v1+v2)-D(v1)-D(v2) it replaced, which
    only converges as O(dk) -- the concrete signature of that O(dk) error
    was measured on real Elk output (bulk h-BN's K/K' valleys, git log) as
    a K-vs-K' gap shrinking by a factor ~2 per dk-halving; this test pins
    the same claim analytically instead, with no Elk run and no genolpq
    truncation floor to muddy the comparison.

    Reusing the diagonal (theta, phi) spin-coherent state from the test
    above would give g12=0 identically (no signal to converge to), so this
    instead uses a skewed reparametrization u=theta, w=phi+theta of the
    exact same physical state -- the round-sphere metric transforms into
    one with a genuine nonzero g_uw = -(1/4) sin^2(u) (derivation: ds^2 =
    (1/4)dtheta^2 + (1/4)sin^2(theta) dphi^2, substitute
    dtheta=du, dphi=dw-du and collect the du*dw cross term).
    """

    def skewed_state(u, w):
        return _spin_state(theta=u, phi=w - u)

    u0, w0 = np.pi / 5, 0.3  # away from sin(u0)=0, where g_uw itself vanishes
    analytic_g_uw = -0.25 * np.sin(u0) ** 2

    def estimates(dk):
        states, overlaps = _overlaps_on_grid(skewed_state, u0, w0, dk)
        v1 = np.array([dk, 0.0, 0.0])
        v2 = np.array([0.0, dk, 0.0])
        centered = qg.compute_quantum_geometry(overlaps, v1, v2)["g"][0, 1]

        # the plain forward polarization identity this replaced, built
        # from the same corner states so the only difference is the
        # stencil, not the underlying data
        d1p = qg.quantum_distance(qg._normalize_overlap(overlaps["m1p"], overlaps["s0"], overlaps["s1p"]))
        d2p = qg.quantum_distance(qg._normalize_overlap(overlaps["m2p"], overlaps["s0"], overlaps["s2p"]))
        d12pp = qg.quantum_distance(qg._normalize_overlap(overlaps["m12pp"], overlaps["s0"], overlaps["s12pp"]))
        forward = (d12pp - d1p - d2p) / (2 * dk * dk)
        return centered, forward

    dks = (0.02, 0.01, 0.005)
    centered_errors = []
    forward_errors = []
    for dk in dks:
        centered, forward = estimates(dk)
        centered_errors.append(abs(centered - analytic_g_uw))
        forward_errors.append(abs(forward - analytic_g_uw))

    # forward: error roughly halves each time dk halves (O(dk))
    assert 1.5 < forward_errors[0] / forward_errors[1] < 2.5
    assert 1.5 < forward_errors[1] / forward_errors[2] < 2.5
    # centered: error roughly quarters each time dk halves (O(dk^2))
    assert 3.0 < centered_errors[0] / centered_errors[1] < 5.0
    assert 3.0 < centered_errors[1] / centered_errors[2] < 5.0
    # guard against a degenerate all-zero stencil bug trivially "converging"
    assert abs(analytic_g_uw) > 0.05
    for centered_error in centered_errors:
        assert centered_error < 0.05 * abs(analytic_g_uw)
    # at the smallest dk the centered stencil is much more accurate
    assert centered_errors[-1] < forward_errors[-1] / 10
