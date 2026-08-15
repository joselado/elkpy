"""Sign-convention and correctness pins for elkpy.parsers.optical --
the Kubo-form quantum geometric tensor and the interband circular
polarization -- on synthetic models, no Elk run needed.

Same spirit as test_berry_gauge_invariance.py / test_wilson_gauge_invariance.py:
the decisive checks here are not internal consistency (a sign-flipped
Berry curvature is exactly as internally consistent as the right one) but
agreement with an INDEPENDENT already-trusted code path applied to the
same underlying wavefunctions, plus one closed-form analytic value.

The model is the massive Dirac (gapped graphene / two-band k.p) Hamiltonian
of a single valley,

    H_tau(k) = v (tau k_x sigma_x + k_y sigma_y) + Delta sigma_z,

with valley index tau = +-1, for which the velocity operator is exactly
p = dH/dk = v (tau sigma_x, sigma_y) -- no numerical differentiation
needed anywhere, so a synthetic (energies, pmat) pair can be handed
straight to parsers.optical. This is the standard model behind
valley-selective circular dichroism (Yao, Xiao & Niu, PRB 77, 235406
(2008); Xiao, Liu, Feng, Xu & Yao, PRL 108, 196802 (2012)), where the
zone-corner transition is perfectly sigma+ selective at one valley and
perfectly sigma- selective at the other.
"""

import numpy as np
import pytest

from elkpy.parsers import berry, optical

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def _dirac_hamiltonian(kx, ky, delta, tau=1, v=1.0):
    return v * (tau * kx * SX + ky * SY) + delta * SZ


def _dirac_states(kx, ky, delta, tau=1, v=1.0):
    """Eigen-decomposition plus the exact velocity matrix in that
    eigenbasis. Returns (energies, pmat) in parsers.optical's own
    convention: energies ascending, pmat shape (3, 2, 2) with
    pmat[a][n, m] = <n|p_a|m>.
    """
    h = _dirac_hamiltonian(kx, ky, delta, tau, v)
    energies, vecs = np.linalg.eigh(h)
    # p = dH/dk, exact for this model -- no finite difference
    p_cart = [v * tau * SX, v * SY, np.zeros((2, 2), dtype=complex)]
    pmat = np.array([vecs.conj().T @ p @ vecs for p in p_cart])
    return energies, pmat


def _dirac_eigenvector(kx, ky, delta, band, tau=1, v=1.0):
    _, vecs = np.linalg.eigh(_dirac_hamiltonian(kx, ky, delta, tau, v))
    return vecs[:, band]


# --- circular dichroism: the sign-of-the-effect prediction ---


@pytest.mark.parametrize("tau,expected", [(1, 1.0), (-1, -1.0)])
def test_circular_polarization_is_perfectly_valley_selective_at_the_gap(tau, expected):
    """At the Dirac point itself the valence->conduction transition is
    perfectly circularly polarized, with opposite handedness at the two
    valleys: eta(K) = +1, eta(K') = -1 (Xiao et al., PRL 108, 196802
    (2012)). This pins BOTH the magnitude and the relative sign of
    circular_polarization()'s P_+- convention against an analytic result,
    which no internal-consistency check could do."""
    energies, pmat = _dirac_states(0.0, 0.0, delta=0.5, tau=tau)
    result = optical.circular_polarization(pmat, valence=1, conduction=2)
    assert result["eta"] == pytest.approx(expected, abs=1e-12)
    # the suppressed handedness is exactly zero here, not merely small
    suppressed = "i_minus" if expected > 0 else "i_plus"
    assert result[suppressed] == pytest.approx(0.0, abs=1e-24)


def test_circular_polarization_decays_away_from_the_valley():
    """Perfect selectivity is a property of the zone corner, not of the
    model everywhere: moving off the Dirac point mixes the two
    handednesses, so |eta| must fall below 1 and keep falling."""
    etas = [
        abs(
            optical.circular_polarization(
                _dirac_states(kx, 0.0, delta=0.5)[1], valence=1, conduction=2
            )["eta"]
        )
        for kx in (0.0, 0.25, 0.5, 1.0)
    ]
    assert etas[0] == pytest.approx(1.0, abs=1e-12)
    assert all(a > b for a, b in zip(etas, etas[1:]))


def test_circular_polarization_rejects_a_forbidden_transition():
    """eta of a dipole-forbidden transition is undefined, not zero."""
    pmat = np.zeros((3, 2, 2), dtype=complex)
    with pytest.raises(ValueError, match="dipole-forbidden"):
        optical.circular_polarization(pmat, valence=1, conduction=2)


def test_degenerate_group_sums_intensities():
    """A band group sums |P_+-|^2 over every (c, v) pair -- the correct
    handling for a degenerate manifold. Built here from two decoupled
    copies of the same valley, whose group eta must equal the single
    transition's eta while carrying twice the oscillator strength."""
    energies, pmat = _dirac_states(0.0, 0.0, delta=0.5, tau=1)
    # block-diagonal 4x4: two identical, non-interacting copies
    big = np.zeros((3, 4, 4), dtype=complex)
    for a in range(3):
        big[a][np.ix_([0, 2], [0, 2])] = pmat[a]
        big[a][np.ix_([1, 3], [1, 3])] = pmat[a]
    single = optical.circular_polarization(pmat, valence=1, conduction=2)
    group = optical.circular_polarization(big, valence=(1, 2), conduction=(3, 4))
    assert group["eta"] == pytest.approx(single["eta"], abs=1e-12)
    assert group["i_plus"] == pytest.approx(2 * single["i_plus"], rel=1e-12)


# --- Kubo geometry vs. an independent code path and a closed form ---


def _direct_curvature(kx, ky, delta, band, h=1e-5, tau=1):
    """Berry curvature by DIRECT numerical differentiation of the model's
    eigenvectors, Omega = -2 Im <d_x u|d_y u>, in the standard convention
    A = i<u|grad_k u> (Xiao, Chang & Niu, RMP 82, 1959 (2010) eq. 1.10).

    Independent of both parsers.optical (no matrix elements) and
    parsers.berry (no link variables) -- the neutral third path used here
    to say which of those two carries the standard sign.
    """
    def vec(x, y):
        _, v = np.linalg.eigh(_dirac_hamiltonian(x, y, delta, tau))
        c = v[:, band]
        # local smooth phase convention, so the finite differences below
        # differentiate one continuous branch rather than random phases
        i = np.argmax(np.abs(c))
        return c * np.exp(-1j * np.angle(c[i]))

    dux = (vec(kx + h, ky) - vec(kx - h, ky)) / (2 * h)
    duy = (vec(kx, ky + h) - vec(kx, ky - h)) / (2 * h)
    return -2 * np.imag(np.vdot(dux, duy))


def _fhs_curvature_from_states(kx, ky, delta, band, dk, tau=1):
    """Berry curvature at (kx, ky) from parsers.berry's OWN Fukui-Hatsugai-
    Suzuki link-variable/plaquette-flux arithmetic (already sign-pinned
    against FHS eq. 8 in test_berry_gauge_invariance.py), evaluated on the
    model's exact eigenvectors around a small square loop, in exactly the
    cyclic corner order, Berry-phase sign (parsers.berry._berry_phase) and
    flux/area normalization parsers.berry.compute_berry_curvature_path
    uses.

    Shares no arithmetic with parsers.optical: overlaps of wavefunctions
    vs. matrix elements of an operator.
    """
    corners = [
        (kx - dk, ky - dk),
        (kx + dk, ky - dk),
        (kx + dk, ky + dk),
        (kx - dk, ky + dk),
    ]
    vecs = [_dirac_eigenvector(cx, cy, delta, band, tau) for cx, cy in corners]
    links = [
        berry._link_variable(np.array([[np.vdot(vecs[i], vecs[(i + 1) % 4])]]))
        for i in range(4)
    ]
    flux = berry._berry_phase(links[0] * links[1] * links[2] * links[3])
    return flux / (2 * dk) ** 2


@pytest.mark.parametrize("tau", [1, -1])
@pytest.mark.parametrize("kx,ky", [(0.0, 0.0), (0.3, -0.2), (0.5, 0.4)])
def test_kubo_curvature_matches_direct_differentiation(tau, kx, ky):
    """The decisive sign pin: Kubo curvature (a sum over states, built
    from velocity matrix elements) against direct numerical
    differentiation of the wavefunctions (no matrix elements at all).
    Agreement in SIGN as well as magnitude is what this buys -- a
    conjugation-sign flip survives every gauge-invariance and
    Hermiticity check, since conj(M) is exactly as gauge-invariant as M.

    Both sides here are the standard convention A = i<u|grad_k u>,
    Omega = curl A (Xiao, Chang & Niu, RMP 82, 1959 (2010))."""
    energies, pmat = _dirac_states(kx, ky, delta=0.5, tau=tau)
    kubo = optical.kubo_berry_curvature(energies, pmat, ist0=1, ist1=1)
    direct = _direct_curvature(kx, ky, delta=0.5, band=0, tau=tau)
    assert kubo == pytest.approx(direct, rel=1e-6)


@pytest.mark.parametrize("tau", [1, -1])
@pytest.mark.parametrize("kx,ky", [(0.0, 0.0), (0.3, -0.2), (0.5, 0.4)])
def test_kubo_curvature_agrees_with_the_parsers_berry_wilson_loop(tau, kx, ky):
    """The two routes elkpy offers to the Berry curvature -- a sum over
    velocity matrix elements (this module) and a Wilson loop over
    wavefunction overlaps (parsers.berry) -- must agree in sign as well as
    magnitude on identical states. They share no arithmetic.

    This equality is what fixed elkpy's Berry-phase sign convention. It
    originally FAILED with a clean factor of -1 (magnitudes agreeing to 7
    digits), which is how the missing King-Smith--Vanderbilt/Resta negation
    in parsers.berry's flux step was found; test_kubo_curvature_matches
    _direct_differentiation above decided which side was standard. Both now
    use A = i<u|grad_k u>, Omega = curl A -- see parsers.berry._berry_phase
    and docs/design.md #22."""
    energies, pmat = _dirac_states(kx, ky, delta=0.5, tau=tau)
    kubo = optical.kubo_berry_curvature(energies, pmat, ist0=1, ist1=1)
    fhs = _fhs_curvature_from_states(kx, ky, delta=0.5, band=0, dk=1e-3, tau=tau)
    assert kubo == pytest.approx(fhs, rel=2e-3)


def test_kubo_curvature_matches_the_closed_form_at_the_dirac_point():
    """Absolute magnitude pin: for H = d.sigma the two-band curvature is
    +-1/(2*Delta^2) at k = 0, with the valence and conduction bands exactly
    opposite (the two-band sum rule sum_n Omega_n = 0)."""
    delta = 0.5
    energies, pmat = _dirac_states(0.0, 0.0, delta=delta)
    valence = optical.kubo_berry_curvature(energies, pmat, ist0=1, ist1=1)
    conduction = optical.kubo_berry_curvature(energies, pmat, ist0=2, ist1=2)
    assert abs(valence) == pytest.approx(1.0 / (2 * delta**2), rel=1e-12)
    assert conduction == pytest.approx(-valence, rel=1e-12)


@pytest.mark.parametrize("tau", [1, -1])
def test_kubo_curvature_is_valley_antisymmetric(tau):
    """Time reversal maps one valley to the other with opposite curvature
    -- the same K/K' antisymmetry the real-binary h-BN tests assert."""
    e_p, p_p = _dirac_states(0.2, 0.1, delta=0.5, tau=1)
    e_m, p_m = _dirac_states(-0.2, 0.1, delta=0.5, tau=-1)
    assert optical.kubo_berry_curvature(e_p, p_p, 1, 1) == pytest.approx(
        -optical.kubo_berry_curvature(e_m, p_m, 1, 1), rel=1e-12
    )


def test_kubo_metric_is_positive_semidefinite_and_dominates_the_curvature():
    """g must be PSD, and det(g) >= (F/2)^2 -- the exact consequence of
    Q = g - (i/2)F being a positive semi-definite Hermitian matrix, the
    same inequality tests/test_calculation_quantum_geometry.py asserts for
    the finite-difference metric."""
    for kx, ky in [(0.0, 0.0), (0.3, -0.2), (0.7, 0.5)]:
        energies, pmat = _dirac_states(kx, ky, delta=0.5)
        result = optical.kubo_quantum_geometry(energies, pmat, ist0=1, ist1=1)
        g, f = result["g"], result["berry_curvature"]
        assert np.allclose(g, g.T)
        assert np.all(np.linalg.eigvalsh(g) >= -1e-14)
        assert np.linalg.det(g) >= (f / 2) ** 2 - 1e-12
        assert np.allclose(result["Q"], result["Q"].conj().T)


def test_kubo_metric_matches_the_closed_form_at_the_dirac_point():
    """At k = 0 the massive Dirac metric is isotropic, g = 1/(4 Delta^2)
    times the identity -- and saturates det(g) = (F/2)^2, the equality
    case of the Q >= 0 bound (a two-band model's lower band is a maximally
    "coherent" state, the CP^1 case docs/physics.tex Part IV already
    pins the finite-difference metric against)."""
    delta = 0.5
    energies, pmat = _dirac_states(0.0, 0.0, delta=delta)
    result = optical.kubo_quantum_geometry(energies, pmat, ist0=1, ist1=1)
    expected = np.eye(2) / (4 * delta**2)
    assert np.allclose(result["g"], expected, rtol=1e-12)
    assert np.linalg.det(result["g"]) == pytest.approx(
        (result["berry_curvature"] / 2) ** 2, rel=1e-12
    )


# --- guards ---


def test_degenerate_window_boundary_raises():
    """A window not separated from the states outside it makes the
    1/(eps_n - eps_m)^2 weight meaningless; fail loud, don't clip."""
    energies = np.array([0.0, 1e-9, 1.0])
    pmat = np.ones((3, 3, 3), dtype=complex)
    with pytest.raises(ValueError, match="not separated"):
        optical.kubo_berry_curvature(energies, pmat, ist0=1, ist1=1)


def test_window_covering_every_state_raises():
    energies = np.array([0.0, 1.0])
    pmat = np.ones((3, 2, 2), dtype=complex)
    with pytest.raises(ValueError, match="every available state"):
        optical.kubo_berry_curvature(energies, pmat, ist0=1, ist1=2)


def test_band_velocity_is_the_diagonal():
    energies, pmat = _dirac_states(0.3, 0.2, delta=0.5)
    v = optical.band_velocity(pmat, ist=1)
    assert v.shape == (3,)
    assert v[0] == pytest.approx(pmat[0, 0, 0].real)
    assert v[2] == pytest.approx(0.0)
