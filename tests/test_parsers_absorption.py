"""Synthetic pins for the polarization-resolved interband absorption
spectrum (elkpy.parsers.optical.circular_absorption) and the two small
parsers it is driven by (parsers.eigval, parsers.dielectric) -- no Elk run
needed.

The integration test (tests/test_calculation_absorption.py) checks the
polarization-SUMMED spectrum against Elk's own task 121, an independent
Fortran implementation. That check is blind to how the total is split
between the two circular channels, and blind to an overall factor shared
by both codes' physics only if it is not shared -- so the pins here do the
complementary jobs: fix the prefactor against a closed form, fix the
+/- split against the already-trusted circular_polarization(), and
demonstrate the k -> -k cancellation that makes the zone-integrated
dichroism vanish while the valley-resolved one does not.

Model: the massive Dirac Hamiltonian of one valley (same one
tests/test_parsers_optical.py uses),

    H_tau(k) = v (tau k_x sigma_x + k_y sigma_y) + Delta sigma_z,

whose velocity operator p = dH/dk is exact, so a synthetic
(energies, pmat) pair can be handed straight to parsers.optical.
"""

import numpy as np
import pytest

from elkpy.parsers import dielectric, eigval, optical

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)

VOLUME = 400.0  # Bohr^3, roughly the h-BN slab cell used in the integration test


def _dirac_states(kx, ky, delta, tau=1, v=1.0):
    h = v * (tau * kx * SX + ky * SY) + delta * SZ
    energies, vecs = np.linalg.eigh(h)
    p_cart = [v * tau * SX, v * SY, np.zeros((2, 2), dtype=complex)]
    pmat = np.array([vecs.conj().T @ p @ vecs for p in p_cart])
    return energies, pmat


def _lorentzian(x, width):
    return (width / np.pi) / (x**2 + width**2)


# --- the prefactor, against a closed form ---


def test_two_level_spectrum_matches_the_closed_form():
    """One k-point, one occupied and one empty state: every ingredient of

        Im eps_pm(w) = (4 pi^2 / (Omega w^2)) W_k f_v (1 - f_c/f_max)
                       (1/2)|P_pm|^2 L(w - Delta)

    is known in closed form, so this pins the prefactor 4 pi^2/Omega, the
    1/w^2, the occupation weight and the Lorentzian normalization at once
    -- an overall factor that is wrong here would still pass every
    internal-consistency check in this file."""
    delta = 0.2
    energies = np.array([0.0, delta])
    px = np.zeros((2, 2), dtype=complex)
    px[1, 0] = px[0, 1] = 0.7
    py = np.zeros((2, 2), dtype=complex)
    py[1, 0] = 0.3j
    py[0, 1] = -0.3j
    pmat = np.array([px, py, np.zeros((2, 2), dtype=complex)])
    omega = np.linspace(0.05, 0.5, 200)
    swidth = 0.01

    out = optical.circular_absorption(
        [(energies, pmat)], omega, occupations=np.array([2.0, 0.0]),
        volume=VOLUME, occmax=2.0, swidth=swidth, broadening="lorentzian",
    )

    p_plus = px[1, 0] + 1j * py[1, 0]
    p_minus = px[1, 0] - 1j * py[1, 0]
    pre = 4.0 * np.pi**2 / VOLUME / omega**2 * 2.0 * _lorentzian(omega - delta, swidth)
    assert out["eps2_plus"] == pytest.approx(pre * 0.5 * abs(p_plus) ** 2)
    assert out["eps2_minus"] == pytest.approx(pre * 0.5 * abs(p_minus) ** 2)


def test_elk_lineshape_matches_dielectric_f90s_own_response():
    """broadening="elk" evaluates src/dielectric.f90's finite-s response
    rather than a broadened delta. Pinned here against that expression
    written out directly from the Fortran (Im eps = 4 pi Re[sigma/(w+is)]
    with sigma's two complex denominators), which is what makes the
    integration test's comparison exact instead of a few-percent one."""
    delta = 0.2
    swidth = 0.01
    energies = np.array([0.0, delta])
    px = np.zeros((2, 2), dtype=complex)
    px[1, 0] = px[0, 1] = 0.7
    pmat = np.array([px, np.zeros((2, 2), dtype=complex), np.zeros((2, 2), dtype=complex)])
    omega = np.linspace(0.0, 0.6, 300)

    out = optical.circular_absorption(
        [(energies, pmat)], omega, np.array([2.0, 0.0]), VOLUME,
        swidth=swidth, broadening="elk",
    )

    # dielectric.f90, transcribed: sigma = i/Omega * f/Delta *
    # [z/(w-D+is) + z/(w+D+is)],  eps'' = 4 pi Re[sigma/(w+is)]
    z = abs(px[1, 0]) ** 2
    sigma = 1j / VOLUME * (2.0 * z / delta) * (
        1.0 / (omega - delta + 1j * swidth) + 1.0 / (omega + delta + 1j * swidth)
    )
    expected = 4.0 * np.pi * np.real(sigma / (omega + 1j * swidth))
    assert out["eps2_total"] == pytest.approx(expected)


def test_the_elk_lineshape_vanishes_at_zero_frequency():
    """The resonant term alone goes negative below w = Delta/2; the
    anti-resonant one cancels it exactly at w = 0. Dropping either term
    would leave a spurious (and, at w -> 0, negative) absorption that no
    peak-region comparison would notice."""
    energies = np.array([0.0, 0.2])
    pmat = np.zeros((3, 2, 2), dtype=complex)
    pmat[0, 1, 0] = pmat[0, 0, 1] = 0.7
    omega = np.linspace(0.0, 0.5, 501)
    out = optical.circular_absorption(
        [(energies, pmat)], omega, np.array([2.0, 0.0]), VOLUME,
        swidth=0.01, broadening="elk",
    )
    assert out["eps2_total"][0] == pytest.approx(0.0, abs=1e-12)
    assert out["eps2_total"].min() > -1e-12


def test_the_two_lineshapes_converge_as_the_broadening_shrinks():
    """The delta form is the s -> 0 limit of the "elk" one, so their peak
    values must approach each other as s falls -- the quantitative version
    of "the few-percent residual in the integration test is the delta
    approximation, not a bug"."""
    energies = np.array([0.0, 0.2])
    pmat = np.zeros((3, 2, 2), dtype=complex)
    pmat[0, 1, 0] = pmat[0, 0, 1] = 0.7
    omega = np.linspace(0.15, 0.25, 2001)
    previous = None
    for swidth in (0.02, 0.01, 0.005):
        common = dict(swidth=swidth)
        a = optical.circular_absorption(
            [(energies, pmat)], omega, np.array([2.0, 0.0]), VOLUME,
            broadening="lorentzian", **common
        )["eps2_total"]
        b = optical.circular_absorption(
            [(energies, pmat)], omega, np.array([2.0, 0.0]), VOLUME,
            broadening="elk", **common
        )["eps2_total"]
        mismatch = np.abs(a - b).max() / b.max()
        if previous is not None:
            assert mismatch < previous
        previous = mismatch
    assert previous < 0.05


def test_the_circular_channels_sum_to_the_two_linear_ones():
    """|P_+|^2 + |P_-|^2 = 2(|p^x|^2 + |p^y|^2), so the polarization-summed
    total is Im eps_xx + Im eps_yy -- the algebraic identity the whole
    task-121 cross-check rests on. Checked here against a linear-basis sum
    written out independently, so the identity is tested rather than
    assumed."""
    rng = np.random.default_rng(7)
    energies = np.array([-0.3, -0.1, 0.4, 0.9])
    occ = np.array([2.0, 2.0, 0.0, 0.0])
    pmat = np.array([rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)) for _ in range(3)])
    pmat = np.array([0.5 * (p + p.conj().T) for p in pmat])
    omega = np.linspace(0.1, 1.5, 300)
    swidth = 0.02

    out = optical.circular_absorption(
        [(energies, pmat)], omega, occ, VOLUME, swidth=swidth, broadening="lorentzian",
    )

    linear = np.zeros_like(omega)
    for v in (0, 1):
        for c in (2, 3):
            weight = abs(pmat[0][c, v]) ** 2 + abs(pmat[1][c, v]) ** 2
            linear += weight * _lorentzian(omega - (energies[c] - energies[v]), swidth)
    linear *= 4.0 * np.pi**2 / VOLUME / omega**2 * 2.0
    assert out["eps2_total"] == pytest.approx(linear)


def test_the_frequency_resolved_eta_matches_circular_polarization():
    """At the gap of a single valley the spectrum's own degree of circular
    polarization must reproduce the already-trusted single-transition
    circular_polarization() value, eta = +-1 at the two valleys. This ties
    the new +/- split to the existing one rather than re-deriving it."""
    delta = 0.25
    for tau, expected in ((1, 1.0), (-1, -1.0)):
        energies, pmat = _dirac_states(0.0, 0.0, delta, tau=tau)
        omega = np.linspace(0.05, 0.6, 400)
        out = optical.circular_absorption(
            [(energies, pmat)], omega, np.array([2.0, 0.0]), VOLUME, swidth=0.01,
            broadening="lorentzian",
        )
        peak = int(np.argmax(out["eps2_total"]))
        assert out["eta"][peak] == pytest.approx(expected, abs=1e-10)
        reference = optical.circular_polarization(pmat, valence=1, conduction=2)
        assert out["eta"][peak] == pytest.approx(reference["eta"], abs=1e-10)


def test_the_two_valleys_cancel_in_the_zone_sum_but_not_under_a_mask():
    """Time reversal maps k -> -k and exchanges sigma+ with sigma-, so a
    mesh closed under negation gives Im eps_+ = Im eps_- however strong
    the local dichroism is. The physics is recovered by restricting the
    sum, which is what the `weights` argument exists for -- the same
    argument documented in circular_absorption()'s docstring."""
    delta = 0.25
    valleys = [_dirac_states(0.02, 0.0, delta, tau=tau) for tau in (1, -1)]
    omega = np.linspace(0.05, 0.6, 200)
    occ = np.array([2.0, 0.0])

    both = optical.circular_absorption(
        valleys, omega, occ, VOLUME, swidth=0.01, broadening="lorentzian"
    )
    assert both["eps2_plus"] == pytest.approx(both["eps2_minus"])

    one = optical.circular_absorption(
        valleys, omega, occ, VOLUME, swidth=0.01, weights=[1.0, 0.0],
        broadening="lorentzian",
    )
    peak = int(np.argmax(one["eps2_total"]))
    assert abs(one["eta"][peak]) > 0.99
    # and the masked halves add back up to the unmasked whole
    other = optical.circular_absorption(
        valleys, omega, occ, VOLUME, swidth=0.01, weights=[0.0, 1.0],
        broadening="lorentzian",
    )
    assert one["eps2_plus"] + other["eps2_plus"] == pytest.approx(
        2.0 * both["eps2_plus"]
    )


def test_the_broadened_delta_conserves_oscillator_strength():
    """Integrating w^2 Im eps over the whole spectrum recovers the bare
    sum of |p|^2 with no lineshape left in it -- true for the Gaussian as
    well as the Lorentzian, since both are normalized delta
    approximations. Catches a mis-normalized lineshape, which a
    peak-height comparison alone would absorb into the prefactor."""
    energies = np.array([0.0, 0.3])
    px = np.zeros((2, 2), dtype=complex)
    px[1, 0] = px[0, 1] = 0.5
    pmat = np.array([px, np.zeros((2, 2), dtype=complex), np.zeros((2, 2), dtype=complex)])
    omega = np.linspace(1e-3, 3.0, 40000)
    expected = 4.0 * np.pi**2 / VOLUME * 2.0 * abs(px[1, 0]) ** 2
    for shape, width in (("lorentzian", 0.01), ("gaussian", 0.01)):
        out = optical.circular_absorption(
            [(energies, pmat)], omega, np.array([2.0, 0.0]), VOLUME,
            swidth=width, broadening=shape,
        )
        integral = np.trapezoid(out["eps2_total"] * omega**2, omega)
        assert integral == pytest.approx(expected, rel=2e-2)


def test_nonpositive_frequencies_are_zero_not_divergent():
    energies = np.array([0.0, 0.3])
    pmat = np.zeros((3, 2, 2), dtype=complex)
    pmat[0, 1, 0] = pmat[0, 0, 1] = 1.0
    out = optical.circular_absorption(
        [(energies, pmat)], np.array([0.0, 0.1]), np.array([2.0, 0.0]), VOLUME,
        broadening="lorentzian",
    )
    assert out["eps2_total"][0] == 0.0
    assert np.isfinite(out["eps2_total"]).all()


def test_mismatched_shapes_raise():
    energies = np.array([0.0, 0.3])
    pmat = np.zeros((3, 2, 2), dtype=complex)
    with pytest.raises(ValueError):
        optical.circular_absorption(
            [(energies, pmat)], np.array([0.1]), np.array([2.0, 0.0, 0.0]), VOLUME
        )
    with pytest.raises(ValueError):
        optical.circular_absorption(
            [(energies, pmat)], np.array([0.1]), np.array([2.0, 0.0]), VOLUME,
            weights=[1.0, 1.0],
        )


# --- the two small parsers ---


EIGVAL_TEXT = """     2 : nkpt
     3 : nstsv

     1  0.000000000       0.000000000       0.000000000      : k-point, vkl
 (state, eigenvalue and occupancy below)
     1 -0.5000000000      2.000000000
     2 -0.2000000000      2.000000000
     3  0.3000000000      0.000000000

     2  0.500000000       0.000000000       0.000000000      : k-point, vkl
 (state, eigenvalue and occupancy below)
     1 -0.4000000000      2.000000000
     2 -0.1000000000      2.000000000
     3  0.4000000000      0.000000000

"""


def test_eigval_parser_reads_kpoints_energies_and_occupations(tmp_path):
    path = tmp_path / "EIGVAL.OUT"
    path.write_text(EIGVAL_TEXT)
    kpoints, energies, occupations = eigval.parse_eigval(path)
    assert kpoints.shape == (2, 3)
    assert energies.shape == (2, 3)
    assert energies[1, 2] == pytest.approx(0.4)
    assert occupations[0].tolist() == [2.0, 2.0, 0.0]
    assert eigval.occupations_if_uniform(path).tolist() == [2.0, 2.0, 0.0]


def test_eigval_refuses_k_dependent_occupations(tmp_path):
    path = tmp_path / "EIGVAL.OUT"
    path.write_text(EIGVAL_TEXT.replace("     3  0.4000000000      0.000000000",
                                        "     3  0.4000000000      1.100000000"))
    with pytest.raises(ValueError, match="metallic"):
        eigval.occupations_if_uniform(path)


def test_dielectric_parser_reads_the_two_blocks(tmp_path):
    path = tmp_path / "EPSILON_11.OUT"
    path.write_text(
        "  0.000000000       1.000000000\n"
        "  0.100000000       2.000000000\n"
        "\n"
        "  0.000000000       0.000000000\n"
        "  0.100000000       5.000000000\n"
        "\n"
    )
    energies, eps = dielectric.parse_epsilon(path)
    assert energies.tolist() == [0.0, 0.1]
    assert eps[1] == pytest.approx(2.0 + 5.0j)
