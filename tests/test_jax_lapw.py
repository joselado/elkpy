"""Phase 0c of the Elk-to-JAX port: the LAPW matching coefficients, differentiated.

Study §6 item 0c.  The kill criterion is agreement to machine precision against
`dmatch.f90`'s d(apwalm)/dr = i(G+p) apwalm -- an identity that is EXACT, because the
atomic position enters `match` only through exp(i(G+p).r_alpha), so this item needs no
finite differences and no tolerance argument.

What it does need is the forward half: an exact derivative of the wrong function is
worthless.  `spherical_bessel` and `spherical_harmonics` are checked against SciPy, and
the assembly -- the 1/sqrt(Omega), the conjugation, the t4pil prefactor, the packed lm
layout and the linear solve, none of which SciPy sees -- against its own defining
equation D A = b, rebuilt from SciPy independently.
"""

import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
scipy_special = pytest.importorskip("scipy.special")

import elkjax  # noqa: E402
from elkjax import lapw, memory, phase0c  # noqa: E402

memory.limit_address_space(16.0)

SLOW = pytest.mark.skipif(
    not os.environ.get("ELKPY_RUN_SLOW_TESTS"),
    reason="set ELKPY_RUN_SLOW_TESTS=1 for the full-size G+k set",
)

LMAX = 6


@pytest.fixture(scope="module")
def setup():
    """A small but realistic G+k set: |G+k|R spans the Bessel branch switch."""
    return phase0c.build(lmax=LMAX, ncell=2, gkmax=5.0)


def test_spherical_bessel_against_scipy():
    """Both recurrence branches, and the switch between them at x = lmax.

    Downward alone is wrong by 7.6e-2 at x=20 and upward alone fails for x < l; Elk
    switches at lmax and so does this, which is why the sweep spans both sides.
    """
    for x in (1e-10, 1e-6, 0.01, 0.5, 1.0, 3.0, 5.0, 6.0, 12.0, 20.0, 40.0):
        got = np.asarray(lapw.spherical_bessel(LMAX, jnp.asarray(x)))
        reference = scipy_special.spherical_jn(np.arange(LMAX + 1), x)
        assert np.allclose(got, reference, rtol=1e-12, atol=1e-15), x


def test_spherical_bessel_derivatives_come_from_autodiff():
    """`jax.jacfwd` of the recurrence, against SciPy's own derivative.

    Elk hand-derives these in `sbesseldm` from a polynomial-in-1/x expansion; that this
    agrees is simultaneously a check on the recurrence and on its differentiability.
    """
    for x in (0.5, 3.0, 12.0):
        got = np.asarray(lapw.spherical_bessel_derivative(LMAX, jnp.asarray(x), 1))
        reference = scipy_special.spherical_jn(np.arange(LMAX + 1), x, derivative=True)
        assert np.allclose(got, reference, rtol=1e-11, atol=1e-15), x


def test_spherical_harmonics_against_scipy_in_elks_packed_layout():
    """Value AND index convention: lm = l(l+1)+m, Condon-Shortley included."""
    rng = np.random.default_rng(0)
    for _ in range(4):
        v = rng.normal(size=3)
        got = np.asarray(lapw.spherical_harmonics(LMAX, jnp.asarray(v), t4pil=False))
        r = np.linalg.norm(v)
        theta, phi = np.arccos(v[2] / r), np.arctan2(v[1], v[0])
        for l in range(LMAX + 1):
            for m in range(-l, l + 1):
                expected = scipy_special.sph_harm_y(l, m, theta, phi)
                assert abs(got[lapw.lm_index(l, m)] - expected) < 1e-13, (l, m)


def test_the_t4pil_prefactor_is_four_pi_minus_i_to_the_l():
    """The trap study §6 names for this item, asserted rather than described."""
    v = jnp.asarray([0.3, -0.7, 1.1])
    with_prefactor = np.asarray(lapw.spherical_harmonics(LMAX, v, t4pil=True))
    without = np.asarray(lapw.spherical_harmonics(LMAX, v, t4pil=False))
    for l in range(LMAX + 1):
        block = slice(l * l, (l + 1) ** 2)
        ratio = with_prefactor[block] / without[block]
        assert np.allclose(ratio, 4 * np.pi * (-1j) ** l, rtol=1e-12), l


def test_the_muffin_tin_function_meets_the_plane_wave_at_the_sphere(setup):
    """Continuity itself: the only forward check that sees the ASSEMBLY.

    SciPy validates the special functions but nothing of 1/sqrt(Omega), the conjugation,
    the t4pil prefactor, the packing or the solve.  This does, because it IS the
    definition -- and it is deliberately not written as `D A = b`, which is
    self-consistent (solve with a matrix, multiply by the same matrix) and therefore
    blind to whether D's rows are the derivative order or the APW index.  The radial
    family u_jl(r) = r^(l+j-1) is known in closed form, so the reconstruction is
    evaluated from the power rule directly.
    """
    assert phase0c.experiment_matching_condition(setup) < 1e-10


def test_the_continuity_check_catches_a_transposed_derivative_matrix(setup):
    """... and here is the convention it exists to pin.

    Handing `match` the transpose of D is exactly the kind of misreading a transcription
    makes, and the dmatch identity cannot see it -- measured 3e-16 on the wrong
    coefficients.  The continuity check fails by O(1).
    """
    flipped = dict(setup)
    flipped["matrices"] = [m.T for m in setup["matrices"]]
    assert phase0c.experiment_matching_condition(flipped) > 1e-2
    for row in phase0c.experiment_dmatch(flipped, directions=(0,)):
        assert row["error"] < 1e-13            # exact, and exactly as wrong


def test_the_harmonic_gradient_is_nan_on_the_z_axis_and_finite_off_it(setup):
    """`genylmv`'s own singularity, pinned in isolation.

    For m != 0 the phase e^{i m phi} has no derivative where phi is undefined, and
    sin(theta) = sqrt(1 - cos^2 theta) has an infinite one at the pole.  The result is
    NaN, which is the GOOD outcome -- audible rather than a plausible finite number.

    This is still the honest behaviour of `spherical_harmonics`, which stays the
    transcription of `genylmv` that item 0c checks.  It is no longer the behaviour of
    `match`: Phase 1f measured this pole taking down the k-derivative of the whole
    assembly at Gamma, so `match` now goes through `solid_harmonics` instead, and the
    two tests below are the ones that pin the fixed path.
    """
    harmonic = lambda v: jnp.real(lapw.spherical_harmonics(2, v, t4pil=False)[
        lapw.lm_index(1, 1)])
    on_axis = np.asarray(jax.grad(harmonic)(jnp.asarray([0.0, 0.0, 1.0])))
    assert np.all(np.isnan(on_axis))
    off_axis = np.asarray(jax.grad(harmonic)(jnp.asarray([0.3, -0.7, 1.1])))
    assert np.all(np.isfinite(off_axis)) and np.linalg.norm(off_axis) > 1e-3


def test_the_dmatch_identity_in_forward_mode(setup):
    """Item 0c's stated criterion: machine precision, no tolerance argument needed."""
    for row in phase0c.experiment_dmatch(setup):
        assert row["error"] < 1e-13, row["direction"]
        assert row["magnitude"] > 1e-3        # the coefficients are not all ~zero


def test_the_dmatch_identity_in_reverse_mode(setup):
    """Forces come from reverse mode, and forward-vs-reverse costs nothing to check."""
    assert phase0c.experiment_reverse(setup)["error"] < 1e-12


def test_forgetting_the_prefactor_still_passes_the_dmatch_identity(setup):
    """Why the gradient check alone cannot validate the port.

    The t4pil error is a fixed complex factor per l, so it commutes with d/dr_alpha and
    the exact identity holds for the wrong coefficients just as well.  A green 0c is
    therefore necessary and not sufficient -- which is the whole reason the forward
    checks above exist.
    """
    ratios = phase0c.experiment_t4pil(setup)
    assert abs(ratios[0] - 4 * np.pi) < 1e-8            # l=0 is unaffected
    assert abs(ratios[1] - 4 * np.pi * 1j) < 1e-8       # ... l=1 is off by 4 pi i
    tangent = jnp.zeros(3).at[0].set(1.0)
    primal, jvp = jax.jvp(lambda r: phase0c.apwalm(setup, r, t4pil=False),
                          (setup["atposc"],), (tangent,))
    exact = lapw.dmatch(setup["vgkc"], primal, 0)
    assert float(jnp.max(jnp.abs(jvp - exact)) / jnp.max(jnp.abs(exact))) < 1e-13


def test_the_gkvector_cutoff_is_a_mask_not_a_compaction(setup):
    """§5's `gk_mask`: a traced program cannot have a data-dependent ngk."""
    vgc = jnp.asarray(np.random.default_rng(1).normal(size=(50, 3)))
    vgkc, gkc, mask = lapw.gkvectors(vgc, jnp.zeros(3), 1.0)
    assert vgkc.shape == (50, 3) and mask.shape == (50,)
    assert bool(jnp.all((gkc < 1.0) == mask))


@SLOW
def test_at_the_full_g_plus_k_set():
    """lmax=8 and ~700 G+k vectors, the size a real run has."""
    big = phase0c.build(lmax=8, ncell=4, gkmax=7.0)
    assert big["gkc"].shape[0] > 500
    assert phase0c.experiment_matching_condition(big) < 1e-10
    for row in phase0c.experiment_dmatch(big):
        assert row["error"] < 1e-13


# ---------------------------------------------------------------------------
# The two removable poles, and the route that avoids them (Phase 1f)
# ---------------------------------------------------------------------------


def test_solid_harmonics_are_the_spherical_ones_times_r_to_the_l(setup):
    """Ties the new function to the already-verified one, off-axis where both hold.

    `solid_harmonics` is a separate function rather than a refactor of
    `spherical_harmonics`, deliberately: that one is what item 0c and the element-wise
    `apwalm` comparison check, and reordering its products would move its last bits for
    nothing. This identity is what stands in for having refactored them together.
    """
    lmax = 6
    rng = np.random.default_rng(0)
    worst = 0.0
    for scale in (1e-3, 1.0, 30.0):
        for _ in range(6):
            v = jnp.asarray(rng.normal(size=3) * scale)
            radius = float(jnp.linalg.norm(v))
            spherical = np.asarray(lapw.spherical_harmonics(lmax, v))
            solid = np.asarray(lapw.solid_harmonics(lmax, v))
            powers = np.concatenate(
                [np.full(2 * l + 1, radius ** l) for l in range(lmax + 1)])
            reference = spherical * powers
            worst = max(worst,
                        np.abs(solid - reference).max() / np.abs(reference).max())
    assert worst < 1e-14


def test_the_scaled_bessel_agrees_with_the_recurrence_where_both_are_accurate():
    """P_{l,io}(x) = j_l^(io)(x) x^(io-l), across its own branch cut.

    The comparison has to be made where BOTH routes are good. Below x ~ 1e-3 the
    recurrence-and-divide route is the inaccurate one -- measured, it is wrong by 100%
    at l=6, order=2, x=1e-6, because it forms j_6'' ~ 1e-27 and divides by x^4 -- which
    is the entire reason the series branch exists. So the series is checked against the
    recurrence at moderate x (by raising its own threshold so it is used there), and
    against the l=0 closed form.
    """
    lmax = 6
    for order in (0, 1, 2):
        for x in (0.2, 0.5, 1.0, 2.0, 3.0):
            series = np.asarray(lapw.spherical_bessel_scaled(
                lmax, jnp.asarray(x ** 2), order, threshold=5.0, terms=30))
            recurrence = np.asarray(lapw.spherical_bessel_scaled(
                lmax, jnp.asarray(x ** 2), order))
            assert np.abs(series - recurrence).max() / np.abs(recurrence).max() < 1e-10

    x = 0.37
    sine, cosine = np.sin(x), np.cos(x)
    closed = {0: sine / x,
              1: (cosine / x - sine / x ** 2) * x,
              2: (-sine / x - 2 * cosine / x ** 2 + 2 * sine / x ** 3) * x ** 2}
    for order, expected in closed.items():
        got = float(np.asarray(lapw.spherical_bessel_scaled(
            lmax, jnp.asarray(x ** 2), order, threshold=5.0, terms=30))[0])
        assert got == pytest.approx(expected, rel=1e-13)


def test_match_is_differentiable_in_k_at_a_zero_length_basis_vector():
    """The fix, at the configuration that broke: a basis containing G+k = 0.

    Both poles are exercised at once. The origin is on the z-axis (so the harmonic's
    direction is undefined) AND has |G+k| = 0 (so sqrt' is infinite), and the second is
    invisible until the first is fixed -- which is why this asserts a finite tangent
    rather than merely a finite harmonic. Central finite differences of the same
    function are the control: `match` is genuinely smooth here, so they converge.
    """
    lmax, rmt, omega = 4, 2.0, 270.0
    gvec = jnp.asarray([[0.0, 0.0, 0.0],          # both poles
                        [0.0, 0.0, 0.61],         # z-axis, nonzero length
                        [0.61, 0.0, 0.0],
                        [-0.3, 0.42, -0.61]])
    matrices = [jnp.asarray(phase0c.radial_derivative_matrix(l, 2 if l <= 2 else 1, rmt))
                for l in range(lmax + 1)]
    atposc = jnp.asarray([0.7, -1.1, 0.4])
    direction = jnp.asarray([0.3, -0.5, 0.8])

    def coefficients(kc):
        return lapw.match(lmax, gvec + kc[None, :], atposc, matrices, rmt, omega)

    origin = jnp.zeros(3)
    value, tangent = jax.jvp(coefficients, (origin,), (direction,))
    assert np.isfinite(np.asarray(value)).all()
    assert np.isfinite(np.asarray(tangent)).all()
    assert np.abs(np.asarray(tangent)).max() > 1e-6

    for step in (1e-4, 1e-5):
        difference = (np.asarray(coefficients(origin + step * direction))
                      - np.asarray(coefficients(origin - step * direction))) / (2 * step)
        error = np.abs(difference - np.asarray(tangent)).max()
        assert error < 1e-6 * np.abs(np.asarray(tangent)).max()
