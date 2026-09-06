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


def test_the_matching_condition_holds(setup):
    """D A = b, with b rebuilt from SciPy and the 4 pi i^l written out explicitly.

    This is the only forward check that sees the assembly rather than the pieces --
    1/sqrt(Omega), the conjugation, the prefactor, the packing and the solve.
    """
    assert phase0c.experiment_matching_condition(setup) < 1e-10


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
    original = lapw.spherical_harmonics
    try:
        lapw.spherical_harmonics = lambda lmax, v, t4pil=True: original(lmax, v, False)
        for row in phase0c.experiment_dmatch(setup, directions=(0,)):
            assert row["error"] < 1e-13                 # still exact, still wrong
    finally:
        lapw.spherical_harmonics = original


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
