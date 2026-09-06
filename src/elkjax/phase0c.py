r"""Phase 0c: is the LAPW position dependence AD-tractable?

Study §6 item 0c.  Run with::

    PYTHONPATH=src taskset -c 0-3 python3 -m elkjax.phase0c

The kill criterion is agreement to machine precision against
:math:`\partial A/\partial r_{\alpha,p}=i({\bf G+p})_pA`, which is what
``dmatch.f90`` computes.  That identity is *exact* — the atomic position enters
``match`` through the single factor :math:`e^{i({\bf G+p})\cdot{\bf r}_\alpha}` and
nowhere else — so unlike every other item in Phase 0 this one needs no reference
implementation, no finite differences and no tolerance argument.  It also has no SCF in
it: the radial derivative matrices are inputs.

"If this fails nothing downstream is worth debugging" (§6).  It does not fail; what the
sweep below is really for is the *forward* half, since an exact derivative of the wrong
function is worthless.  ``spherical_harmonics`` and ``spherical_bessel`` are therefore
checked against SciPy — an independent implementation that knows nothing about Elk — and
the ``t4pil`` prefactor is checked by the one test that can see it.
"""

import math

import numpy as np

import jax
import jax.numpy as jnp

from . import lapw, memory

LMAX = 8


def build(lmax=LMAX, ncell=4, gkmax=7.0, rmt=2.0, alat=10.26, seed=0,
          orders=None):
    r"""A :math:`{\bf G+k}` set and a set of radial derivative matrices.

    ``alat`` is bulk silicon's lattice constant in Bohr, and ``gkmax``/``rmt`` are its
    usual magnitudes, so :math:`|{\bf G+k}|R_{\rm MT}` spans the range a real run does
    — which matters, because that is the argument of the Bessel functions and it is
    where their branch switch lives.
    """
    rng = np.random.default_rng(seed)
    b = 2.0 * np.pi / alat
    grid = np.arange(-ncell, ncell + 1)
    ivg = np.array([(i, j, k) for i in grid for j in grid for k in grid], dtype=float)
    vgc = jnp.asarray(ivg * b)
    vkc = jnp.asarray(rng.normal(scale=0.1, size=3))
    vgkc, gkc, mask = lapw.gkvectors(vgc, vkc, gkmax)
    keep = np.asarray(mask)
    vgkc, gkc = vgkc[keep], gkc[keep]

    if orders is None:                       # Elk's usual APW+lo pattern: mostly 1
        orders = [2 if l <= 2 else 1 for l in range(lmax + 1)]
    matrices = [jnp.asarray(radial_derivative_matrix(l, order, rmt))
                for l, order in enumerate(orders)]
    atposc = jnp.asarray(rng.normal(scale=2.0, size=3))
    omega = alat ** 3 / 4.0
    return dict(lmax=lmax, vgkc=vgkc, gkc=gkc, atposc=atposc, matrices=matrices,
                rmt=rmt, omega=omega, orders=orders)


def radial_basis(l, order):
    r"""A stand-in APW radial family with derivatives known in closed form.

    :math:`u_{jl}(r)=r^{\,l+j-1}`, :math:`j=1\ldots M_l`.  Not a real APW radial
    function — those solve the radial Schrödinger equation at the linearisation energy
    and are Phase 1 — but it has the two properties this item needs: the right
    :math:`r^l` behaviour at the origin, and derivatives that can be written down.
    Writing them down is the whole point: it is what turns :math:`DA=b` from a statement
    about a matrix into a statement about a **function**.
    """
    return [l + j for j in range(order)]          # the exponents p_j = l + j - 1


def radial_derivative_matrix(l, order, rmt):
    r""":math:`D_{ij}=d^{\,i-1}u_{jl}/dr^{\,i-1}|_{R}`, in ``match.f90``'s own layout.

    Rows are the derivative order, columns the APW order :math:`j` — read off
    ``match.f90``'s ``a(1,jo)=apwfr(nr,1,jo,l,ias)`` and
    ``a(io,jo)=polynm(io-1,...)``.  Getting this transposed is the one convention a
    self-consistent :math:`DA=b` check cannot see, which is why :func:`radial_basis`
    exists.
    """
    exponents = radial_basis(l, order)
    matrix = np.zeros((order, order))
    for i in range(order):
        for j, p in enumerate(exponents):
            if p >= i:
                matrix[i, j] = math.prod(range(p - i + 1, p + 1)) * rmt ** (p - i)
    return matrix


def apwalm(setup, atposc=None, t4pil=True):
    """``match`` at this setup, optionally at a displaced atomic position."""
    return lapw.match(setup["lmax"], setup["vgkc"], setup["gkc"],
                      setup["atposc"] if atposc is None else atposc,
                      setup["matrices"], setup["rmt"], setup["omega"], t4pil)


def experiment_dmatch(setup, directions=(0, 1, 2)):
    r"""``jax.jvp(match)`` against ``dmatch``'s :math:`i({\bf G+p})_pA`."""
    rows = []
    for direction in directions:
        tangent = jnp.zeros(3).at[direction].set(1.0)
        primal, jvp = jax.jvp(lambda r: apwalm(setup, r), (setup["atposc"],), (tangent,))
        exact = lapw.dmatch(setup["vgkc"], primal, direction)
        scale = float(jnp.max(jnp.abs(exact)))
        rows.append(dict(direction=direction,
                         error=float(jnp.max(jnp.abs(jvp - exact))) / scale,
                         magnitude=scale))
    return rows


def experiment_reverse(setup, seed=1):
    r"""The same identity in **reverse** mode, contracted against a random cotangent.

    Forward mode is what ``dmatch`` mirrors, but forces come from reverse mode, and for
    a scalar-in scalar-out contraction the two must agree exactly — the check that cost
    nothing and would have caught Phase 0b's mistake a week earlier.
    """
    rng = np.random.default_rng(seed)
    shape = apwalm(setup).shape
    weight = jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))
    loss = lambda r: jnp.real(jnp.vdot(weight, apwalm(setup, r)))
    reverse = np.asarray(jax.grad(loss)(setup["atposc"]))
    primal = apwalm(setup)
    forward = np.array([
        float(jnp.real(jnp.vdot(weight, lapw.dmatch(setup["vgkc"], primal, p))))
        for p in range(3)])
    return dict(reverse=reverse, exact=forward,
                error=float(np.max(np.abs(reverse - forward)) / np.max(np.abs(forward))))


def experiment_scipy(lmax=LMAX):
    """The forward half: an exact derivative of the wrong function is worthless."""
    from scipy.special import spherical_jn, sph_harm_y

    rng = np.random.default_rng(0)
    bessel = 0.0
    for x in (1e-10, 1e-6, 0.01, 0.5, 1.0, 3.0, 7.0, 8.0, 12.0, 20.0, 40.0):
        got = np.asarray(lapw.spherical_bessel(lmax, jnp.asarray(x)))
        reference = spherical_jn(np.arange(lmax + 1), x)
        bessel = max(bessel, np.max(np.abs(got - reference)
                                    / np.maximum(np.abs(reference), 1e-30)))
    derivative = 0.0
    for x in (0.5, 3.0, 12.0):
        got = np.asarray(lapw.spherical_bessel_derivative(lmax, jnp.asarray(x), 1))
        reference = spherical_jn(np.arange(lmax + 1), x, derivative=True)
        derivative = max(derivative, np.max(np.abs(got - reference)
                                            / np.maximum(np.abs(reference), 1e-30)))
    harmonic = 0.0
    for _ in range(5):
        v = rng.normal(size=3)
        got = np.asarray(lapw.spherical_harmonics(lmax, jnp.asarray(v), t4pil=False))
        r = np.linalg.norm(v)
        theta, phi = np.arccos(v[2] / r), np.arctan2(v[1], v[0])
        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                harmonic = max(harmonic, abs(got[lapw.lm_index(l, m)]
                                             - sph_harm_y(l, m, theta, phi)))
    return dict(bessel=bessel, derivative=derivative, harmonic=harmonic)


def experiment_matching_condition(setup):
    r"""Continuity itself: does the muffin-tin function meet the plane wave at :math:`R`?

    This is the forward check with teeth, and it is deliberately not written as
    :math:`DA=b`.  That form is self-consistent — it solves with a matrix and then
    multiplies by the same matrix — so it cannot see whether :math:`D`'s **rows** are
    the derivative order or the APW index.  Here the radial family
    :math:`u_{jl}(r)=r^{\,l+j-1}` is known in closed form, so the reconstruction

    .. math::

        \phi^{(i)}_{lm}(R)=\sum_j A_{jlm}\,\frac{d^{\,i}}{dr^{\,i}}r^{\,l+j-1}
        \bigg|_{R}
        \overset{!}{=}
        \frac{4\pi i^l}{\sqrt\Omega}|{\bf G+p}|^{i}j^{(i)}_l(|{\bf G+p}|R)
        e^{i({\bf G+p})\cdot{\bf r}_\alpha}Y^*_{lm}(\widehat{{\bf G+p}})

    is evaluated from the power rule directly, never by reusing the matrix ``match`` was
    handed.  A transposed convention anywhere then fails.  The right-hand side is the
    Rayleigh expansion of :math:`e^{i({\bf G+p})\cdot{\bf r}}/\sqrt\Omega` about the
    atom, rebuilt from SciPy with the :math:`4\pi i^l` written out rather than hidden in
    ``genylmv``.  Orders up to 2, which is what the fixture uses.
    """
    from scipy.special import spherical_jn, sph_harm_y

    coefficients = np.asarray(apwalm(setup))
    vgkc = np.asarray(setup["vgkc"])
    gkc = np.asarray(setup["gkc"])
    atposc = np.asarray(setup["atposc"])
    rmt, omega, lmax = setup["rmt"], setup["omega"], setup["lmax"]
    scale = 1.0 / np.sqrt(omega)
    phase = np.exp(1j * (vgkc @ atposc))
    argument = gkc * rmt

    worst = 0.0
    for l, order in enumerate(setup["orders"]):
        if order > 2:
            raise ValueError("this check is written for APW orders 1 and 2")
        bessel = [spherical_jn(l, argument),
                  spherical_jn(l, argument, derivative=True)][:order]
        plane_wave = np.stack([bessel[i] * gkc ** i for i in range(order)])
        exponents = radial_basis(l, order)
        for m in range(-l, l + 1):
            harmonics = np.array([
                sph_harm_y(l, m, np.arccos(np.clip(v[2] / g, -1.0, 1.0)),
                           np.arctan2(v[1], v[0]))
                for v, g in zip(vgkc, gkc)])
            target = (4.0 * np.pi * (1j) ** l * scale) * plane_wave * (
                phase * np.conj(harmonics))[None, :]
            amplitudes = coefficients[:, :order, lapw.lm_index(l, m)]     # (ngk, order)
            reconstructed = np.zeros_like(target)
            for i in range(order):
                for j, p in enumerate(exponents):
                    if p >= i:
                        factor = math.prod(range(p - i + 1, p + 1)) * rmt ** (p - i)
                        reconstructed[i] += factor * amplitudes[:, j]
            worst = max(worst, np.max(np.abs(reconstructed - target))
                        / max(np.max(np.abs(target)), 1e-30))
    return worst


def experiment_t4pil(setup):
    r"""What forgetting ``genylmv``'s :math:`4\pi(-i)^l` actually does.

    Study §6 names this trap for item 0c specifically.  It is worth measuring rather
    than describing, because the wrong result is not obviously wrong: every :math:`l`
    block is off by a fixed complex factor, so the array is finite, smooth, correctly
    shaped, and **exactly right for** :math:`l=0` — and the ``dmatch`` identity holds
    for it just as well, since the error commutes with the position derivative.

    The measured ratio is :math:`\overline{4\pi(-i)^l}=4\pi i^l`, not
    :math:`4\pi(-i)^l`, because ``match`` uses ``conjg(ylmgp)``.  That conjugation is
    exactly where ``match.f90``'s own documented :math:`b_i\propto4\pi i^l` comes from.
    """
    with_prefactor = apwalm(setup)
    without = apwalm(setup, t4pil=False)
    ratios = []
    for l in range(setup["lmax"] + 1):
        block = slice(l * l, (l + 1) ** 2)
        a, b = with_prefactor[:, 0, block], without[:, 0, block]
        keep = jnp.abs(b) > 1e-12
        ratios.append(complex(jnp.mean(jnp.where(keep, a / jnp.where(keep, b, 1.0), 0.0))
                              * b.size / jnp.sum(keep)))
    return ratios


def main():
    memory.limit_address_space(16.0)
    setup = build()
    ngk = setup["gkc"].shape[0]
    print(f"== setup: lmax={setup['lmax']}, {ngk} G+k vectors, "
          f"|G+k|R in [{float(setup['gkc'].min()) * setup['rmt']:.2f}, "
          f"{float(setup['gkc'].max()) * setup['rmt']:.2f}], "
          f"APW orders {setup['orders']} ==")

    print("\n== forward half: the special functions against SciPy ==")
    s = experiment_scipy()
    print(f"  j_l(x)      worst relative {s['bessel']:.2e}")
    print(f"  dj_l/dx     worst relative {s['derivative']:.2e}   (from jax.jacfwd)")
    print(f"  Y_lm        worst absolute {s['harmonic']:.2e}")
    print(f"  continuity  worst relative {experiment_matching_condition(setup):.2e}   "
          f"(phi(R), phi'(R) vs the plane wave, from SciPy)")

    print("\n== 0c: jax.jvp(match) vs dmatch's i(G+p)_p A ==")
    for row in experiment_dmatch(setup):
        print(f"  direction {row['direction']}: relative error {row['error']:.2e}  "
              f"(|A| up to {row['magnitude']:.3e})")
    reverse = experiment_reverse(setup)
    print(f"  reverse mode, contracted: relative error {reverse['error']:.2e}")

    print("\n== the t4pil trap: ratio of correct to prefactor-free, per l ==")
    for l, ratio in enumerate(experiment_t4pil(setup)):
        print(f"  l={l}: {ratio.real:+.4f} {ratio.imag:+.4f}i   "
              f"(conj(4pi(-i)^l) = {np.conj(4 * np.pi * (-1j) ** l):+.4f})")

    print(f"\npeak RSS {memory.peak_rss_bytes() / memory.GB:.2f} GB")


if __name__ == "__main__":
    main()
