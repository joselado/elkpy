r"""Phase 2 of the JAX port: integrals over the unit cell.

Elk splits every real-space function into a packed muffin-tin part and a
value on the interstitial FFT grid, so the cell integral of one
(`rfint.f90` + `rfmtint.f90`) and the inner product of two (`rfinp.f90`) are
the two operations everything downstream is expressed in: the total charge,
every energy component, and any check that a density is what it should be.

.. math::
   \int_\Omega f\,d^3r
     = \frac{\Omega}{N_{\rm FFT}}\sum_i f_i\,\Theta_i
     + \sqrt{4\pi}\sum_\alpha\sum_r w^{(2)}_r f^{\alpha}_{00}(r)

The two conventions worth stating.  Only the $\ell=0$ coefficient of the
muffin-tin expansion contributes to a plain integral, with the
$\sqrt{4\pi}=1/Y_{00}$ from $\int R_{00}\,d\Omega$; and the interstitial sum is
weighted by the characteristic function, so the muffin-tin spheres are not
counted twice.  An inner product, by contrast, sums over **every** $(\ell,m)$,
because the real spherical harmonics are orthonormal.

`wr2mt` is Elk's own quadrature weight array from `wsplint`, a Simpson-like
rule -- not $r^2dr$ -- and is taken from the export rather than rebuilt.
"""

import numpy as np

import jax.numpy as jnp

from elkpy.parsers.eigenstates import unpack_muffin_tin

_SQRT_FOURPI = float(np.sqrt(4.0 * np.pi))


def _shapes(groundstate, ias):
    is_ = int(np.asarray(groundstate["idxis"])[ias]) - 1
    return (is_, int(np.asarray(groundstate["nrmt"])[is_]),
            int(np.asarray(groundstate["nrmti"])[is_]))


def muffin_tin_integral(packed, groundstate, ias):
    """`rfmtint.f90`: one sphere's contribution to a cell integral."""
    is_, nr, nri = _shapes(groundstate, ias)
    lmmaxi = int(groundstate["lmmaxi"])
    lmmaxo = int(groundstate["lmmaxo"])
    weights = jnp.asarray(np.asarray(groundstate["wr2mt"])[is_, :nr])
    packed = jnp.asarray(packed)
    inner = packed[:lmmaxi * nri:lmmaxi]
    start = lmmaxi * nri
    outer = packed[start:start + lmmaxo * (nr - nri):lmmaxo]
    return _SQRT_FOURPI * jnp.sum(weights * jnp.concatenate([inner, outer]))


def cell_integral(muffin_tin, interstitial, groundstate):
    """`rfint.f90`: the integral of one function over the unit cell.

    `muffin_tin` is `(natmtot, npmtmax)` in Elk's packing, `interstitial` is
    `(ngtot,)` on the FFT grid.
    """
    omega = float(groundstate["omega"])
    ngtot = int(groundstate["ngtot"])
    total = (omega / ngtot) * jnp.sum(
        jnp.asarray(interstitial) * jnp.asarray(groundstate["cfunir"]))
    for ias in range(int(groundstate["natmtot"])):
        total = total + muffin_tin_integral(
            jnp.asarray(muffin_tin)[ias], groundstate, ias)
    return total


def cell_inner_product(mt_a, ir_a, mt_b, ir_b, groundstate):
    r"""`rfinp.f90`: :math:`\int_\Omega f\,g\,d^3r`.

    Sums over EVERY $(\ell,m)$ in the muffin tin -- the real spherical
    harmonics are orthonormal, so the angular integral of the product is the
    coefficient-wise sum, not just the $\ell=0$ term a plain integral needs.
    """
    omega = float(groundstate["omega"])
    ngtot = int(groundstate["ngtot"])
    total = (omega / ngtot) * jnp.sum(
        jnp.asarray(ir_a) * jnp.asarray(ir_b)
        * jnp.asarray(groundstate["cfunir"]))
    lmmaxi = int(groundstate["lmmaxi"])
    lmmaxo = int(groundstate["lmmaxo"])
    mt_a, mt_b = jnp.asarray(mt_a), jnp.asarray(mt_b)
    for ias in range(int(groundstate["natmtot"])):
        is_, nr, nri = _shapes(groundstate, ias)
        weights = jnp.asarray(np.asarray(groundstate["wr2mt"])[is_, :nr])
        dense_a = _dense(mt_a[ias], nr, nri, lmmaxi, lmmaxo)
        dense_b = _dense(mt_b[ias], nr, nri, lmmaxi, lmmaxo)
        total = total + jnp.sum(weights * jnp.sum(dense_a * dense_b, axis=1))
    return total


def _dense(packed, nr, nri, lmmaxi, lmmaxo):
    """`unpack_muffin_tin`, but traceable.

    The parser's version calls `np.asarray`, which is right for reading a
    query's output and wrong inside anything `jax.grad` has to see through --
    and the port's whole justification is differentiability, so a module that
    silently refuses to be differentiated is a defect rather than a
    limitation.  Found by differentiating the total energy with respect to the
    density (§2k).
    """
    inner = packed[:lmmaxi * nri].reshape(nri, lmmaxi)
    outer = packed[lmmaxi * nri:lmmaxi * nri + lmmaxo * (nr - nri)]
    top = jnp.zeros((nri, lmmaxo), dtype=packed.dtype).at[:, :lmmaxi].set(inner)
    return jnp.concatenate([top, outer.reshape(nr - nri, lmmaxo)], axis=0)
