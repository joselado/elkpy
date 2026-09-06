r"""Phase 2 of the JAX port: the interstitial real-space grid.

Elk keeps interstitial functions on a real-space FFT grid `ngridg` and moves
them to and from :math:`\mathbf G`-space with `zfftifc`.  Two of its
conventions have to be reproduced exactly by anything comparing against an
exported array, and neither is visible in the function being compared.

**`igfft` is not the identity.**  Elk's G-vectors are held in a list sorted by
:math:`|\mathbf G|` (`ivg`), and `igfft` maps a position in that list to the
position in the FFT array.  So "the first `n` G-vectors" is a SPHERE in
reciprocal space, not a corner of the box.

**`vxcir` has been filtered.**  `potks.f90` calls `trimrfg` on it, which zeroes
every Fourier component with :math:`|\mathbf G| > 2\,k_{\max}` -- i.e. every
G-vector whose index exceeds `ngvc`.  Without reproducing that, a pointwise
comparison of the exchange-correlation functional against Elk's own `vxcir`
stops at the trimmed content: measured 2.5e-5 relative on bulk Si, which looks
exactly like a mediocre transcription and is not one.
"""

import numpy as np

import jax.numpy as jnp


def reciprocal_vectors(groundstate):
    r"""The Cartesian :math:`\mathbf G` at every point of the FFT grid,
    laid out in the FFT array's own order.

    `vgc` in the export covers only the first `ngvec` entries of Elk's sorted
    G-list, and the grid has `ngtot > ngvec` slots; the rest are recovered
    from the integer vectors `ivg` and the reciprocal lattice.  `igfft` is
    what puts a list entry in its FFT slot, and it is not the identity --
    Elk's list is sorted by :math:`|\mathbf G|`, so "the first n G-vectors"
    is a sphere and not a corner of the box.
    """
    ngtot = int(groundstate["ngtot"])
    bvec = np.asarray(groundstate["bvec"])
    ivg = np.asarray(groundstate["ivg"]).astype(float)
    igfft = np.asarray(groundstate["igfft"]) - 1
    out = np.zeros((3, ngtot))
    out[:, igfft] = bvec @ ivg
    return out


def laplacian(values, groundstate):
    r""":math:`\nabla^2 f` on the interstitial grid, spectrally.

    Exact for a function the grid represents exactly, which `vclir` is not
    near a muffin-tin sphere -- see `tests/test_calculation_poisson.py`.
    """
    g2 = jnp.asarray((reciprocal_vectors(groundstate) ** 2).sum(axis=0))
    ngridg = groundstate["ngridg"]
    return to_real(-g2 * to_reciprocal(values, ngridg), ngridg)


def gradient(values, groundstate):
    r""":math:`\nabla f` on the interstitial grid, spectrally: `(3, ngtot)`.

    The adjoint of this operation is the divergence, which is what makes a
    GGA's :math:`-\nabla\cdot(\partial(\rho\varepsilon)/\partial\nabla\rho)`
    term fall out of `jax.grad` with nothing hand-coded.
    """
    gvec = jnp.asarray(reciprocal_vectors(groundstate))
    ngridg = groundstate["ngridg"]
    spectrum = to_reciprocal(values, ngridg)
    return jnp.stack([to_real(1j * gvec[a] * spectrum, ngridg)
                      for a in range(3)])


def to_reciprocal(values, ngridg):
    """Real-space FFT grid -> the complex G-space array, in Elk's layout."""
    shape = tuple(int(n) for n in ngridg)
    return jnp.reshape(jnp.fft.fftn(jnp.reshape(
        jnp.asarray(values), shape, order="F")), (-1,), order="F")


def to_real(values, ngridg):
    """The inverse of `to_reciprocal`, discarding the imaginary part."""
    shape = tuple(int(n) for n in ngridg)
    return jnp.reshape(jnp.real(jnp.fft.ifftn(jnp.reshape(
        jnp.asarray(values), shape, order="F"))), (-1,), order="F")


def trim(values, groundstate):
    r"""`trimrfg.f90`: zero every Fourier component with
    :math:`|\mathbf G| > 2\,k_{\max}`.

    `groundstate` is a GROUNDSTATE export, from which `ngridg`, `igfft` and
    `ngvc` are read.  Applied to the exchange-correlation potential, this is
    what takes a pointwise comparison against Elk's `vxcir` from 2.5e-5
    relative to machine precision.
    """
    ngridg = groundstate["ngridg"]
    keep = np.zeros(int(groundstate["ngtot"]), dtype=bool)
    keep[np.asarray(groundstate["igfft"])[:int(groundstate["ngvc"])] - 1] = True
    spectrum = to_reciprocal(values, ngridg)
    return to_real(jnp.where(jnp.asarray(keep), spectrum, 0.0), ngridg)
