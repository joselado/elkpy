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
