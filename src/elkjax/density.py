r"""Phase 2 item 2b: the interstitial valence density from the eigenvectors.

This is the step that closes the SCF loop's circle.  §§2a-2g go from a density
to a total energy; this goes the other way, from the first-variational
eigenvectors back to a density.  What is here is the **interstitial** half, and
the muffin-tin half is not (see "What this does not do" below).

**The formula.**  For a non-magnetic ground state ``rhomagk``'s interstitial
accumulation is

.. math::

    \rho^{\rm I}({\bf r})=\frac1\Omega\sum_{\bf k}w_{\bf k}\sum_n f_{n\bf k}
    \,\bigl|\psi_{n\bf k}({\bf r})\bigr|^2 ,

with :math:`\psi` built by scattering ``evecfv(igp, n)`` into the **coarse** FFT
grid at ``igfc(igkig(igp))`` and transforming.  In the interstitial an LAPW
state is an exact plane-wave sum, so this is not an approximation: the only
truncation is the basis's own.

**Three things about it are easy to get wrong and are pinned by tests.**

* The density is accumulated on the **coarse** grid (``ngdgc``, ``ngtc``), not
  the fine one the ``GROUNDSTATE`` query's ``rhoir`` lives on.  ``rfirctof``
  interpolates afterwards by zero-padding, so the coarse function is recoverable
  from the fine one *exactly* -- which is what :func:`coarsen` does, and why the
  comparison needs no interpolation of its own.
* ``rhonorm`` then adds a **uniform constant** :math:`(N-N_{\rm calc})/\Omega`,
  not a rescaling.  So the difference from Elk is a constant, and asserting
  *that* is sharper than any tolerance -- a constant is a one-parameter family
  and a pointwise error breaks it.
* The zone sum needs the k-set Elk actually used, weights included.  The
  ``DENSITYK`` query supplies it; rebuilding it from ``ngridk`` would differ
  whenever ``reducek`` is nonzero, which is the default.

**What this does not do.**  The muffin-tin half needs ``wfmtsv`` on the coarse
*radial* mesh, then ``rhomagsh``, ``rfmtctof`` and ``rhocore`` -- which brings in
the core states and so ``gencore``/``rdirac``.  None of that is here.  Nor is the
magnetic case: ``rmk1``/``rmk2`` are not transcribed, only ``rmk3``.
"""

import numpy as np

import jax.numpy as jnp

__all__ = ["interstitial_wavefunction", "interstitial_density", "coarsen",
           "normalisation_offset"]

EPSOCC = 1.0e-8


def _inverse_fft(values, grid):
    """Elk's ``zfftifc(3, n, +1, z)``: sign +1 and no normalisation."""
    return jnp.fft.ifftn(jnp.asarray(values).reshape(grid, order="F")
                         ).reshape(-1, order="F") * np.prod(grid)


def _forward_fft(values, grid):
    """Elk's ``zfftifc(3, n, -1, z)``: sign -1 with a 1/N."""
    return jnp.fft.fftn(jnp.asarray(values).reshape(grid, order="F")
                        ).reshape(-1, order="F") / np.prod(grid)


def interstitial_wavefunction(coefficients, igkig, densityk):
    r""":math:`\psi_n({\bf r})` on the coarse interstitial grid.

    ``coefficients`` is one state's ``evecfv(1:ngk, n)``; ``igkig`` maps each
    :math:`{\bf G+k}` to the global :math:`{\bf G}` index, and ``igfc`` maps
    that to a slot of the coarse FFT array.  The two-step indirection is the
    part a transcription gets wrong, and it is Elk's own: ``igkig`` is a
    property of the k-point, ``igfc`` of the grid.
    """
    grid = tuple(int(n) for n in densityk["ngdgc"])
    igfc = np.asarray(densityk["igfc"])
    slots = igfc[np.asarray(igkig) - 1] - 1
    array = jnp.zeros(int(densityk["ngtc"]), dtype=complex)
    array = array.at[jnp.asarray(slots)].set(jnp.asarray(coefficients))
    return _inverse_fft(array, grid)


def interstitial_density(densityk, omega, ispn=0):
    r"""``rhomagk``'s ``rmk3`` branch, summed over the zone.

    Returns the coarse-grid interstitial valence density, *before* ``rhonorm``.
    States with :math:`|f|<` ``epsocc`` are skipped exactly as Elk skips them;
    including them would change the result by less than :math:`10^{-8}` times a
    wavefunction modulus, which is above the tolerance this is compared at.
    """
    total = jnp.zeros(int(densityk["ngtc"]))
    for ik in range(int(densityk["nkpt"])):
        weight = float(densityk["wkpt"][ik])
        occupations = np.asarray(densityk["occsv"][ik])
        vectors = densityk["evecfv"][(ik, ispn)]
        igkig = densityk["igkig"][(ik, ispn)]
        for ist, occupation in enumerate(occupations[:vectors.shape[0]]):
            if abs(occupation) < EPSOCC:
                continue
            psi = interstitial_wavefunction(vectors[ist], igkig, densityk)
            total = total + (occupation * weight / omega) * jnp.abs(psi) ** 2
    return total


def coarsen(fine, groundstate, densityk):
    """The exact inverse of ``rfirctof``: the fine grid back to the coarse one.

    ``rfirctof`` transforms the coarse function, copies its ``ngvc`` Fourier
    components into the fine array and zeroes the rest.  So the fine function
    carries no content the coarse one did not, and taking those same components
    back recovers it exactly -- no interpolation, no smoothing, and no
    tolerance introduced by the comparison itself.
    """
    fine_grid = tuple(int(n) for n in groundstate["ngridg"])
    coarse_grid = tuple(int(n) for n in densityk["ngdgc"])
    ngvc = int(densityk["ngvc"])
    igfft = np.asarray(groundstate["igfft"])[:ngvc] - 1
    igfc = np.asarray(densityk["igfc"])[:ngvc] - 1

    spectrum = _forward_fft(fine, fine_grid)
    array = jnp.zeros(int(densityk["ngtc"]), dtype=complex)
    array = array.at[jnp.asarray(igfc)].set(spectrum[jnp.asarray(igfft)])
    return jnp.real(_inverse_fft(array, coarse_grid))


def normalisation_offset(mine, elk_coarse):
    r"""``rhonorm``'s uniform shift, as ``(mean difference, deviation)``.

    ``rhonorm`` adds :math:`(N-N_{\rm calc})/\Omega` to every point rather than
    rescaling, so the difference between a correct transcription and Elk's
    stored density is a *constant*.  The second return value is how far from
    constant it actually is, which is the quantity worth asserting.
    """
    difference = np.asarray(elk_coarse) - np.asarray(mine)
    return float(difference.mean()), float(difference.std())
